# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from typing import Optional, Callable, List, Union

import itertools
import json
import os
import re
import shutil
import time

from openai import AsyncOpenAI, OpenAI

from ipyh.constants import (
    CELL_TIMEOUT_SEC,
    DEFAULT_TOOL_LIST,
    DEFAULT_SYSTEM_PROMPT,
    HELPERS,
    MAX_OUTPUT_CHARS,
    OUTPUT_DIR,
    PRIVATE_PYTHON,
    TASK_PYTHON,
)
from ipyh.normalizers import QUIET_ENV, python_default_normalizer, truncate
from ipyh.skills import load_skills, render_skills

_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")  # colour in IPython tracebacks
_TIMEOUT_LINE = re.compile(
    r"^\s*#\s*timeout:\s*(\d+)"
)  # first line `# timeout: N` sets the cell timeout
_PYTHON_LINE = re.compile(
    r"^\s*(import |from |def |class |print\(|for |if |while |with |try:|[A-Za-z_][\w.]*\s*=[^=])"
)


def _shell_script(code: str, output: str) -> Optional[str]:
    """The cell as a `%%bash` cell if it is a shell script the model wrote as Python, else None.

    Models often write `!cmd` on the first line and then bare shell lines below it. IPython fails
    to compile such a cell, so nothing ran and it is safe to rerun it as bash. We only do so when
    the cell failed with a SyntaxError, starts with `!`, and none of its other lines look like Python.
    """
    lines = code.strip().splitlines()
    looks_like_shell = (
        len(lines) > 1
        and lines[0].lstrip().startswith("!")
        and not any(_PYTHON_LINE.match(line) for line in lines[1:])
    )
    if "SyntaxError" not in output or not looks_like_shell:
        return None
    # Drop the `!` prefixes: in a bash cell every line is already a shell line
    return "%%bash\n" + "\n".join(
        line.lstrip()[1:] if line.lstrip().startswith("!") else line for line in lines
    )


# A harness folder holds this file, next to the files it names (prompt, tool code, skills)
HARNESS_FILE = "harness.json"
# `Agent` settings a harness may carry under its `agent` key
AGENT_SETTINGS = {"max_iteration", "max_tokens", "context_window", "keep_tool_results", "time_limit_sec", "stop_tools"}


class Harness:
    """
    A simple harness object which consists of a ipython shell access. The harness
    has also configurable components such as system prompt, how to process tools, etc.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        tool_processor: Optional[Callable] = python_default_normalizer,
        tools: Optional[List[dict]] = None,
        skills: Optional[List[Union[str, Path]]] = None,
        cell_timeout_sec: int = CELL_TIMEOUT_SEC,
        agent_settings: Optional[dict] = None,
        **kwargs,
    ):
        self.system_prompt = system_prompt
        self.tool_processor = tool_processor
        self.cell_timeout_sec = cell_timeout_sec
        # Passed to the `Agent` built by `harness + model`, e.g. {"max_iteration": 30}
        self.agent_settings = dict(agent_settings or {})
        # Agent Skills: SKILL.md files, skill folders or folders of skill folders. Their names and
        # descriptions are listed in the system prompt, and the model `view()`s a SKILL.md when needed
        self.skills = load_skills(skills or [])

        # Kernel tools: each is a Python function, defined in the kernel by `code`, that the model
        # calls as its own tool, e.g. {"name": "f", "description": ..., "parameters": {...}, "code": "def f(x): ..."}
        self.kernel_tools = {tool["name"]: tool for tool in tools or []}
        self.tool_list = DEFAULT_TOOL_LIST + [
            {"type": "function", "function": {key: tool[key] for key in ("name", "description", "parameters")}}
            for tool in self.kernel_tools.values()
        ]
        self._saved_outputs = itertools.count(1)

    @classmethod
    def from_config(cls, path: Union[str, Path], **kwargs) -> "Harness":
        """A harness from a JSON file, or from a harness folder holding a `harness.json`.

        Keys, all optional: `system_prompt` (or `system_prompt_file`), `tools` (kernel tools; each
        gives its `code` inline or as a `code_file`), `skills` (paths), `cell_timeout_sec`, and
        `agent` (settings for the `Agent`, e.g. `max_iteration`). Relative paths are resolved from
        the JSON file's folder."""
        path = Path(path).expanduser()
        if path.is_dir():
            path = path / HARNESS_FILE
        root = path.parent
        config = json.loads(path.read_text())

        system_prompt = config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        if config.get("system_prompt_file"):
            system_prompt = (root / config["system_prompt_file"]).read_text()
        tools = []
        for tool in config.get("tools") or []:
            tool = dict(tool)
            if "code" not in tool and tool.get("code_file"):
                tool["code"] = (root / tool["code_file"]).read_text()
            tools.append(tool)
        unknown = set(config.get("agent", {})) - AGENT_SETTINGS
        if unknown:
            raise ValueError(f"{path}: unknown agent settings {sorted(unknown)}; use {sorted(AGENT_SETTINGS)}")
        skills = [root / Path(skill).expanduser() for skill in config.get("skills", [])]
        return cls(
            system_prompt=system_prompt,
            tools=tools,
            skills=skills + list(kwargs.pop("skills", None) or []),
            cell_timeout_sec=config.get("cell_timeout_sec", CELL_TIMEOUT_SEC),
            agent_settings=config.get("agent"),
            **kwargs,
        )

    def save(self, folder: Union[str, Path], meta: Optional[dict] = None) -> Path:
        """Writes the harness as a folder that `from_config` loads: `harness.json`, the system
        prompt in `system_prompt.md`, each kernel tool's code in `tools/<name>.py` and a copy of
        each skill folder in `skills/<name>/`. Returns the folder."""
        folder = Path(folder)
        (folder / "tools").mkdir(parents=True, exist_ok=True)
        (folder / "system_prompt.md").write_text(self.system_prompt)
        tools = []
        for tool in self.kernel_tools.values():
            code_file = f"tools/{tool['name']}.py"
            (folder / code_file).write_text(tool["code"])
            tools.append({
                **{key: tool[key] for key in ("name", "description", "parameters")},
                "code_file": code_file,
            })
        for skill in self.skills:
            target = folder / "skills" / skill.name
            if Path(skill.path).parent.resolve() != target.resolve():
                shutil.copytree(Path(skill.path).parent, target, dirs_exist_ok=True)
        config = {
            "system_prompt_file": "system_prompt.md",
            "tools": tools,
            "skills": ["skills"] if self.skills else [],
            "cell_timeout_sec": self.cell_timeout_sec,
            "agent": self.agent_settings,
        }
        if meta:
            config["meta"] = meta
        (folder / HARNESS_FILE).write_text(json.dumps(config, indent=2) + "\n")
        return folder

    def render_system_prompt(self, python: Optional[str]) -> str:
        """The system prompt for a kernel on `python` (None: a private interpreter), then the skills."""
        prompt = self.system_prompt.format(
            python=TASK_PYTHON.format(path=python) if python else PRIVATE_PYTHON
        )
        # Appended after `format`, so braces in a skill's description are kept as they are
        skills = render_skills(self.skills)
        return f"{prompt}\n\n{skills}" if skills else prompt

    def __add__(self, model) -> "Agent":
        if not isinstance(model, (OpenAI, AsyncOpenAI)):
            raise ValueError(
                "The model must be an `openai.OpenAI` or `openai.AsyncOpenAI` client"
            )

        # To avoid circular import
        from ipyh import Agent

        return Agent(model=model, harness=self, **self.agent_settings)

    def execute_tool(
        self,
        tool_call,
        post_process_output: bool = False,
        timeout: Optional[int] = None,
        **tool_kwargs,
    ):
        km, kc = tool_kwargs["km"], tool_kwargs["kc"]
        timeout = timeout or self.cell_timeout_sec

        name = tool_call.get("function", {}).get("name")
        try:
            arguments = json.loads(tool_call["function"]["arguments"] or "{}")
            # A kernel tool call becomes a call of its function; the ipython tool runs the code itself
            code = f"{name}(**{arguments!r})" if name in self.kernel_tools else arguments["code"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            return f"[invalid tool call arguments: {error}]"
        override = _TIMEOUT_LINE.match(code)
        if override:
            timeout = int(override.group(1))

        started = time.monotonic()
        output = self._run_cell(code, timeout, km, kc)
        shell = _shell_script(code, output)
        if shell:
            output = "[ran as %%bash: the cell held shell lines]\n" + self._run_cell(shell, timeout, km, kc)
        elapsed = time.monotonic() - started
        if elapsed > 10:
            output += f"\n[ran in {elapsed:.0f}s]"

        saved_to = None
        if len(output) > MAX_OUTPUT_CHARS:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            saved_to = f"{OUTPUT_DIR}/out-{next(self._saved_outputs)}.log"
            with open(saved_to, "w") as f:
                f.write(output)
        output = truncate(output, saved_to=saved_to)

        if post_process_output:
            output = self.tool_processor(output)

        return output

    def _run_cell(self, code: str, timeout: int, km, kc) -> str:
        out = []

        def hook(msg):
            kind, content = msg["msg_type"], msg["content"]
            if kind == "stream":
                out.append(content["text"])
            elif kind in ("execute_result", "display_data"):
                out.append(content["data"].get("text/plain", "") + "\n")
            elif kind == "error":
                out.append("\n".join(content["traceback"]) + "\n")

        try:
            kc.execute_interactive(
                code, output_hook=hook, timeout=timeout, allow_stdin=False
            )
        except TimeoutError:
            km.interrupt_kernel()
            out.append(
                f"\n[cell timed out after {timeout:.0f}s; kernel interrupted, state kept]\n"
            )
            # The interrupted cell's last messages would be taken for the next cell's: absorb them with an empty one
            try:
                kc.execute_interactive(
                    "", output_hook=lambda msg: None, timeout=30, allow_stdin=False
                )
            except TimeoutError:
                # Still busy: native code (a ctypes call, a C extension) ignores the interrupt, and every
                # later cell would hang behind it. Restart the kernel so the agent can go on.
                self._restart_kernel(km, kc)
                out[-1] = (
                    f"\n[cell timed out after {timeout:.0f}s and ignored the interrupt (stuck in native code); "
                    "the kernel was restarted: variables and imports are gone, the tools are defined again]\n"
                )
        return _ANSI_SGR.sub("", "".join(out)) or "[no output]"

    def kernel_setup(self) -> str:
        """The first cell of a new or restarted kernel: quiet env vars, the file helpers, then the
        kernel tools' definitions."""
        return "\n".join([
            f"import os; os.environ.update({json.dumps(QUIET_ENV)})",
            HELPERS,
            *(tool["code"] for tool in self.kernel_tools.values()),
        ])

    def _restart_kernel(self, km, kc) -> None:
        km.restart_kernel(now=True)
        kc.wait_for_ready(timeout=120)
        try:
            kc.execute_interactive(
                self.kernel_setup(), output_hook=lambda msg: None, timeout=120, allow_stdin=False
            )
        except TimeoutError:
            pass
