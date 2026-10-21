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
from typing import Optional, Callable, List

import itertools
import json
import os
import re
import time

from openai import AsyncOpenAI, OpenAI

from ipyh.constants import (
    CELL_TIMEOUT_SEC,
    DEFAULT_TOOL_LIST,
    DEFAULT_SYSTEM_PROMPT,
    DEFINE_TOOL,
    HELPERS,
    MAX_CREATED_TOOLS,
    MAX_OUTPUT_CHARS,
    MAX_TOOL_CODE_CHARS,
    ONE_TOOL,
    OUTPUT_DIR,
    PRIVATE_PYTHON,
    RESERVED_TOOL_NAMES,
    TASK_PYTHON,
    TOOL_CREATION_NOTE,
    TOOLS_SECTION,
)
from ipyh.normalizers import python_default_normalizer, truncate

_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")  # colour in IPython tracebacks
_TIMEOUT_LINE = re.compile(
    r"^\s*#\s*timeout:\s*(\d+)"
)  # first line `# timeout: N` sets the cell timeout
_PYTHON_LINE = re.compile(
    r"^\s*(import |from |def |class |print\(|for |if |while |with |try:|[A-Za-z_][\w.]*\s*=[^=])"
)


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
        tool_creation: bool = False,
        **kwargs,
    ):
        self.system_prompt = system_prompt
        self.tool_processor = tool_processor
        # When set, the model gets `define_tool` and adds kernel tools of its own while it runs
        self.tool_creation = tool_creation

        # Kernel tools: each is a Python function, defined in the kernel by `code`, that the model
        # calls as its own tool, e.g. {"name": "f", "description": ..., "parameters": {...}, "code": "def f(x): ..."}
        self.kernel_tools = {tool["name"]: tool for tool in tools or []}
        self.configured_tools = set(self.kernel_tools)
        # Tools the model defined with `define_tool`, in order (a redefinition is recorded again)
        self.created_tools: List[dict] = []
        self._update_tool_list()
        self._saved_outputs = itertools.count(1)

    @classmethod
    def from_config(cls, path: str, **kwargs) -> "Harness":
        """A harness from a JSON file with an optional `system_prompt`, `tools` (kernel tools) and
        `tool_creation`."""
        with open(path) as f:
            config = json.load(f)
        return cls(
            system_prompt=config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            tools=config.get("tools"),
            tool_creation=config.get("tool_creation", False),
            **kwargs,
        )

    def _update_tool_list(self) -> None:
        # The agent sends `tool_list` with every model call, so a tool defined now is offered next turn
        self.tool_list = (
            DEFAULT_TOOL_LIST
            + ([DEFINE_TOOL] if self.tool_creation else [])
            + [
                {"type": "function", "function": {key: tool[key] for key in ("name", "description", "parameters")}}
                for tool in self.kernel_tools.values()
            ]
        )

    def setup_code(self) -> str:
        """Run once in each new kernel: the file helpers, then the kernel tools' definitions."""
        return "\n".join([HELPERS, *(tool["code"] for tool in self.kernel_tools.values())])

    def render_system_prompt(self, python: Optional[str]) -> str:
        """The system prompt for a kernel on `python` (None: a private interpreter).

        `{python}` is substituted by replacement, not `str.format`, so prompts may hold braces. With
        kernel tools (or tool creation) the prompt lists them; without, it is sent unchanged.
        """
        prompt = self.system_prompt.replace(
            "{python}", TASK_PYTHON.format(path=python) if python else PRIVATE_PYTHON
        )
        configured = [tool for name, tool in self.kernel_tools.items() if name in self.configured_tools]
        if configured or self.tool_creation:
            prompt = prompt.replace(*ONE_TOOL)
        if configured:
            listing = "\n".join(f"- `{tool['name']}`: {tool['description']}" for tool in configured)
            prompt += TOOLS_SECTION.format(tools=listing)
        if self.tool_creation:
            prompt += TOOL_CREATION_NOTE.format(limit=MAX_CREATED_TOOLS)
        return prompt

    def define_tool(self, arguments: dict, km, kc) -> str:
        """Run a `define_tool` call: validate it, define the function in the kernel, offer it as a tool."""
        try:
            tool = self._validate_tool(arguments)
        except (ValueError, SyntaxError) as error:
            return f"[define_tool rejected: {error}]"
        # A sentinel printed after the definition tells that it ran without error
        sentinel = f"[defined {tool['name']}]"
        output = self._run_cell(f"{tool['code']}\nprint({sentinel!r})", CELL_TIMEOUT_SEC, km, kc)
        if sentinel not in output:
            return f"[define_tool failed: the code raised while being defined]\n{output}"
        self.kernel_tools[tool["name"]] = tool
        self.created_tools.append(tool)
        self._update_tool_list()
        return f"Tool `{tool['name']}` is defined; call it as a tool from your next turn."

    def _validate_tool(self, data: dict) -> dict:
        name, code = (data.get("name") or "").strip(), data.get("code") or ""
        reserved = RESERVED_TOOL_NAMES | self.configured_tools
        if not name.isidentifier() or name in reserved:
            raise ValueError(f"the name must be a Python identifier other than {sorted(reserved)}")
        created = {tool["name"] for tool in self.created_tools}
        if name not in created and len(created) >= MAX_CREATED_TOOLS:
            raise ValueError(f"at most {MAX_CREATED_TOOLS} tools; redefine one of {sorted(created)} instead")
        parameters = data.get("parameters") or {"type": "object", "properties": {}}
        if isinstance(parameters, str):  # some models send the schema as a JSON string
            parameters = json.loads(parameters)
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError('parameters must be a JSON schema object: {"type": "object", "properties": {...}}')
        if len(code) > MAX_TOOL_CODE_CHARS:
            raise ValueError(f"keep the code under {MAX_TOOL_CODE_CHARS} characters")
        if not re.search(rf"^def {re.escape(name)}\(", code, re.M):
            raise ValueError(f"the code must define `def {name}(...)` at the top level")
        compile(code, f"<tool {name}>", "exec")
        description = (data.get("description") or "").strip()
        if not description:
            raise ValueError("describe what the tool does; you will see this description")
        return {"name": name, "description": description[:1000], "parameters": parameters, "code": code}

    def __add__(self, model) -> "Agent":
        if not isinstance(model, (OpenAI, AsyncOpenAI)):
            raise ValueError(
                "The model must be an `openai.OpenAI` or `openai.AsyncOpenAI` client"
            )

        # To avoid circular import
        from ipyh import Agent

        return Agent(model=model, harness=self)

    def execute_tool(
        self,
        tool_call,
        post_process_output: bool = False,
        timeout: int = CELL_TIMEOUT_SEC,
        **tool_kwargs,
    ):
        km, kc = tool_kwargs["km"], tool_kwargs["kc"]

        name = tool_call.get("function", {}).get("name")
        try:
            arguments = json.loads(tool_call["function"]["arguments"] or "{}")
            if name == "define_tool" and self.tool_creation:
                return self.define_tool(arguments, km, kc)
            # A kernel tool call becomes a call of its function; the ipython tool runs the code itself
            code = f"{name}(**{arguments!r})" if name in self.kernel_tools else arguments["code"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            return f"[invalid tool call arguments: {error}]"
        override = _TIMEOUT_LINE.match(code)
        if override:
            timeout = int(override.group(1))

        started = time.monotonic()
        output = self._run_cell(code, timeout, km, kc)
        # Shell lines in a Python cell (`!cmd` then bare shell) fail to compile, so nothing ran: run them as bash
        lines = code.strip().splitlines()
        if (
            "SyntaxError" in output
            and lines[0].lstrip().startswith("!")
            and len(lines) > 1
            and not any(_PYTHON_LINE.match(line) for line in lines[1:])
        ):
            shell = "\n".join(
                line.lstrip()[1:] if line.lstrip().startswith("!") else line
                for line in lines
            )
            output = "[ran as %%bash: the cell held shell lines]\n" + self._run_cell(
                "%%bash\n" + shell, timeout, km, kc
            )
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
                pass
        return _ANSI_SGR.sub("", "".join(out)) or "[no output]"
