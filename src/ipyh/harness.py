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
    CELL_TIMEOUT_SEC, DEFAULT_TOOL_LIST, DEFAULT_SYSTEM_PROMPT, MAX_OUTPUT_CHARS, OUTPUT_DIR, PRIVATE_PYTHON, TASK_PYTHON,
)
from ipyh.normalizers import python_default_normalizer, truncate

_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")  # colour in IPython tracebacks
_TIMEOUT_LINE = re.compile(r"^\s*#\s*timeout:\s*(\d+)")  # first line `# timeout: N` sets the cell timeout
_PYTHON_LINE = re.compile(r"^\s*(import |from |def |class |print\(|for |if |while |with |try:|[A-Za-z_][\w.]*\s*=[^=])")


class Harness:
    """
    A simple harness object which consists of a ipython shell access. The harness
    has also configurable components such as system prompt, how to process tools, etc.
    """
    def __init__(
        self,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        tool_processor: Optional[Callable] = python_default_normalizer,
        **kwargs
    ):
        self.system_prompt = system_prompt
        self.tool_processor = tool_processor

        # TODO: make it configurable later
        self.tool_list = DEFAULT_TOOL_LIST
        self._saved_outputs = itertools.count(1)

    def render_system_prompt(self, python: Optional[str]) -> str:
        """The system prompt for a kernel on `python` (None: a private interpreter)."""
        return self.system_prompt.format(python=TASK_PYTHON.format(path=python) if python else PRIVATE_PYTHON)

    def __add__(self, model) -> "Agent":
        if not isinstance(model, (OpenAI, AsyncOpenAI)):
            raise ValueError("The model must be an `openai.OpenAI` or `openai.AsyncOpenAI` client")

        # To avoid circular import
        from ipyh import Agent

        return Agent(model=model, harness=self)

    def execute_tool(self, tool_call, post_process_output: bool = False, timeout: int = CELL_TIMEOUT_SEC, **tool_kwargs):
        km, kc = tool_kwargs["km"], tool_kwargs["kc"]

        try:
            code = json.loads(tool_call["function"]["arguments"] or "{}")["code"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            return f"[invalid tool call arguments: {error}]"
        override = _TIMEOUT_LINE.match(code)
        if override:
            timeout = int(override.group(1))

        started = time.monotonic()
        output = self._run_cell(code, timeout, km, kc)
        # Shell lines in a Python cell (`!cmd` then bare shell) fail to compile, so nothing ran: run them as bash
        lines = code.strip().splitlines()
        if "SyntaxError" in output and lines[0].lstrip().startswith("!") and len(lines) > 1 \
                and not any(_PYTHON_LINE.match(line) for line in lines[1:]):
            shell = "\n".join(line.lstrip()[1:] if line.lstrip().startswith("!") else line for line in lines)
            output = "[ran as %%bash: the cell held shell lines]\n" + self._run_cell("%%bash\n" + shell, timeout, km, kc)
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
            kc.execute_interactive(code, output_hook=hook, timeout=timeout, allow_stdin=False)
        except TimeoutError:
            km.interrupt_kernel()
            out.append(f"\n[cell timed out after {timeout:.0f}s; kernel interrupted, state kept]\n")
            # The interrupted cell's last messages would be taken for the next cell's: absorb them with an empty one
            try:
                kc.execute_interactive("", output_hook=lambda msg: None, timeout=30, allow_stdin=False)
            except TimeoutError:
                pass
        return _ANSI_SGR.sub("", "".join(out)) or "[no output]"
