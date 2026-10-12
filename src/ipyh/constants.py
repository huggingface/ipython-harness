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
import re

# OpenAI compatible tool list: a single `ipython` tool.
DEFAULT_TOOL_LIST = [
    {
        "type": "function",
        "function": {
            "name": "ipython",
            "description": "Execute code in a persistent IPython kernel and return its output.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "IPython cell to run."}},
                "required": ["code"],
            },
        },
    }
]

# `{python}` is replaced by TASK_PYTHON or PRIVATE_PYTHON. The hard line breaks are part of the
# prompt: changing them changes the tokens the model sees.
DEFAULT_SYSTEM_PROMPT = """You are an autonomous agent solving a task inside a Linux container.
You have exactly one tool, `ipython`, which runs code in a persistent IPython kernel; variables,
imports and the working directory persist between calls. Run shell commands with `!cmd` or a
`%%bash` cell. {python} The kernel's own files live in /opt/ipy and /tmp/ipy: they are not part
of the task, so never inspect, modify or point anything at them.

How to work:
- Write files with a `%%writefile <path>` cell, never by embedding their contents in Python
  string literals.
- Start long-running services in the background (`nohup … &`) and confirm they respond.
- Never modify files the task tells you to keep unchanged; add new files instead.

Before you finish:
1. Re-read the task and list every requirement: each output file and its path, each name,
   column, format and value, and each service or side effect it asks for.
2. Check each requirement with code against what exists now: read the files back and query
   services and APIs directly rather than trusting your own variables. When the task states a
   tolerance or threshold, compare your result against it.
3. Fix anything that does not match, then check again.
Only then reply with a short summary and no tool call."""

# Fills `{python}` when the kernel runs on the task's own interpreter (`{path}`).
TASK_PYTHON = (
    "The kernel uses the task's Python ({path}): its installed packages import directly, and "
    "`!pip install …` installs into it."
)
# Fills `{python}` when the environment has no usable `python3` and the kernel runs a private one.
PRIVATE_PYTHON = (
    "The container has no usable `python3`, so the kernel runs a private Python; `%pip install …` "
    "installs packages for the kernel."
)

# Sent once when the model replies with neither a tool call nor a final answer.
NUDGE = (
    "Your last reply had no tool call and no final answer. If the task is complete, reply with a "
    "short summary and no tool call; otherwise continue by calling the `ipython` tool."
)
# A tool call the server failed to parse, left as text in the reply.
TOOL_MARKUP = re.compile(r"<tool_call>|</tool_call>|<function=|<parameter=")

MAX_OUTPUT_CHARS = 12_000  # longer cell output keeps its head and tail
CELL_TIMEOUT_SEC = 600  # a cell running longer is interrupted; the kernel keeps its state
RETRY_DELAYS_SEC = (2, 5, 15, 30, 60, 120)  # model endpoint: connection errors and 5xx
