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
                "properties": {
                    "code": {"type": "string", "description": "IPython cell to run."}
                },
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
- Change existing files with `edit(path, old, new)`, which replaces text that must occur exactly
  once, and read them with `view(path, start=1, n=200)`, instead of rewriting or printing whole files.
- A cell is interrupted after 300 s; start it with a `# timeout: N` line when it needs longer. Long
  outputs are cut, and the full output is saved to the file named in the output.
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

MAX_OUTPUT_CHARS = 12_000  # longer cell output keeps its head and tail; the full output goes to OUTPUT_DIR
OUTPUT_DIR = "/tmp/ipyh"
CELL_TIMEOUT_SEC = 300  # a cell running longer is interrupted (the kernel keeps its state); `# timeout: N` overrides
MAX_TOKENS_PER_TURN = 12_000  # reasoning + answer per model call, so one turn cannot use the whole time budget

# Sent when a reply hits MAX_TOKENS_PER_TURN without a tool call.
LENGTH_NOTE = (
    "Your last reply was cut at the per-turn token limit before any tool call. Think in smaller "
    "steps: run a cell now to check or build part of the solution."
)

# Run in the kernel at start: file helpers the model calls from its cells.
HELPERS = """
def edit(path, old, new, count=1):
    "Replace `old` with `new` in `path`; `old` must occur exactly `count` times."
    text = open(path).read()
    found = text.count(old)
    if found != count:
        raise ValueError(f"edit: the text occurs {found} times in {path}, expected {count}; view() the file and copy it exactly")
    with open(path, "w") as f:
        f.write(text.replace(old, new))
    print(f"edited {path}")

def view(path, start=1, n=200):
    "Print `n` lines of `path` from line `start`, numbered."
    lines = open(path, errors="replace").read().splitlines()
    end = min(len(lines), start - 1 + n)
    print("\\n".join(f"{i:>6}  {lines[i - 1]}" for i in range(max(start, 1), end + 1)))
    if end < len(lines):
        print(f"[{len(lines) - end} more lines: view({path!r}, start={end + 1})]")
"""
RETRY_DELAYS_SEC = (2, 5, 15, 30, 60, 120)  # model endpoint: connection errors and 5xx

# Compaction: when a turn ends within `max_tokens + COMPACT_HEADROOM` of `context_window`, or the server
# rejects a request as too long, the older turns are replaced by a model-written checkpoint
COMPACT_HEADROOM = 4_000  # tokens: room for the cell output appended after the measured turn
COMPACT_KEEP_CHARS = 40_000  # the most recent turns kept verbatim (~10k tokens)
COMPACT_CELL_CHARS = 2_000  # each cell and its output, as the summarizer reads them
# Matches a server's "prompt too long" rejection (OpenAI, vLLM, llama.cpp, SGLang)
CONTEXT_OVERFLOW = re.compile(
    r"context.?(length|size|window)|too many tokens|prompt is too long", re.I
)

# Sent after the transcript of the turns being compacted.
COMPACT_PROMPT = """The conversation above is being compacted. Write a checkpoint that another agent will use to
continue the task in the same IPython kernel, whose variables, files and running processes are
all still there.

Use exactly these sections, and keep exact paths, names, commands, values and error messages:
## Progress
What is done and checked, and what was tried and failed (and why).
## Kernel and filesystem state
Variables and functions defined, files written or edited, packages installed, services running
(with their ports).
## Requirements
Each requirement of the task, and whether it is met, unchecked or not started.
## Next steps
The actions that remain, in order.

If the conversation starts with an earlier checkpoint, merge it into this one. Reply with the
checkpoint only."""

# Follows the task in the compacted conversation, then the checkpoint.
SUMMARY_PREFIX = (
    "Your earlier turns on this task were compacted to save context; the kernel's state is "
    "unchanged. The full earlier conversation is in {path}: `view()` it for a detail the "
    "checkpoint lacks. Checkpoint:"
)
