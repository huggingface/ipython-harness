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

# OpenAI compatible default tool list which consists of simply using ipython
DEFAULT_TOOL_LIST = [
    {
      "type": "function",
          "function": {
            "name": "ipython",
            "description": "Execute code in a persistent IPython kernel and return its output.",
            "parameters": {
              "type": "object",
              "properties": {
                "code": {
                  "type": "string",
                  "description": "IPython cell to run."
                }
              },
              "required": [
                "code"
              ]
            }
          }
    }
]

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
Only then reply with a short summary and no tool call.
"""
