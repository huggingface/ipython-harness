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
from typing import Optional

import asyncio
import itertools
import json
import re
import secrets
import sys
import subprocess
import time

import openai
from jupyter_client.manager import start_new_kernel

from ipyh.constants import (
    LENGTH_NOTE,
    MAX_TOKENS_PER_TURN,
    NUDGE,
    RETRY_DELAYS_SEC,
    TOOL_MARKUP,
)
from ipyh.harness import Harness
from ipyh.normalizers import QUIET_ENV


_XML_CALL = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.S)
_XML_PARAMETER = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)
_JSON_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def recover_tool_calls(text: str, tool_list: list) -> list:
    """
    Complete tool calls left as text that the server did not parse (some providers leave them in
    the reasoning), in the XML `<function=...><parameter=...>` or the JSON `<tool_call>` format.
    """
    schemas = {
        tool["function"]["name"]: tool["function"].get("parameters", {}).get("properties", {})
        for tool in tool_list
    }

    def typed(name, key, value):
        # XML values are text: decode them only where the tool's schema expects something else
        if schemas.get(name, {}).get(key, {}).get("type", "string") == "string":
            return value
        try:
            return json.loads(value)
        except ValueError:
            return value

    calls = [
        (name, {key: typed(name, key, value) for key, value in _XML_PARAMETER.findall(body)})
        for name, body in _XML_CALL.findall(text)
    ]
    for raw in [] if calls else _JSON_CALL.findall(text):
        try:
            call = json.loads(raw)
            calls.append((call["name"], call.get("arguments") or {}))
        except (ValueError, KeyError, TypeError):
            pass
    return [
        {
            "id": f"call_{secrets.token_hex(6)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        for name, arguments in calls
        if name in schemas
    ]


class Agent:
    """
    Drives a model through a task with the harness's `ipython` tool, in a kernel on this machine
    (the model's code runs with your permissions: use a throwaway container).

    `model` is an `openai.OpenAI` or `openai.AsyncOpenAI` client. Use `run_task` / `run_step` from
    async code, or `run_task_sync` / `run_step_sync` from a plain script.
    """

    def __init__(
        self,
        model,
        harness: Harness,
        **kwargs,
    ):
        # Retries are done in `_complete`, and a slow generation is waited for.
        self.model = model.with_options(
            max_retries=0, timeout=openai.Timeout(None, connect=30.0)
        )
        self.sync_client = isinstance(model, openai.OpenAI)
        self.harness = harness

        self.trajectories = None
        self.current_task = None
        # None: no limit on model calls per task
        self.max_iteration = kwargs.get("max_iteration")
        # A server can serve several models, `model_name` picks one of them. Otherwise
        # we use the first model listed by the server's `/v1/models` endpoint
        self.model_name = kwargs.get("model_name")
        # Interpreter named in the system prompt (None: "a private Python"); defaults to the kernel's
        self.python = kwargs.get("python", sys.executable)
        # Reasoning + answer tokens per model call, so one turn cannot use the whole time budget
        self.max_tokens = kwargs.get("max_tokens", MAX_TOKENS_PER_TURN)
        # When set, tool results end with the time used so far, e.g. `[elapsed 12m of 60m]`
        self.time_limit_sec = kwargs.get("time_limit_sec")

    async def run_task(
        self, query: str, debug: bool = False, approve_commands: bool = False
    ):
        if self.trajectories is None:
            self.trajectories = {}

        if self.model_name is None:
            self.model_name = (await self._call(self.model.models.list)).data[0].id

        random_hash = secrets.token_hex(8)
        self.trajectories[random_hash] = {}
        self.current_task = random_hash

        km, kc = await asyncio.to_thread(
            start_new_kernel, kernel_name="python3", stderr=subprocess.DEVNULL
        )
        # Quiet progress bars and colour in the commands the model runs, and define the file helpers
        # and kernel tools
        setup = f"import os; os.environ.update({json.dumps(QUIET_ENV)})\n{self.harness.setup_code()}"
        await asyncio.to_thread(
            self.harness.execute_tool,
            {"function": {"arguments": json.dumps({"code": setup})}},
            km=km,
            kc=kc,
        )

        conversation = [
            {
                "role": "system",
                "content": self.harness.render_system_prompt(self.python),
            },
            {"role": "user", "content": query},
        ]
        self.trajectories[random_hash]["kernels"] = [km, kc]
        self.trajectories[random_hash]["trajectory"] = conversation
        # One entry per model call: token usage and reasoning (which is not sent back to the model)
        self.trajectories[random_hash]["turns"] = []
        self.trajectories[random_hash]["nudged"] = False
        self.trajectories[random_hash]["done"] = False
        self.trajectories[random_hash]["started"] = time.monotonic()

        # First iteration, in debug mode the caller drives the next ones with `run_step`
        message = await self.run_step(approve_commands=approve_commands)

        if debug:
            return message

        try:
            for _ in (
                range(self.max_iteration - 1)
                if self.max_iteration
                else itertools.count()
            ):
                if self.trajectories[random_hash]["done"]:
                    break
                message = await self.run_step(approve_commands=approve_commands)
        finally:
            self.shutdown_kernel(random_hash)

        return message

    async def run_step(
        self, optional_prompt: Optional[str] = None, approve_commands: bool = False
    ):
        """
        Runs a single turn on the current task: queries the model, then executes
        every tool call it made. Returns the assistant message.
        """
        if self.current_task is None:
            raise ValueError("No task is running, call `run_task` first")

        trajectory = self.trajectories[self.current_task]
        km, kc = trajectory["kernels"]
        conversation = trajectory["trajectory"]

        if optional_prompt is not None:
            conversation.append({"role": "user", "content": optional_prompt})

        response = await self._complete(conversation)
        message = response.choices[0].message

        content = message.content or ""
        reasoning = (
            (message.model_extra or {}).get("reasoning_content")
            or (message.model_extra or {}).get("reasoning")
            or ""
        )
        assistant_message = {"role": "assistant", "content": content}
        tool_calls = [call.model_dump() for call in message.tool_calls or []]
        if not tool_calls:
            # A call left as text (e.g. in the reasoning) is run as the model intended
            tool_calls = recover_tool_calls(content + "\n" + reasoning, self.harness.tool_list)

        # Keep the latest reasoning in the history, so work done while thinking is not re-derived; older
        # turns drop it to bound the context
        for previous in conversation:
            previous.pop("reasoning_content", None)
        if reasoning:
            assistant_message["reasoning_content"] = reasoning
        conversation.append(assistant_message)
        usage = response.usage
        trajectory["turns"].append(
            {
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "reasoning": reasoning,
            }
        )

        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

            for tool_call in tool_calls:
                if approve_commands:
                    approve = input(
                        f"The model is about to call the following tool: {tool_call}\napprove ? y/n\n"
                    )
                    if approve.lower() != "y":
                        print("Stopped due to non approved command. Exiting.")
                        exit(1)

                result = await asyncio.to_thread(
                    self.harness.execute_tool,
                    tool_call,
                    post_process_output=True,
                    km=km,
                    kc=kc,
                )
                conversation.append(
                    {"role": "tool", "tool_call_id": tool_call["id"], "content": result}
                )
            if self.time_limit_sec:
                minutes = (time.monotonic() - trajectory["started"]) / 60
                conversation[-1]["content"] += (
                    f"\n[elapsed {minutes:.0f}m of {self.time_limit_sec / 60:.0f}m]"
                )
            trajectory["nudged"] = False

        # A reply cut at the token limit is not an answer: ask for smaller steps
        elif response.choices[0].finish_reason == "length":
            conversation.append({"role": "user", "content": LENGTH_NOTE})

        # No tool call and a non-empty answer means the model considers the task done. An empty
        # reply, or a tool call the server failed to parse, gets one nudge first.
        else:
            if trajectory["nudged"] or (
                content.strip() and not TOOL_MARKUP.search(content + reasoning)
            ):
                trajectory["done"] = True
            else:
                trajectory["nudged"] = True
                conversation.append({"role": "user", "content": NUDGE})

        return message

    def run_task_sync(
        self, query: str, debug: bool = False, approve_commands: bool = False
    ):
        return asyncio.run(
            self.run_task(query, debug=debug, approve_commands=approve_commands)
        )

    def run_step_sync(
        self, optional_prompt: Optional[str] = None, approve_commands: bool = False
    ):
        return asyncio.run(
            self.run_step(optional_prompt, approve_commands=approve_commands)
        )

    def shutdown_kernel(self, task: Optional[str] = None):
        km, kc = self.trajectories[task or self.current_task]["kernels"]
        kc.stop_channels()
        km.shutdown_kernel(now=True)

    async def _call(self, fn, *args, **kwargs):
        # A sync client's calls run in a worker thread, so neither client blocks an event loop
        if self.sync_client:
            return await asyncio.to_thread(fn, *args, **kwargs)
        return await fn(*args, **kwargs)

    async def _complete(self, conversation):
        # Retries connection errors, rate limits (429) and 5xx; other errors raise with the server's reason
        for delay in (*RETRY_DELAYS_SEC, None):
            try:
                return await self._call(
                    self.model.chat.completions.create,
                    model=self.model_name,
                    messages=conversation,
                    tools=self.harness.tool_list,
                    max_tokens=self.max_tokens,
                )
            except openai.APIConnectionError:
                if delay is None:
                    raise
            except openai.APIStatusError as error:
                retryable = error.status_code >= 500 or error.status_code == 429
                if not retryable or delay is None:
                    raise
            await asyncio.sleep(delay)
