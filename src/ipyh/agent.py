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
import os
import secrets
import sys
import subprocess
import time
from pathlib import Path

import openai
from jupyter_client.manager import start_new_kernel

from ipyh.constants import (
    COMPACT_CELL_CHARS,
    COMPACT_HEADROOM,
    COMPACT_KEEP_CHARS,
    COMPACT_PROMPT,
    CONTEXT_OVERFLOW,
    LENGTH_NOTE,
    MAX_TOKENS_PER_TURN,
    NUDGE,
    OUTPUT_DIR,
    RETRY_DELAYS_SEC,
    SUMMARY_PREFIX,
    TOOL_MARKUP,
    TOOL_STUB_CHARS,
)
from ipyh.harness import Harness
from ipyh.normalizers import truncate
from ipyh.utils import recover_tool_calls, render_transcript


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
        # The model's context window in tokens, to compact before it fills. None: compact only when
        # the server rejects a request as too long
        self.context_window = kwargs.get("context_window")
        # Tool results sent verbatim: older ones become a one-line stub in the request (the stored
        # trajectory keeps them in full). None: send every result verbatim
        self.keep_tool_results = kwargs.get("keep_tool_results")
        # Kernel tools that end the task when a call to them succeeds (its output has no traceback),
        # e.g. a tool that submits the answer: the model is not asked for a closing reply
        self.stop_tools = set(kwargs.get("stop_tools") or [])

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
        await asyncio.to_thread(
            self.harness.execute_tool,
            {"function": {"arguments": json.dumps({"code": self.harness.kernel_setup()})}},
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
        self.trajectories[random_hash]["query"] = query
        # The turns each compaction replaced, oldest first
        self.trajectories[random_hash]["compacted"] = []
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

        try:
            response = await self._complete(self._request_view(conversation))
        except openai.BadRequestError as error:
            if not CONTEXT_OVERFLOW.search(str(error)):
                raise
            await self.compact()
            response = await self._complete(self._request_view(conversation))
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

            stopped = False
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
                try:
                    json.loads(tool_call["function"]["arguments"] or "{}")
                except (KeyError, TypeError, json.JSONDecodeError):
                    # A malformed call (e.g. cut at the token limit) stays in the history as an empty
                    # one: servers reject a history whose arguments are not valid JSON
                    tool_call["function"]["arguments"] = "{}"
                conversation.append(
                    {"role": "tool", "tool_call_id": tool_call["id"], "content": result}
                )
                if tool_call.get("function", {}).get("name") in self.stop_tools and "Traceback" not in result:
                    stopped = True
            if stopped:
                trajectory["done"] = True
                return message
            if self.time_limit_sec:
                minutes = (time.monotonic() - trajectory["started"]) / 60
                conversation[-1]["content"] += (
                    f"\n[elapsed {minutes:.0f}m of {self.time_limit_sec / 60:.0f}m]"
                )
            trajectory["nudged"] = False
            if self.context_window and usage and usage.total_tokens > (
                self.context_window - self.max_tokens - COMPACT_HEADROOM
            ):
                await self.compact()

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

    async def compact(self):
        """
        Replaces the current task's older turns by a checkpoint the model writes, keeping the system
        prompt, the task and the most recent turns. The kernel keeps its state, so the checkpoint only
        has to say what was done and what is left; the full transcript is saved to a file.
        """
        trajectory = self.trajectories[self.current_task]
        conversation = trajectory["trajectory"]
        # The kept turns start at an assistant message, so every tool result keeps its call
        starts = [i for i, message in enumerate(conversation) if message["role"] == "assistant"]
        cut = next(
            (
                i
                for i in starts
                if len(json.dumps(conversation[i:])) <= COMPACT_KEEP_CHARS
            ),
            starts[-1] if starts else len(conversation),
        )
        older = conversation[1:cut]  # from the task (and an earlier checkpoint) onwards
        if len(older) < 2:
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = Path(OUTPUT_DIR, f"transcript-{self.current_task}-{len(trajectory['compacted']) + 1}.md")
        await asyncio.to_thread(path.write_text, render_transcript(older))

        transcript = render_transcript(older, limit=COMPACT_CELL_CHARS)
        if self.context_window:
            # ~3 characters per token, leaving room for the prompt and the reply
            transcript = truncate(transcript, 3 * (self.context_window - 2 * self.max_tokens))
        request = f"<conversation>\n{transcript}\n</conversation>\n\n{COMPACT_PROMPT}"
        response = await self._complete([{"role": "user", "content": request}], tools=False)
        summary = response.choices[0].message.content or "(the checkpoint could not be written)"

        trajectory["compacted"].append(conversation[1:cut])
        conversation[1:cut] = [
            {
                "role": "user",
                "content": f"{trajectory['query']}\n\n{SUMMARY_PREFIX.format(path=path)}\n{summary}",
            }
        ]

    def _request_view(self, conversation: list) -> list:
        """The conversation as sent: tool results older than the last `keep_tool_results` become stubs.

        The cut moves in steps of `keep_tool_results`, so the request's prefix stays the same for
        several turns (a server's prompt cache keeps working between steps).
        """
        keep = self.keep_tool_results
        results = [i for i, message in enumerate(conversation) if message["role"] == "tool"]
        if not keep or len(results) <= keep:
            return conversation
        stub_count = (len(results) - keep) // keep * keep
        view = list(conversation)
        for i in results[:stub_count]:
            content = str(view[i]["content"])
            if len(content) > TOOL_STUB_CHARS:
                first = content.strip().splitlines()[0][:TOOL_STUB_CHARS] if content.strip() else ""
                view[i] = {**view[i], "content": f"{first}\n[older output, {len(content)} chars, dropped to "
                                                 "save context; run the call again if you need it]"}
        return view

    def shutdown_kernel(self, task: Optional[str] = None):
        km, kc = self.trajectories[task or self.current_task]["kernels"]
        kc.stop_channels()
        km.shutdown_kernel(now=True)

    async def _call(self, fn, *args, **kwargs):
        # A sync client's calls run in a worker thread, so neither client blocks an event loop
        if self.sync_client:
            return await asyncio.to_thread(fn, *args, **kwargs)
        return await fn(*args, **kwargs)

    async def _complete(self, conversation, tools: bool = True):
        # Retries connection errors, rate limits (429) and 5xx; other errors raise with the server's reason
        for delay in (*RETRY_DELAYS_SEC, None):
            try:
                return await self._call(
                    self.model.chat.completions.create,
                    model=self.model_name,
                    messages=conversation,
                    tools=self.harness.tool_list if tools else openai.NOT_GIVEN,
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
