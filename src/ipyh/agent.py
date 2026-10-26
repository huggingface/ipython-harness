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
from typing import Iterable, Optional

import asyncio
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
        *,
        model_name: Optional[str] = None,
        python: Optional[str] = sys.executable,
        max_iteration: Optional[int] = None,
        max_tokens: int = MAX_TOKENS_PER_TURN,
        time_limit_sec: Optional[float] = None,
        context_window: Optional[int] = None,
        keep_tool_results: Optional[int] = None,
        stop_tools: Iterable[str] = (),
    ):
        """
        - `model_name`: the served model to use (None: the first one listed by `/v1/models`).
        - `python`: the interpreter named in the system prompt (None: "a private Python").
        - `max_iteration`: model calls per task (None: no limit).
        - `max_tokens`: reasoning + answer tokens per call, so one turn cannot use the whole budget.
        - `time_limit_sec`: when set, tool results end with the time used, e.g. `[elapsed 12m of 60m]`.
        - `context_window`: the model's context in tokens, to compact before it fills (None: compact
          only when the server rejects a request as too long).
        - `keep_tool_results`: tool results sent verbatim; older ones are sent as one-line stubs (the
          stored trajectory keeps them in full). None: send every result verbatim.
        - `stop_tools`: kernel tools whose successful call (no traceback) ends the task, e.g. one that
          submits the answer, so the model is not asked for a closing reply.
        """
        # Retries are done in `_complete`, and a slow generation is waited for
        self.model = model.with_options(max_retries=0, timeout=openai.Timeout(None, connect=30.0))
        self.sync_client = isinstance(model, openai.OpenAI)
        self.harness = harness
        self.model_name = model_name
        self.python = python
        self.max_iteration = max_iteration
        self.max_tokens = max_tokens
        self.time_limit_sec = time_limit_sec
        self.context_window = context_window
        self.keep_tool_results = keep_tool_results
        self.stop_tools = set(stop_tools)
        # The latest reasoning is sent back with the history, unless the server rejects it
        self.send_reasoning = True

        self.trajectories = {}
        self.current_task = None

    async def run_task(
        self, query: str, debug: bool = False, approve_commands: bool = False
    ):
        if self.model_name is None:
            self.model_name = (await self._call(self.model.models.list)).data[0].id

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

        self.current_task = secrets.token_hex(8)
        task = self.trajectories[self.current_task] = {
            "kernels": [km, kc],
            "query": query,
            "trajectory": [
                {"role": "system", "content": self.harness.render_system_prompt(self.python)},
                {"role": "user", "content": query},
            ],
            # One entry per model call: token usage and reasoning
            "turns": [],
            # The turns each compaction replaced, oldest first
            "compacted": [],
            "nudged": False,
            "done": False,
            "started": time.monotonic(),
        }

        # First turn; in debug mode the caller drives the next ones with `run_step`
        message = await self.run_step(approve_commands=approve_commands)
        if debug:
            return message
        try:
            while not task["done"] and len(task["turns"]) < (self.max_iteration or float("inf")):
                message = await self.run_step(approve_commands=approve_commands)
        finally:
            self.shutdown_kernel(self.current_task)
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
            if "reasoning_content" in str(error):
                # The server (e.g. Cerebras) rejects reasoning in the history: stop sending it
                self.send_reasoning = False
                for previous in conversation:
                    previous.pop("reasoning_content", None)
            elif CONTEXT_OVERFLOW.search(str(error)):
                await self.compact()
            else:
                raise
            response = await self._complete(self._request_view(conversation))
        message = response.choices[0].message

        content = message.content or ""
        extra = message.model_extra or {}
        reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""
        assistant_message = {"role": "assistant", "content": content}
        tool_calls = [call.model_dump() for call in message.tool_calls or []]
        if not tool_calls:
            # A call left as text (e.g. in the reasoning) is run as the model intended
            tool_calls = recover_tool_calls(content + "\n" + reasoning, self.harness.tool_list)

        # Keep the latest reasoning in the history, so work done while thinking is not re-derived; older
        # turns drop it to bound the context
        for previous in conversation:
            previous.pop("reasoning_content", None)
        if reasoning and self.send_reasoning:
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
                    trajectory["done"] = True
            if trajectory["done"]:
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
