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
"""Experimental policies (opt-in, not imported by the core library).

BestOfN: every turn, each generator model proposes its next reply in parallel and a decision model
(`/v1/systemone`, e.g. Clef-Flash in llama.cpp's router) picks one. All models are served from the
same address; requests select them by name. Each turn records every candidate's token usage and the
decision, so compute can be compared with a single larger model.

ThinkingLadder: one model; each turn starts with little or no thinking, and a step that a gate rejects
(cheap signals, or a decision model) is regenerated with more thinking before it runs.
"""
import asyncio
import json
import os
import time
from typing import List, Optional

import openai

from ipyh.agent import Agent
from ipyh.harness import _PYTHON_LINE

STATE_CHARS = 12_000  # with three candidates (≤3.5k chars each) the request stays under the 8k-token batch
PART_CHARS = 1_500


def _trim(text: str, n: int = PART_CHARS) -> str:
    text = text or ""
    return text if len(text) <= n else f"{text[: n // 2]}\n[…]\n{text[-n // 2:]}"


def _describe(message) -> str:
    """One candidate as the decision model sees it: its text and the cells it would run."""
    parts = [_trim(message.content or "", 1_000)] if message.content else []
    for call in message.tool_calls or []:
        try:
            code = json.loads(call.function.arguments or "{}").get("code", "")
        except json.JSONDecodeError:
            code = call.function.arguments
        parts.append(f"runs cell:\n{_trim(code, 2_500)}")
    return "\n".join(parts) or "(empty reply)"


def _state(conversation: list) -> str:
    """The task and the most recent steps, oldest first, within STATE_CHARS."""
    head = f"TASK:\n{_trim(conversation[1]['content'], 8_000)}\n\nRECENT STEPS (oldest first):\n"
    budget, parts = STATE_CHARS - len(head), []
    for message in reversed(conversation[2:]):
        if message["role"] == "tool":
            part = f"--- output\n{_trim(message['content'])}\n"
        elif message["role"] == "assistant":
            cells = [json.loads(c["function"]["arguments"] or "{}").get("code", "") for c in message.get("tool_calls") or []]
            part = "".join(f"--- cell\n{_trim(code)}\n" for code in cells) or f"--- reply\n{_trim(message['content'])}\n"
        else:
            part = f"--- note\n{_trim(message['content'])}\n"
        if len(part) > budget:
            break
        parts.insert(0, part)
        budget -= len(part)
    return head + "".join(parts)


class BestOfN(Agent):
    """Each turn: all `generators` propose, `decision_model` picks the reply most likely to progress."""

    QUESTION = (
        "Which candidate next step is most likely to lead to completing the task exactly as specified? "
        "Prefer steps that check facts, read the right files and test the result over guesses."
    )

    def __init__(self, model, harness, generators: List[str], decision_model: str, decision_url: Optional[str] = None, **kwargs):
        super().__init__(model, harness, **kwargs)
        # Each generator is a model served at the main address, or `name=url` for one on its own server
        # (e.g. one vLLM per model on an HF Job), reached with the main client's key
        self.generators, self._clients = [], {}
        for entry in generators:
            name, _, url = entry.partition("=")
            self.generators.append(name)
            if url:
                self._clients[name] = (openai.OpenAI if self.sync_client else openai.AsyncOpenAI)(
                    base_url=url, api_key=model.api_key, max_retries=0, timeout=openai.Timeout(None, connect=30.0))
        self.decision_model = decision_model
        # Decisions go to the generators' address, or to `decision_url` (e.g. d3 on its own server), with the
        # key from DECISION_API_KEY (default: the generators' key)
        self.decision = self.model
        if decision_url:
            client_class = openai.OpenAI if self.sync_client else openai.AsyncOpenAI
            key = os.environ.get("DECISION_API_KEY") or model.api_key
            self.decision = client_class(base_url=decision_url, api_key=key, max_retries=2, timeout=openai.Timeout(600, connect=30.0))
        self._last_choice: Optional[dict] = None

    async def _propose(self, conversation, generator: str):
        """One generator's reply and how long it took."""
        started = time.monotonic()
        response = await super()._complete(conversation, model=generator, client=self._clients.get(generator))
        return response, time.monotonic() - started

    async def _complete(self, conversation):
        results = await asyncio.gather(*(self._propose(conversation, g) for g in self.generators), return_exceptions=True)
        proposals = [(g, r[0]) for g, r in zip(self.generators, results) if not isinstance(r, BaseException)]
        seconds = {g: round(r[1], 2) for g, r in zip(self.generators, results) if not isinstance(r, BaseException)}
        if not proposals:
            raise results[0]

        criteria = {f"candidate_{i + 1}": _describe(r.choices[0].message) for i, (_, r) in enumerate(proposals)}
        question = {"pick": {"type": "choice", "instructions": self.QUESTION, "criteria": criteria}}
        body = {"model": self.decision_model, "state": _state(conversation), "questions": question}
        started = time.monotonic()
        try:
            reply = await self._call(self.decision.post, "/systemone", body=body, cast_to=object)
            answer = reply["answers"]["pick"]
        except Exception as error:  # a rejected decision (e.g. state too long) falls back to the first candidate
            answer = {"choice": "candidate_1", "probabilities": {}, "error": str(error)[:200]}
        index = int(answer["choice"].rsplit("_", 1)[1]) - 1
        self._last_choice = {
            "chosen": proposals[index][0],
            "error": answer.get("error"),
            # Wall time of the decision request (queueing included) and of each generator's reply
            "decision_sec": round(time.monotonic() - started, 2),
            # Generators whose request failed this turn (the others still competed)
            "failed": {g: f"{type(r).__name__}: {str(r)[:200]}" for g, r in zip(self.generators, results)
                       if isinstance(r, BaseException)},
            "probabilities": {proposals[i][0]: answer["probabilities"].get(f"candidate_{i + 1}") for i in range(len(proposals))},
            # What the decision model read, so each choice can be inspected (view_best_of_n.py)
            "state": body["state"],
            "candidates": [
                {"model": g, "seconds": seconds[g], "prompt_tokens": r.usage.prompt_tokens if r.usage else None,
                 "completion_tokens": r.usage.completion_tokens if r.usage else None,
                 "tool_call": bool(r.choices[0].message.tool_calls),
                 "shown": criteria[f"candidate_{i + 1}"],
                 "reasoning": _trim((r.choices[0].message.model_extra or {}).get("reasoning_content")
                                    or (r.choices[0].message.model_extra or {}).get("reasoning") or "", 4_000)}
                for i, (g, r) in enumerate(proposals)
            ],
        }
        picked = await self.review(self._last_choice, index)
        if picked != index:  # someone else picked: keep the decision model's own pick in the record
            self._last_choice["overridden"] = proposals[index][0]
            self._last_choice["chosen"] = proposals[picked][0]
        return proposals[picked][1]

    async def review(self, choice: dict, index: int) -> int:
        """Called with each decision before the picked step runs; returns the candidate index to run.

        The default keeps the decision model's pick; play_best_of_n.py overrides it to wait for a person.
        """
        return index

    async def run_step(self, optional_prompt: Optional[str] = None):
        message = await super().run_step(optional_prompt)
        if self._last_choice is not None:
            turn = self.trajectories[self.current_task]["turns"][-1]
            turn["decision"] = self._last_choice
            # The turn's cost is every candidate's, not only the picked one's
            for key in ("prompt_tokens", "completion_tokens"):
                turn[f"{key}_all"] = sum(c[key] or 0 for c in self._last_choice["candidates"])
            self._last_choice = None
        return message


# Qwen3.8's chat template: thinking off, or on at one of three efforts (request `chat_template_kwargs`)
THINKING = {
    "off": {"enable_thinking": False},
    "low": {"enable_thinking": True, "reasoning_effort": "low"},
    "medium": {"enable_thinking": True, "reasoning_effort": "medium"},
    "xhigh": {"enable_thinking": True, "reasoning_effort": "xhigh"},
}
LEVELS = list(THINKING)
RETRY_NOTE = (
    "Your draft for this step was:\n{draft}\n\nIt was not run: {reason}. Think it through again, check the task's "
    "requirements against what you know so far, and give a better next step (or a corrected final answer)."
)
STEP_QUESTION = (
    "Proposed next step:\n{step}\n\nWill running this step make real progress toward completing the task exactly "
    "as specified?"
)
FINAL_QUESTION = "Has the task been completed exactly as specified, with every requirement met?"


def _cell_parses(code: str) -> Optional[str]:
    """None when an IPython cell compiles as Python (shell and magic lines aside), else the SyntaxError."""
    if code.lstrip().startswith("%%"):  # a cell magic (bash, writefile, …) is not Python
        return None
    rows = code.strip().splitlines()
    if rows and rows[0].lstrip().startswith("!") and not any(_PYTHON_LINE.match(r) for r in rows[1:]):
        return None  # shell lines after `!cmd`: the harness runs the cell as bash
    lines = []
    for line in code.splitlines():
        stripped = line.lstrip()
        shell = stripped.startswith(("!", "%")) or (" = !" in line or "=!" in line.replace(" ", "")[:40])
        lines.append(line[: len(line) - len(stripped)] + "pass" if shell else line)
    try:
        compile("\n".join(lines), "<cell>", "exec")
    except SyntaxError as error:
        return f"SyntaxError: {error.msg} (line {error.lineno})"
    return None


def _errored(conversation: list) -> bool:
    """Whether the tool outputs after the latest assistant message show an error."""
    outputs = []
    for message in reversed(conversation):
        if message["role"] != "tool":
            break
        outputs.append(str(message.get("content")))
    text = "\n".join(outputs)
    return bool(outputs) and any(m in text for m in (
        "Traceback (most recent call last)", "Error:", "command not found", "No such file or directory", "[cell timed out"))


class ThinkingLadder(Agent):
    """Each turn starts with little or no thinking; a rejected step is regenerated with more thinking.

    `gate` decides whether a step is accepted before it runs:
    - "none": every step is accepted; the turn uses `start` (a fixed thinking level).
    - "signals": a cell that does not parse, or a final answer given below "medium", is rejected; a turn that
      follows an error starts one level higher per consecutive error.
    - "decision": a decision model (`/v1/systemone`, e.g. d3) is asked whether the step will make progress (or,
      for a final answer, whether the task is complete); below `threshold` the step is rejected.
    A rejected step is regenerated at the next level with the draft and the reason added for that call only;
    the history keeps the accepted step. `top` is the highest level; its step is always accepted. Each turn
    records every attempt (level, verdict, tokens, time).
    """

    def __init__(self, model, harness, gate: str = "none", start: str = "off", top: str = "xhigh",
                 decision_model: Optional[str] = None, decision_url: Optional[str] = None, threshold: float = 0.9,
                 **kwargs):
        super().__init__(model, harness, **kwargs)
        if gate not in ("none", "signals", "decision") or start not in THINKING or top not in THINKING:
            raise ValueError(f"unknown gate or thinking level: {gate}, {start}, {top}")
        self.gate, self.start, self.top, self.threshold = gate, LEVELS.index(start), LEVELS.index(top), threshold
        self.decision_model = decision_model
        self.decision = None
        if gate == "decision":
            if not decision_url:
                raise ValueError("gate 'decision' needs decision_url")
            client_class = openai.OpenAI if self.sync_client else openai.AsyncOpenAI
            key = os.environ.get("DECISION_API_KEY") or model.api_key
            self.decision = client_class(base_url=decision_url, api_key=key, max_retries=2, timeout=openai.Timeout(600, connect=30.0))
        self._ladder: Optional[dict] = None

    def _first_level(self, conversation: list) -> int:
        if self.gate != "signals":
            return self.start
        # One level up per consecutive latest step whose outputs showed an error
        steps = [i for i, m in enumerate(conversation) if m["role"] == "assistant" and m.get("tool_calls")]
        ends = [next((j for j in range(i + 1, len(conversation)) if conversation[j]["role"] != "tool"), len(conversation))
                for i in steps]
        streak = 0
        for end in reversed(ends):
            if not _errored(conversation[:end]):
                break
            streak += 1
        return min(self.top, self.start + streak)

    async def _judge(self, conversation: list, message, level: int):
        """(accepted, reason, decision score or None) for one generated step."""
        calls = message.tool_calls or []
        if not calls and not (message.content or "").strip():
            return True, "", None  # an empty reply: the agent's own nudge handles it
        if self.gate == "signals":
            for call in calls:
                try:
                    code = json.loads(call.function.arguments or "{}").get("code", "")
                except json.JSONDecodeError as error:
                    return False, f"the tool call's arguments are not valid JSON ({error})", None
                problem = _cell_parses(code)
                if problem:
                    return False, f"the cell does not parse ({problem})", None
            if not calls and (message.content or "").strip() and level < LEVELS.index("medium"):
                return False, "a final answer needs a careful check of every requirement first", None
            return True, "", None
        if self.gate == "decision":
            if calls:
                state, question = _state(conversation), {"type": "noul", "instructions": STEP_QUESTION.format(step=_describe(message))}
            else:
                final = conversation + [{"role": "assistant", "content": message.content or ""}]
                state, question = _state(final), {"type": "noul", "instructions": FINAL_QUESTION}
            try:
                reply = await self._call(self.decision.post, "/systemone",
                                         body={"model": self.decision_model, "state": state, "questions": {"q": question}},
                                         cast_to=object)
                score = reply["answers"]["q"]["noul"]
            except Exception as error:  # the decision model failed: accept rather than stall the run
                return True, f"decision failed: {str(error)[:150]}", None
            if score >= self.threshold:
                return True, "", score
            return False, f"a reviewer judged it unlikely to help (confidence {score:.2f})", score
        return True, "", None

    async def _complete(self, conversation):
        level = self._first_level(conversation)
        attempts, request = [], conversation
        while True:
            started = time.monotonic()
            response = await super()._complete(request, extra_body={"chat_template_kwargs": THINKING[LEVELS[level]]})
            seconds = time.monotonic() - started
            message = response.choices[0].message
            if response.choices[0].finish_reason == "length" and not message.tool_calls:
                accepted, reason, score = True, "cut at the token limit (the agent asks for smaller steps)", None
            else:
                accepted, reason, score = await self._judge(conversation, message, level)
            usage = response.usage
            attempts.append({"level": LEVELS[level], "accepted": accepted or level >= self.top, "reason": reason,
                             "score": score, "seconds": round(seconds, 2),
                             "prompt_tokens": usage.prompt_tokens if usage else None,
                             "completion_tokens": usage.completion_tokens if usage else None,
                             "shown": _describe(message)})
            if accepted or level >= self.top:
                break
            note = RETRY_NOTE.format(draft=_describe(message), reason=reason)
            request = conversation + [{"role": "user", "content": note}]
            level += 1
        self._ladder = {"gate": self.gate, "attempts": attempts}
        return response

    async def run_step(self, optional_prompt: Optional[str] = None):
        message = await super().run_step(optional_prompt)
        if self._ladder is not None:
            turn = self.trajectories[self.current_task]["turns"][-1]
            turn["ladder"] = self._ladder
            for key in ("prompt_tokens", "completion_tokens"):
                turn[f"{key}_all"] = sum(a[key] or 0 for a in self._ladder["attempts"])
            self._ladder = None
        return message
