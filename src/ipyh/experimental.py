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
"""
import asyncio
import json
from typing import List, Optional

from ipyh.agent import Agent

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

    def __init__(self, model, harness, generators: List[str], decision_model: str, **kwargs):
        super().__init__(model, harness, **kwargs)
        self.generators = generators
        self.decision_model = decision_model
        self._last_choice: Optional[dict] = None

    async def _complete(self, conversation):
        results = await asyncio.gather(
            *(super(BestOfN, self)._complete(conversation, model=g) for g in self.generators), return_exceptions=True
        )
        proposals = [(g, r) for g, r in zip(self.generators, results) if not isinstance(r, BaseException)]
        if not proposals:
            raise results[0]

        criteria = {f"candidate_{i + 1}": _describe(r.choices[0].message) for i, (_, r) in enumerate(proposals)}
        question = {"pick": {"type": "choice", "instructions": self.QUESTION, "criteria": criteria}}
        # /v1/systemone on the same address as the generators, through the same client (auth, timeouts)
        body = {"model": self.decision_model, "state": _state(conversation), "questions": question}
        try:
            reply = await self._call(self.model.post, "/systemone", body=body, cast_to=object)
            answer = reply["answers"]["pick"]
        except Exception as error:  # a rejected decision (e.g. state too long) falls back to the first candidate
            answer = {"choice": "candidate_1", "probabilities": {}, "error": str(error)[:200]}
        index = int(answer["choice"].rsplit("_", 1)[1]) - 1
        self._last_choice = {
            "chosen": proposals[index][0],
            "error": answer.get("error"),
            "probabilities": {proposals[i][0]: answer["probabilities"].get(f"candidate_{i + 1}") for i in range(len(proposals))},
            "candidates": [
                {"model": g, "prompt_tokens": r.usage.prompt_tokens if r.usage else None,
                 "completion_tokens": r.usage.completion_tokens if r.usage else None,
                 "tool_call": bool(r.choices[0].message.tool_calls)}
                for g, r in proposals
            ],
        }
        return proposals[index][1]

    async def run_step(self, optional_prompt: Optional[str] = None):
        message = await super().run_step(optional_prompt)
        if self._last_choice is not None:
            self.trajectories[self.current_task]["turns"][-1]["decision"] = self._last_choice
            self._last_choice = None
        return message
