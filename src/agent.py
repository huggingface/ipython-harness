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

import secrets
import sys
import subprocess

from openai import OpenAI
from jupyter_client.manager import start_new_kernel

from ipyh.harness import Harness

class Agent:
    """
    """
    def __init__(
        self,
        model: OpenAI,
        harness: Harness,
        **kwargs,
    ):
        self.model = model
        self.harness = harness

        self.trajectories = None
        self.current_task = None
        self.max_iteration = kwargs.get("max_iteration", 100)
        # A server can serve several models, `model_name` picks one of them. Otherwise
        # we use the first model listed by the server's `/v1/models` endpoint
        self.model_name = kwargs.get("model_name")

    def run_task(self, query: str, debug: bool = False):
        if self.trajectories is None:
            self.trajectories = {}

        if self.model_name is None:
            self.model_name = self.model.models.list().data[0].id

        random_hash = secrets.token_hex(8)
        self.trajectories[random_hash] = {}
        self.current_task = random_hash

        km, kc = start_new_kernel(stderr=subprocess.DEVNULL)

        # TODO: make the system prompt
        conversation = [
            {"role": "system", "content": self.harness.system_prompt.format(python=sys.executable)},
            {"role": "user", "content": query}
        ]
        self.trajectories[random_hash]['kernels'] = [km, kc]
        self.trajectories[random_hash]['trajectory'] = conversation

        # First iteration, in debug mode the caller drives the next ones with `run_step`
        message = self.run_step()

        if debug:
            return message

        for _ in range(self.max_iteration - 1):
            # No tool call means the model considers the task done
            if not message.tool_calls:
                break
            message = self.run_step()

        return message

    def run_step(self, optional_prompt: Optional[str] = None):
        """
        Runs a single turn on the current task: queries the model, then executes
        every tool call it made. Returns the assistant message.
        """
        if self.current_task is None:
            raise ValueError("No task is running, call `run_task` first")

        trajectory = self.trajectories[self.current_task]
        km, kc = trajectory['kernels']
        conversation = trajectory['trajectory']

        if optional_prompt is not None:
            conversation.append({"role": "user", "content": optional_prompt})

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=conversation,
            tools=self.harness.tool_list
        )
        message = response.choices[0].message

        assistant_message = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_message["tool_calls"] = [call.model_dump() for call in message.tool_calls]
        conversation.append(assistant_message)

        for call in message.tool_calls or []:
            tool_call = call.model_dump()
            result = self.harness.execute_tool(
                tool_call,
                post_process_output=True,
                km=km,
                kc=kc
            )
            conversation.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})

        return message

