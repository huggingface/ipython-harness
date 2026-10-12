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
        self.max_iteration = kwargs.get("max_iteration", 100)

    def run_task(self, query: str, debug: bool = False):
        if self.trajectories is None:
            self.trajectories = {}
        
        random_hash = secrets.token_hex(8)
        self.trajectories[random_hash] = {}       
        
        km, kc = start_new_kernel(stderr=subprocess.DEVNULL)
        
        conversation = []
        self.trajectories[random_hash]['kernels'] = [km, kc]
        self.trajectories[random_hash]['trajectory'] = conversation

        import pdb; pdb.set_trace()

        # TODO: make the system prompt 
        conversation = [
            {"role": "system", "content": self.harness.system_prompt.format(python=sys.executable)},
            {"role": "user", "content": query}
        ]
        first_response = self.model.chat.completions.create(
            model='unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL',
            messages=conversation,
            tools=self.harness.tool_list
        )
        first_message = first_response.choices[0].message
        
        # First iteration, we do it by hand
        for call in first_message.tool_calls or []:
            tool_call = call.model_dump()
            result = self.harness.execute_tool(
                tool_call, 
                post_process_output=True,
                km=km,
                kc=kc
            )
            # TODO: here post-process (normalize)
            conversation.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})
        
        if debug:
            return
        


        # for i in range(self.max_iteration):
            

        

    # def run_step(self, optional_prompt: str = None):

