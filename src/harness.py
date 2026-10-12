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
from typing import Optional, Callable, List

import json

from openai import OpenAI

from ipyh.constants import DEFAULT_TOOL_LIST, DEFAULT_SYSTEM_PROMPT
from ipyh.normalizers import python_default_normalizer

class Harness:
    """
    A simple harness object which consists of a ipython shell access. The harness
    has also configurable components such as system prompt, how to process tools, etc.
    """
    def __init__(
        self, 
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        tool_processor: Optional[Callable] = python_default_normalizer,
        **kwargs
    ):
        self.system_prompt = system_prompt
        self.tool_processor = tool_processor
        
        # TODO: make it configurable later
        self.tool_list = DEFAULT_TOOL_LIST
    
    def get_tools(self) -> List[dict]:
        return self.tool_list

    def __add__(self, model: OpenAI) -> "Agent":
        if not isinstance(model, OpenAI):
            raise ValueError("The model must be an instance of OpenAI model class")
        
        # To avoid circular import
        from ipyh import Agent

        return Agent(model=model, harness=self)

    def execute_tool(self, tool_call, post_process_output: bool = False, timeout: int = 600, **tool_kwargs):
        km, kc = tool_kwargs["km"], tool_kwargs["kc"]
        
        code = json.loads(tool_call["function"]["arguments"])["code"]
        out = []

        def hook(msg):
            kind, content = msg["msg_type"], msg["content"]
            if kind == "stream":
                out.append(content["text"])
            elif kind in ("execute_result", "display_data"):
                out.append(content["data"].get("text/plain", "") + "\n")
            elif kind == "error":
                out.append("\n".join(content["traceback"]) + "\n")

        try:
            kc.execute_interactive(code, output_hook=hook, timeout=timeout, allow_stdin=False)
        except TimeoutError:
            km.interrupt_kernel()
            out.append(f"\n[cell timed out after {timeout:.0f}s]\n")
        output = "".join(out) or "[no output]"

        if post_process_output:
            output = self.tool_processor(output)

        return output
