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
"""Harbor agent plugin: `import_path: ipyh.harbor:IPyHarnessAgent` (requires the `harbor` package).

The model loop runs in the Harbor process against an OpenAI-compatible endpoint
(OPENAI_BASE_URL, OPENAI_API_KEY); cells run in a RemoteKernel inside the task's sandbox. Writes an
ATIF trajectory to agent/trajectory.json after every step.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent as AtifAgent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.utils.env import resolve_env_vars
from harbor.utils.trajectory_utils import format_trajectory_json
from openai import AsyncOpenAI

from ipyh import __version__
from ipyh.agent import Agent, Task
from ipyh.constants import CELL_TIMEOUT_SEC
from ipyh.harness import Harness
from ipyh.kernels import RemoteKernel


class IPyHarnessAgent(BaseInstalledAgent):
    # max_turns=None: no turn cap, so Harbor's agent timeout is the only budget.
    def __init__(
        self,
        *args: Any,
        max_turns: Optional[int] = None,
        cell_timeout_sec: int = CELL_TIMEOUT_SEC,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._max_turns = max_turns
        self._cell_timeout_sec = cell_timeout_sec
        self._kernel: Optional[RemoteKernel] = None

    @staticmethod
    def name() -> str:
        return "ipyh"

    def version(self) -> str:
        return __version__

    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("curl",))

        async def run(command: str, timeout_sec: Optional[int]):
            result = await environment.exec(command=command, timeout_sec=timeout_sec)
            return result.stdout, result.stderr, result.return_code

        async def as_root(command: str, timeout_sec: Optional[int]):
            result = await self.exec_as_root(
                environment, command=command, timeout_sec=timeout_sec
            )
            return result.stdout, result.stderr, result.return_code

        async def as_agent(command: str, timeout_sec: Optional[int]):
            result = await self.exec_as_agent(
                environment, command=command, timeout_sec=timeout_sec
            )
            return result.stdout, result.stderr, result.return_code

        self._kernel = RemoteKernel(run, cell_timeout_sec=self._cell_timeout_sec)
        await self._kernel.install(root_exec=as_root, agent_exec=as_agent)

    def _endpoint(self) -> tuple:
        values = {}
        for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY"):
            found = self._resolve_env(key)
            if found is None:
                raise RuntimeError(f"{key} is required for the {self.name()} agent")
            values[key] = found[1]
        values = resolve_env_vars(values)
        return values["OPENAI_BASE_URL"].rstrip("/"), values["OPENAI_API_KEY"]

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if self._kernel is None:
            raise RuntimeError("install() must run before run()")
        base_url, api_key = self._endpoint()
        session_id = uuid.uuid4().hex
        harness = Harness(cell_timeout_sec=self._cell_timeout_sec)

        def dump(task: Task) -> None:
            context.n_input_tokens = task.prompt_tokens
            context.n_output_tokens = task.completion_tokens
            trajectory = self._trajectory(task, session_id, harness)
            (self.logs_dir / "trajectory.json").write_text(
                format_trajectory_json(trajectory.to_json_dict())
            )

        agent = Agent(
            model=AsyncOpenAI(base_url=base_url, api_key=api_key),
            harness=harness,
            model_name=(self.model_name or "").split("/", 1)[-1],
            max_iteration=self._max_turns,
            on_step=dump,
        )
        context.n_input_tokens = context.n_output_tokens = 0
        await agent.run_task(instruction, kernel=self._kernel)

    def _trajectory(self, task: Task, session_id: str, harness: Harness) -> Trajectory:
        steps = []
        for record in task.steps:
            step_id = len(steps) + 1
            if record["source"] == "user":
                steps.append(
                    Step(
                        step_id=step_id,
                        timestamp=record["timestamp"],
                        source="user",
                        message=record["message"],
                    )
                )
                continue
            calls = [
                ToolCall(
                    tool_call_id=call["id"],
                    function_name="ipython",
                    arguments={"code": call["code"]},
                )
                for call in record["tool_calls"]
            ]
            results = [
                ObservationResult(source_call_id=o["call_id"], content=o["content"])
                for o in record["observations"]
            ]
            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=record["timestamp"],
                    source="agent",
                    model_name=self.model_name,
                    message=record["message"],
                    reasoning_content=record["reasoning"],
                    tool_calls=calls or None,
                    observation=Observation(results=results) if results else None,
                    metrics=Metrics(
                        prompt_tokens=record["prompt_tokens"],
                        completion_tokens=record["completion_tokens"],
                        extra={"latency_sec": record["latency_sec"]},
                    ),
                    llm_call_count=1,
                )
            )
        return Trajectory(
            session_id=session_id,
            agent=AtifAgent(
                name=self.name(),
                version=self.version(),
                model_name=self.model_name,
                tool_definitions=harness.tool_list,
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=task.prompt_tokens,
                total_completion_tokens=task.completion_tokens,
                total_steps=len(steps),
            ),
        )
