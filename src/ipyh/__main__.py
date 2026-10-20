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
"""Run one task and save the conversation:

    OPENAI_API_KEY=... python -m ipyh --base-url URL --model NAME --instruction-file task.md --output run.json

The kernel runs on this machine, in the current directory: use a throwaway container. The output is
rewritten after every turn, so a run that is killed still leaves its conversation so far.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

from ipyh import Harness, __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ipyh",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible /v1 URL",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="served model name (default: the first one listed)",
    )
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON file with the conversation and per-turn usage",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter named in the system prompt; 'none' for a private one",
    )
    parser.add_argument(
        "--harness-config",
        type=Path,
        help="JSON file with a system_prompt and kernel tools (see Harness.from_config)",
    )
    parser.add_argument(
        "--time-limit-sec",
        type=float,
        default=None,
        help="when set, tool results show the time used so far",
    )
    args = parser.parse_args()

    harness = Harness.from_config(args.harness_config) if args.harness_config else Harness()
    agent = harness + AsyncOpenAI(
        base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY")
    )
    agent.model_name = args.model
    agent.python = None if args.python.lower() == "none" else args.python
    agent.time_limit_sec = args.time_limit_sec

    def save() -> None:
        state = agent.trajectories[agent.current_task]
        record = {
            "version": __version__,
            "model": agent.model_name,
            "done": state["done"],
            "conversation": state["trajectory"],
            "turns": state["turns"],
        }
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1))
        tmp.replace(args.output)

    async def run() -> None:
        await agent.run_task(args.instruction_file.read_text(), debug=True)
        try:
            save()
            while not agent.trajectories[agent.current_task]["done"]:
                await agent.run_step()
                save()
        finally:
            agent.shutdown_kernel()

    asyncio.run(run())


if __name__ == "__main__":
    main()
