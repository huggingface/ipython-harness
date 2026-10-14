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
    parser = argparse.ArgumentParser(prog="python -m ipyh", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"), help="OpenAI-compatible /v1 URL")
    parser.add_argument("--model", default=None, help="served model name (default: the first one listed)")
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON file with the conversation and per-turn usage")
    parser.add_argument("--python", default=sys.executable, help="interpreter named in the system prompt; 'none' for a private one")
    parser.add_argument("--time-limit-sec", type=float, default=None, help="when set, tool results show the time used so far")
    parser.add_argument("--policy", choices=["single", "best-of-n", "ladder"], default="single",
                        help="best-of-n (experimental): --generators propose each turn, --decision picks one; "
                             "ladder (experimental): thinking rises when --gate rejects a step")
    parser.add_argument("--gate", choices=["none", "signals", "decision"], default="none",
                        help="ladder: what rejects a step (decision: --decision at --decision-url)")
    parser.add_argument("--thinking-start", default="off", help="ladder: off, low, medium or xhigh")
    parser.add_argument("--thinking-top", default="xhigh", help="ladder: highest level; its step is always run")
    parser.add_argument("--gate-threshold", type=float, default=0.9, help="ladder with decision: lowest accepted score")
    parser.add_argument("--generators", default="", help="comma-separated model names served at --base-url")
    parser.add_argument("--decision", default=None, help="decision model name (/v1/systemone)")
    parser.add_argument("--decision-url", default=None,
                        help="its own /v1 URL when not served at --base-url; key from DECISION_API_KEY")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    if args.policy == "best-of-n":
        from ipyh.experimental import BestOfN

        agent = BestOfN(client, Harness(), generators=args.generators.split(","), decision_model=args.decision,
                        decision_url=args.decision_url)
    elif args.policy == "ladder":
        from ipyh.experimental import ThinkingLadder

        agent = ThinkingLadder(client, Harness(), gate=args.gate, start=args.thinking_start, top=args.thinking_top,
                               decision_model=args.decision, decision_url=args.decision_url,
                               threshold=args.gate_threshold)
    else:
        agent = Harness() + client
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
