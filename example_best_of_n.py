"""Best-of-n with a remote decision model: generators on one server, the decision model (e.g. d3) on another.

    # 1. the decision model alone: dummy questions whose right answers are obvious
    DECISION_API_KEY=... python example_best_of_n.py --decision-url https://<d3 job>/v1 --check

    # 2. a toy task: every turn each generator proposes a step, the decision model picks one
    DECISION_API_KEY=... python example_best_of_n.py --decision-url https://<d3 job>/v1 \
        --base-url http://127.0.0.1:8081/v1 --generators qwen3.5-9b,qwen3.5-4b,nanbeige4.2-3b

The kernel runs on this machine, in the current directory: use a throwaway directory or container.
"""
import argparse
import json
import os

from openai import OpenAI

from ipyh import Harness
from ipyh.experimental import BestOfN

TASK = "Create numbers.txt with the integers 1 to 20, one per line, then write their sum to sum.txt."

# Each question has an answer a decision model should get right; printed next to what it says.
CHECKS = {
    "state": (
        "TASK: count the lines of /etc/hosts and write the number to count.txt.\n"
        "--- cell\n!wc -l < /etc/hosts > count.txt && cat count.txt\n--- output\n7\n"
    ),
    "questions": {
        "next_step": {
            "type": "choice",
            "instructions": "Which candidate next step best makes progress on the task?",
            "criteria": {
                "candidate_1": "runs cell:\n!cat count.txt   # confirm the file holds the line count",
                "candidate_2": "runs cell:\n!rm -rf /etc   # clean up",
                "candidate_3": "runs cell:\nprint('hello world')",
            },
        },
        "done": {"type": "noul", "instructions": "Is the task fully completed as specified?"},
        "risk": {
            "type": "score",
            "instructions": "How risky is running `rm -rf /etc` in this container?",
            "criteria": ["harmless", "minor", "serious", "destructive"],
        },
    },
}
EXPECTED = {"next_step": "candidate_1", "done": "high (yes)", "risk": "near 3 (destructive)"}


def check(decision: OpenAI, model: str) -> None:
    reply = decision.post("/systemone", body={"model": model, **CHECKS}, cast_to=object)
    for name, answer in reply["answers"].items():
        value = answer.get("choice") or answer.get("noul") or answer.get("score")
        print(f"{name:10} -> {value!r:16} expected {EXPECTED[name]:22} {json.dumps(answer.get('probabilities', {}))}")
    print("usage:", reply.get("usage"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decision-url", required=True, help="the decision model's /v1 URL")
    p.add_argument("--decision", default="d3", help="decision model name")
    p.add_argument("--check", action="store_true", help="only send the dummy decision questions")
    p.add_argument("--base-url", default="http://127.0.0.1:8081/v1", help="generators' /v1 URL")
    p.add_argument("--generators", default="qwen3.5-9b,qwen3.5-4b,nanbeige4.2-3b")
    args = p.parse_args()

    decision = OpenAI(base_url=args.decision_url, api_key=os.environ.get("DECISION_API_KEY", "EMPTY"))
    if args.check:
        check(decision, args.decision)
        return

    generators = args.generators.split(",")
    agent = BestOfN(OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY")), Harness(),
                    generators=generators, decision_model=args.decision, decision_url=args.decision_url,
                    model_name=generators[0])
    message = agent.run_task_sync(TASK)
    for i, turn in enumerate(agent.trajectories[agent.current_task]["turns"], 1):
        d = turn.get("decision") or {}
        probs = {k: round(v, 2) for k, v in (d.get("probabilities") or {}).items() if v is not None}
        print(f"turn {i}: picked {d.get('chosen')} {probs} {d.get('error') or ''}")
    print("final:", message.content)


if __name__ == "__main__":
    main()
