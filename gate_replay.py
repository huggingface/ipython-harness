"""Can a decision model (`/v1/systemone`, e.g. d3) gate an agent? Replays recorded runs and scores it.

    DECISION_API_KEY=... python gate_replay.py --decision-url https://<d3 job>/v1 --output gate.jsonl \
        --trials '../harbor-hf/harbor-runs/<stamp>/jobs/ipyh--*/*/' [--trials ...] [--steps 300]

Two tests on Harbor trial folders (result.json + agent/ipyh.json), as experimental.BestOfN states them:

- final: at each trial's final answer, "has the task been completed exactly as specified?" (yes/no).
  Label: the verifier's reward (1.0 = completed). A useful finish gate gives failed trials a low yes.
- step: before a random step, "will this step make real progress?" (yes/no), the step shown as the
  candidate. Label: whether the step's output was an error (traceback, failed command, timeout).
  A useful step gate gives the steps that errored a low yes.

Prints the AUROC of each test (0.5 = chance, 1.0 = perfect) and what share of items each yes-threshold
would reject; every item is written to --output (JSON lines) for further analysis.
"""
import argparse
import asyncio
import glob
import json
import os
import random
import re

from openai import AsyncOpenAI

from ipyh.experimental import _describe, _state
from load_test_decision import as_message

INFRA = {"RemoteProtocolError", "ReadTimeout", "ReadError", "ConnectTimeout", "ConnectError", "SandboxError",
         "RuntimeError", "NetworkConnectionError", "CancelledError", "VerifierTimeoutError"}
ERROR = re.compile(r"Traceback \(most recent call last\)|\w+Error:|command not found|No such file or directory"
                   r"|\[cell timed out|exited with|returned non-zero")

FINAL = {
    "type": "noul",
    "instructions": "Has the task been completed exactly as specified, with every requirement met?",
    "criteria": {"true": "the work shown meets every requirement of the task",
                 "false": "some requirement is missing, wrong or not verified"},
}


def step_question(step: str) -> dict:
    return {
        "type": "noul",
        "instructions": f"Proposed next step:\n{step}\n\nWill running this step make real progress toward completing "
                        "the task exactly as specified?",
        "criteria": {"true": "a useful, correct next step", "false": "a wrong, wasted or harmful step"},
    }


def load(patterns: list) -> list:
    trials = []
    for folder in sorted({f for p in patterns for f in glob.glob(p)}):
        try:
            result = json.load(open(os.path.join(folder, "result.json")))
            record = json.load(open(os.path.join(folder, "agent", "ipyh.json")))
        except (OSError, ValueError):
            continue
        error = (result.get("exception_info") or {}).get("exception_type")
        if error in INFRA:
            continue
        reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward") or 0.0
        trials.append({"trial": os.path.basename(folder.rstrip("/")), "task": result["task_name"], "error": error,
                       "reward": reward, "conversation": record["conversation"], "done": record.get("done")})
    return trials


def items(trials: list, n_steps: int, seed: int) -> list:
    out = []
    for t in trials:  # final answers of trials that ended on their own
        conv = t["conversation"]
        if t["error"] is None and t["done"] and conv[-1]["role"] == "assistant" and not conv[-1].get("tool_calls"):
            out.append({"test": "final", "trial": t["trial"], "task": t["task"], "label": t["reward"] >= 0.999,
                        "reward": t["reward"], "body": {"state": _state(conv), "questions": {"q": FINAL}}})
    steps = []
    for t in trials:
        conv = t["conversation"]
        for i, m in enumerate(conv):
            if m["role"] == "assistant" and m.get("tool_calls") and i >= 2:
                output = "".join(str(x.get("content")) for x in conv[i + 1:i + 1 + len(m["tool_calls"])])
                steps.append((t, i, bool(ERROR.search(output))))
    for t, i, errored in random.Random(seed).sample(steps, min(n_steps, len(steps))):
        conv = t["conversation"]
        out.append({"test": "step", "trial": t["trial"], "task": t["task"], "label": not errored,
                    "reward": t["reward"], "turn": i,
                    "body": {"state": _state(conv[:i]), "questions": {"q": step_question(_describe(as_message(conv[i])))}}})
    return out


def auroc(scores_true: list, scores_false: list) -> float:
    """P(a positive item scores above a negative one), ties count half."""
    if not scores_true or not scores_false:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in scores_true for b in scores_false)
    return wins / (len(scores_true) * len(scores_false))


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decision-url", required=True)
    p.add_argument("--decision", default="d3")
    p.add_argument("--trials", action="append", required=True, help="glob of Harbor trial folders (repeatable)")
    p.add_argument("--steps", type=int, default=300, help="random steps to score")
    p.add_argument("--concurrency", type=int, default=4, help="requests in flight (the server may run one at a time)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True, help="JSON lines, one scored item each")
    args = p.parse_args()

    trials = load(args.trials)
    todo = items(trials, args.steps, args.seed)
    print(f"{len(trials)} trials; {sum(i['test'] == 'final' for i in todo)} final answers, "
          f"{sum(i['test'] == 'step' for i in todo)} steps to score", flush=True)
    client = AsyncOpenAI(base_url=args.decision_url, api_key=os.environ.get("DECISION_API_KEY", "EMPTY"),
                         max_retries=2, timeout=600)
    gate, done = asyncio.Semaphore(args.concurrency), 0

    async def score(item: dict) -> None:
        nonlocal done
        async with gate:
            try:
                reply = await client.post("/systemone", body={"model": args.decision, **item["body"]}, cast_to=object)
                answer = reply["answers"]["q"]
                item["yes"] = answer.get("noul")  # a noul answer is {"type": "noul", "noul": P(true)}
                item["answer_error"] = answer.get("error") or (None if item["yes"] is not None else f"no `noul` in {answer}"[:200])
            except Exception as error:  # report, keep going
                item["yes"], item["answer_error"] = None, f"{type(error).__name__}: {str(error)[:200]}"
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(todo)}", flush=True)

    await asyncio.gather(*(score(i) for i in todo))
    with open(args.output, "w") as f:
        for i in todo:
            f.write(json.dumps({k: v for k, v in i.items() if k != "body"} | {"state_chars": len(i["body"]["state"])}) + "\n")

    for test, what in (("final", "completed (reward 1) vs not"), ("step", "step ran cleanly vs errored")):
        scored = [i for i in todo if i["test"] == test and i["yes"] is not None]
        good, bad = [i["yes"] for i in scored if i["label"]], [i["yes"] for i in scored if not i["label"]]
        failed = sum(1 for i in todo if i["test"] == test and i["yes"] is None)
        print(f"\n[{test}] {what}: {len(good)} vs {len(bad)} items ({failed} not scored)")
        print(f"  AUROC {auroc(good, bad):.3f}   mean yes: {sum(good) / max(1, len(good)):.2f} vs {sum(bad) / max(1, len(bad)):.2f}")
        for th in (0.3, 0.5, 0.7):
            rej_good = sum(y < th for y in good) / max(1, len(good))
            rej_bad = sum(y < th for y in bad) / max(1, len(bad))
            print(f"  reject when yes < {th}: rejects {rej_bad:.0%} of the bad, {rej_good:.0%} of the good")


if __name__ == "__main__":
    asyncio.run(main())
