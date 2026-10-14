"""Load-test a decision model (`/v1/systemone`, e.g. d3) with best-of-n requests built from real runs.

    DECISION_API_KEY=... python load_test_decision.py --decision-url https://<d3 job>/v1 \
        --runs '../harbor-hf/harbor-runs/<stamp>/jobs/*/*/agent/ipyh.json' --concurrency 1,5,10,20

Each request is what experimental.BestOfN sends: a state taken at a random point of a recorded run, and
three candidates (that run's next three steps, as the decision model would see them). For each
concurrency level it sends the requests at once, `--concurrency` in flight, and reports latency
percentiles, throughput and failures. A server that runs one request at a time shows latency growing
with concurrency while throughput stays flat.
"""
import argparse
import asyncio
import glob
import json
import os
import random
import statistics
import time
from types import SimpleNamespace

from openai import AsyncOpenAI

from ipyh.experimental import BestOfN, _describe, _state


def as_message(message: dict) -> SimpleNamespace:
    """A recorded assistant message in the shape `_describe` reads (an API response message)."""
    calls = [SimpleNamespace(function=SimpleNamespace(arguments=c["function"]["arguments"]))
             for c in message.get("tool_calls") or []]
    return SimpleNamespace(content=message.get("content"), tool_calls=calls)


def build_requests(paths: list, n: int, model: str, seed: int) -> list:
    rnd = random.Random(seed)
    pool = []
    for path in paths:
        try:
            conversation = json.loads(open(path).read())["conversation"]
        except (OSError, ValueError, KeyError):
            continue
        assistant = [i for i, m in enumerate(conversation) if m["role"] == "assistant"]
        # a point with three later steps: those are the candidates
        for k in range(len(assistant) - 3):
            pool.append((conversation, assistant[k], assistant[k + 1:k + 4]))
    if not pool:
        raise SystemExit("no usable runs: --runs must match ipyh.json files with at least 4 model turns")
    requests = []
    for conversation, at, nexts in rnd.sample(pool, min(n, len(pool))):
        criteria = {f"candidate_{i + 1}": _describe(as_message(conversation[j])) for i, j in enumerate(nexts)}
        question = {"pick": {"type": "choice", "instructions": BestOfN.QUESTION, "criteria": criteria}}
        requests.append({"model": model, "state": _state(conversation[:at]), "questions": question})
    return requests


async def level(client: AsyncOpenAI, requests: list, concurrency: int) -> dict:
    gate = asyncio.Semaphore(concurrency)
    latencies, failures = [], []

    async def one(body: dict) -> None:
        async with gate:
            started = time.monotonic()
            try:
                reply = await client.post("/systemone", body=body, cast_to=object)
                reply["answers"]["pick"]["choice"]  # a usable answer, as BestOfN reads it
                latencies.append(time.monotonic() - started)
            except Exception as error:  # report, keep going
                failures.append(f"{type(error).__name__}: {str(error)[:120]}")

    started = time.monotonic()
    await asyncio.gather(*(one(b) for b in requests))
    wall = time.monotonic() - started
    q = sorted(latencies)
    pick = (lambda p: q[min(len(q) - 1, int(p * len(q)))]) if q else (lambda p: float("nan"))
    return {"concurrency": concurrency, "ok": len(q), "failed": len(failures), "p50_s": pick(0.5),
            "p90_s": pick(0.9), "max_s": q[-1] if q else float("nan"),
            "mean_s": statistics.mean(q) if q else float("nan"), "per_min": 60 * len(q) / wall,
            "failures": failures[:3]}


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decision-url", required=True, help="the decision model's /v1 URL")
    p.add_argument("--decision", default="d3", help="decision model name")
    p.add_argument("--runs", required=True, help="glob of ipyh.json files (quote it)")
    p.add_argument("--concurrency", default="1,5,10,20", help="levels, comma-separated")
    p.add_argument("--requests", type=int, default=40, help="requests per level")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None, help="also write the results as JSON")
    args = p.parse_args()

    requests = build_requests(sorted(glob.glob(args.runs)), args.requests, args.decision, args.seed)
    sizes = sorted(len(json.dumps(r)) for r in requests)
    print(f"{len(requests)} requests per level; body size median {sizes[len(sizes) // 2]} chars, max {sizes[-1]}")
    client = AsyncOpenAI(base_url=args.decision_url, api_key=os.environ.get("DECISION_API_KEY", "EMPTY"),
                         max_retries=0, timeout=600)
    results = []
    print(f"{'concurrency':>11} {'ok':>4} {'failed':>6} {'p50 s':>7} {'p90 s':>7} {'max s':>7} {'per min':>8}")
    for c in (int(x) for x in args.concurrency.split(",")):
        r = await level(client, requests, c)
        results.append(r)
        print(f"{c:>11} {r['ok']:>4} {r['failed']:>6} {r['p50_s']:>7.2f} {r['p90_s']:>7.2f} {r['max_s']:>7.2f} {r['per_min']:>8.1f}",
              flush=True)
        for failure in r["failures"]:
            print(f"{'':>13}{failure}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
