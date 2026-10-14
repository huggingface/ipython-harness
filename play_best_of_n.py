"""Step through best-of-n runs in the browser, one decision at a time.

    DECISION_API_KEY=... python play_best_of_n.py --runs-dir runs \
        --base-url http://127.0.0.1:8081/v1 --generators qwen3.5-9b,qwen3.5-4b,nanbeige4.2-3b \
        --decision d3 --decision-url https://<d3 job>/v1 --port 8765 [--task-file task.md]

Open http://127.0.0.1:8765 and type a task in "New task" (or pass --task-file to start one right
away). Each turn the generators propose and the decision model picks; the run then waits, showing
every candidate and the pick, until you run the pick, run another candidate instead, or stop. A new
task stops the current run at its next decision and starts fresh: a new kernel, in its own folder
under --work-dir. Each run is saved to --runs-dir after every turn (view_best_of_n.py renders it).

The kernel runs on this machine: run this in a throwaway container.
"""
import argparse
import asyncio
import json
import os
import queue
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs

from openai import AsyncOpenAI

from ipyh import Harness, __version__
from ipyh.experimental import BestOfN
from view_best_of_n import esc, render


class Stop(Exception):
    """The person stopped the run from the page, or started another task."""


class Board:
    """What the page shows, shared between the agent and the web server threads."""

    def __init__(self, runs_dir: Path):
        self.lock = threading.Lock()
        self.runs_dir = runs_dir
        self.title, self.record, self.pending, self.status = "", {}, None, "waiting for a task"
        self.actions: "queue.Queue[str]" = queue.Queue()  # clicks on a waiting decision
        self.tasks: "queue.Queue[str]" = queue.Queue()  # new tasks, run one after another

    def update(self, record=None, pending=None, status="", title: Optional[str] = None):
        # A JSON round trip snapshots the agent's lists, so the page never reads them mid-change
        with self.lock:
            if title is not None:
                self.title = title
            if record is not None:
                self.record = json.loads(json.dumps(record))
            self.pending = json.loads(json.dumps(pending)) if pending else None
            self.status = status

    def new_task(self, text: str) -> None:
        self.tasks.put(text)
        self.actions.put("stop")  # releases a run waiting on a decision; ignored when none is

    def header(self) -> str:
        runs = sorted(self.runs_dir.glob("run-*.json"), reverse=True)
        items = "".join(f"<li><a href='/run/{esc(p.name)}'>{esc(p.stem)}</a></li>" for p in runs[:20])
        busy = self.status not in ("waiting for a task", "finished", "stopped") and not self.status.startswith("failed")
        return (
            "<div class='newtask'><form method='post' action='/task'>"
            "<b>New task</b>" + (" <span class='muted'>(stops the current run at its next decision)</span>" if busy else "")
            + "<textarea name='task' placeholder='Describe the task for the models…'></textarea>"
            "<button class='primary'>Start</button></form>"
            + (f"<details><summary>Previous runs</summary><ul class='runs'>{items}</ul></details>" if items else "")
            + "</div>"
        )

    def page(self) -> str:
        with self.lock:
            idle = self.status in ("waiting for a task", "finished", "stopped") or self.status.startswith("failed")
            return render(self.record, self.title or "no run yet", refresh=self.pending is None and not idle,
                          pending=self.pending, status=self.status, header=self.header())


class SteppedBestOfN(BestOfN):
    """BestOfN that waits for a person after each decision."""

    def __init__(self, *args, board: Board, snapshot, **kwargs):
        super().__init__(*args, **kwargs)
        self.board, self.snapshot = board, snapshot

    async def review(self, choice: dict, index: int) -> int:
        if not self.board.tasks.empty():  # a new task arrived while the models were working
            raise Stop()
        while not self.board.actions.empty():  # drop clicks made before this decision was shown
            self.board.actions.get_nowait()
        self.board.update(self.snapshot(), pending=choice, status="waiting for you")
        action = await asyncio.to_thread(self.board.actions.get)
        self.board.update(self.snapshot(), status="running the step, then asking the models")
        if action == "stop":
            raise Stop()
        if action.startswith("pick:"):
            picked = int(action.split(":", 1)[1])
            if 0 <= picked < len(choice["candidates"]):
                return picked
        return index


def serve(board: Board, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _html(self, text: str, code: int = 200):
            body = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/run/"):  # a saved run, read-only
                name = self.path[len("/run/"):]
                match = [p for p in board.runs_dir.glob("run-*.json") if p.name == name]
                if not match:
                    return self._html("not found", 404)
                record = json.loads(match[0].read_text())
                return self._html(render(record, name, refresh=False, header="<p><a href='/'>← back</a></p>"))
            self._html(board.page())

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            form = parse_qs(self.rfile.read(length).decode())
            if self.path == "/task" and form.get("task", [""])[0].strip():
                board.new_task(form["task"][0].strip())
            elif self.path == "/action" and form.get("action"):
                board.actions.put(form["action"][0])
            self.send_response(303)  # back to the page, so a reload does not repeat the click
            self.send_header("Location", "/")
            self.end_headers()

        def log_message(self, *args):  # keep the terminal quiet
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task-file", type=Path, default=None, help="start with this task (more from the page)")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"), help="one run-<time>.json per task")
    p.add_argument("--work-dir", type=Path, default=Path("tasks"), help="each task runs in its own folder here")
    p.add_argument("--base-url", default="http://127.0.0.1:8081/v1", help="generators' /v1 URL")
    p.add_argument("--generators", default="qwen3.5-9b,qwen3.5-4b,nanbeige4.2-3b")
    p.add_argument("--decision", default="d3", help="decision model name")
    p.add_argument("--decision-url", default=None, help="its own /v1 URL; key from DECISION_API_KEY")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    generators = args.generators.split(",")
    runs_dir, work_dir = args.runs_dir.resolve(), args.work_dir.resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    board = Board(runs_dir)
    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    async def run_one(task: str) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = runs_dir / f"run-{stamp}.json"
        folder = work_dir / stamp
        folder.mkdir(parents=True, exist_ok=True)
        os.chdir(folder)  # the kernel starts in the current directory

        def snapshot() -> dict:
            state = agent.trajectories[agent.current_task]
            return {"version": __version__, "model": "best-of-n", "generators": generators, "decision": args.decision,
                    "done": state["done"], "conversation": state["trajectory"], "turns": state["turns"]}

        def save(status: str) -> None:
            record = snapshot()
            output.write_text(json.dumps(record, indent=1))
            board.update(record, status=status)

        agent = SteppedBestOfN(client, Harness(), generators=generators, decision_model=args.decision,
                               decision_url=args.decision_url, model_name=generators[0].partition("=")[0],
                               board=board, snapshot=snapshot)
        board.update({"conversation": [{}, {"content": task}]}, status="starting the kernel, asking the models",
                     title=output.name)
        print(f"[{stamp}] new task, saved to {output}", flush=True)
        try:
            await agent.run_task(task, debug=True)
            save("asking the models")
            while not agent.trajectories[agent.current_task]["done"]:
                await agent.run_step()
                save("asking the models")
            save("finished")
        except Stop:
            save("stopped")
        except Exception as error:  # e.g. a generator server that is down: show it, keep the page up
            board.update(status=f"failed: {type(error).__name__}: {str(error)[:200]}")
            print(f"[{stamp}] failed: {error!r}", flush=True)
        finally:
            try:
                agent.shutdown_kernel()
            except Exception:  # the kernel never started
                pass

    async def loop() -> None:
        if args.task_file:
            board.tasks.put(args.task_file.read_text())
        while True:
            task = await asyncio.to_thread(board.tasks.get)
            while not board.actions.empty():  # a "stop" queued with this task has done its job
                board.actions.get_nowait()
            await run_one(task)

    server = serve(board, args.port)
    print(f"open http://127.0.0.1:{args.port}  (models: {', '.join(generators)}; decisions: {args.decision})", flush=True)
    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
