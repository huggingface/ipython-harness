"""Render a best-of-n run (`python -m ipyh --policy best-of-n --output run.json`) as one HTML page.

    python view_best_of_n.py run.json            # writes run.html next to it
    python view_best_of_n.py run.json --watch    # rewrites it every few seconds while the run goes on

Each turn shows every generator's candidate as the decision model read it, the probability it gave
each one, the one it picked, and what the picked cell printed. Open the HTML file in a browser; with
--watch the page reloads itself. play_best_of_n.py serves the same page with buttons to step through.
"""
import argparse
import html
import json
import time
from pathlib import Path
from typing import Optional

STYLE = """
:root { --bg: #fbfaf7; --fg: #1f1e1c; --muted: #6b6862; --card: #ffffff; --line: #e4e1da;
        --pick: #2a78d4; --pick-bg: #eef4fc; --bar: #c9c5bc; --code: #f4f2ed; --err: #b3261e; --wait: #a15c00; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #1b1a18; --fg: #ecebe8; --muted: #a29f98; --card: #24231f; --line: #3a3833;
          --pick: #6aa8f0; --pick-bg: #1f2a38; --bar: #55524b; --code: #2c2b27; --err: #f2b8b5; --wait: #f0b45c; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 64px; background: var(--bg); color: var(--fg);
       font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1400px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
.meta, .muted { color: var(--muted); }
details.task { margin: 12px 0 24px; }
section.turn { border-top: 1px solid var(--line); padding: 18px 0; }
section.pending h2 { color: var(--wait); }
.turn h2 { font-size: 15px; margin: 0 0 10px; }
.cands { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
.cand { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; min-width: 0; }
.cand.picked { border: 2px solid var(--pick); background: var(--pick-bg); }
.cand header { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
.cand .name { font-weight: 600; }
.badge { font-size: 12px; font-weight: 600; color: var(--pick); }
.prob { height: 6px; background: var(--line); border-radius: 3px; margin: 6px 0; overflow: hidden; }
.prob > span { display: block; height: 100%; background: var(--bar); }
.picked .prob > span { background: var(--pick); }
pre { background: var(--code); border-radius: 6px; padding: 8px; margin: 6px 0 0; white-space: pre-wrap;
      word-break: break-word; font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; max-height: 420px; overflow: auto; }
summary { cursor: pointer; color: var(--muted); margin-top: 6px; }
.out { margin-top: 10px; }
.err { color: var(--err); }
.controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 4px; }
form { display: inline; margin: 0; }
button { font: inherit; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--line);
         background: var(--card); color: var(--fg); cursor: pointer; }
button.primary { background: var(--pick); border-color: var(--pick); color: #fff; }
.cand button { margin-top: 8px; font-size: 12px; padding: 4px 10px; }
.newtask { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; margin-bottom: 20px; }
.newtask textarea { width: 100%; min-height: 90px; margin: 8px 0; padding: 8px; border-radius: 6px; border: 1px solid var(--line);
                    background: var(--bg); color: var(--fg); font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; resize: vertical; }
.runs { margin: 8px 0 0; padding-left: 18px; }
.runs a { color: var(--pick); }
"""

# Keep the newest turn in view after each reload
SCROLL = "<script>document.querySelector('section.turn:last-of-type')?.scrollIntoView({block: 'start'});</script>"
# Reload every 3 s, except while someone is typing in a text box (a reload would lose the text)
RELOAD = ("<script>setInterval(() => { const t = document.querySelector('textarea');"
          " if (!t || (!t.value && document.activeElement !== t)) location.reload(); }, 3000);</script>")


def esc(text) -> str:
    return html.escape(str(text or ""))


def pre(text) -> str:
    return f"<pre>{esc(text)}</pre>"


def steps(conversation: list) -> list:
    """Per assistant message (one per turn), the tool outputs and notes that follow it."""
    out = []
    for message in conversation[2:]:
        if message["role"] == "assistant":
            out.append({"assistant": message, "after": []})
        elif out:
            out[-1]["after"].append(message)
    return out


def candidates(decision: dict, buttons: bool) -> str:
    """The candidate cards of one decision; with `buttons`, each card can be picked instead."""
    probs, chosen = decision.get("probabilities") or {}, decision.get("chosen")
    cards = []
    for i, c in enumerate(decision.get("candidates") or []):
        p, picked = probs.get(c["model"]), c["model"] == chosen
        tokens = f"{c.get('completion_tokens')} out" if c.get("completion_tokens") is not None else ""
        if c.get("seconds") is not None:
            tokens += f" · {c['seconds']:.0f}s"
        label = "picked by you" if picked and decision.get("overridden") else "picked"
        cards.append(
            f"<div class='cand{' picked' if picked else ''}'><header><span class='name'>{esc(c['model'])}</span>"
            f"<span>{f'<span class=badge>{label}</span> ' if picked else ''}"
            f"<span class='muted'>{f'p={p:.2f}' if p is not None else ''} {tokens}</span></span></header>"
            f"<div class='prob'><span style='width:{(p or 0) * 100:.0f}%'></span></div>"
            f"{pre(c.get('shown'))}"
            + (f"<details><summary>Reasoning</summary>{pre(c['reasoning'])}</details>" if c.get("reasoning") else "")
            + (f"<form method='post' action='/action'><button name='action' value='pick:{i}'>Run this one instead</button></form>"
               if buttons and not picked else "")
            + "</div>"
        )
    return f"<div class='cands'>{''.join(cards)}</div>"


def from_ladder(ladder: dict) -> dict:
    """A ThinkingLadder turn in the shape of a decision: one card per attempt, the accepted one picked."""
    attempts = ladder.get("attempts") or []
    accepted = next((a for a in attempts if a["accepted"]), attempts[-1] if attempts else {})
    return {
        "chosen": accepted.get("level"),
        "probabilities": {a["level"]: a["score"] for a in attempts if a.get("score") is not None},
        "rejected": {a["level"]: a["reason"] for a in attempts if not a["accepted"]},
        "candidates": [{"model": a["level"], "shown": a.get("shown"), "completion_tokens": a.get("completion_tokens"),
                        "seconds": a.get("seconds")} for a in attempts],
    }


def decision_notes(decision: dict) -> str:
    notes = []
    for level, reason in (decision.get("rejected") or {}).items():
        notes.append(f"<div class='muted'>thinking {esc(level)}: not run, {esc(reason)}</div>")
    if decision.get("decision_sec") is not None:
        notes.append(f"<div class='muted'>decision took {decision['decision_sec']:.1f}s</div>")
    for model, error in (decision.get("failed") or {}).items():
        notes.append(f"<div class='err'>{esc(model)} failed this turn: {esc(error)}</div>")
    if decision.get("error"):
        notes.append(f"<div class='err'>decision failed, fell back to the first candidate: {esc(decision['error'])}</div>")
    if decision.get("state"):
        notes.append(f"<details><summary>State sent to the decision model</summary>{pre(decision['state'])}</details>")
    return "".join(notes)


def render(record: dict, title: str, refresh: bool, pending: Optional[dict] = None, status: str = "",
           header: str = "") -> str:
    """The run as HTML. `pending`: a decision waiting for a person (play_best_of_n.py), shown with buttons.
    `header`: extra HTML above the run (play_best_of_n.py's new-task form and run list)."""
    conversation, turns = record.get("conversation") or [], record.get("turns") or []
    task = conversation[1]["content"] if len(conversation) > 1 else ""
    state = status or ("done" if record.get("done") else "in progress")
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Best-of-n run</title><style>{STYLE}</style></head><body><main>",
        header,
        f"<h1>Best-of-n run</h1><div class='meta'>{esc(title)} · {len(turns)} turns · {esc(state)}</div>",
        f"<details class='task' open><summary>Task</summary>{pre(task)}</details>" if task else "",
    ]
    for n, (turn, step) in enumerate(zip(turns, steps(conversation)), 1):
        decision = turn.get("decision") or (from_ladder(turn["ladder"]) if turn.get("ladder") else {})
        if turn.get("ladder"):
            title_html = (f"Turn {n} · ran at thinking <b>{esc(decision['chosen'])}</b>"
                          f" ({len(decision['candidates'])} attempt{'s' if len(decision['candidates']) > 1 else ''})")
        elif decision.get("overridden"):
            title_html = f"Turn {n} · you picked <b>{esc(decision['chosen'])}</b> (the decision model picked {esc(decision['overridden'])})"
        elif decision.get("chosen"):
            title_html = f"Turn {n} · picked <b>{esc(decision['chosen'])}</b>"
        else:
            title_html = f"Turn {n} · single model"
        parts.append(f"<section class='turn'><h2>{title_html}</h2>{decision_notes(decision)}")
        if decision.get("candidates"):
            parts.append(candidates(decision, buttons=False))
        else:  # single-model turn: show the reply itself
            message = step["assistant"]
            cells = [json.loads(c["function"]["arguments"] or "{}").get("code", "") for c in message.get("tool_calls") or []]
            parts.append(pre("\n".join([message.get("content") or ""] + cells)))
        for message in step["after"]:
            label = "Output" if message["role"] == "tool" else "Note to the model"
            parts.append(f"<div class='out'><span class='muted'>{label}</span>{pre(message['content'])}</div>")
        parts.append("</section>")
    if pending:
        parts.append(
            f"<section class='turn pending'><h2>Turn {len(turns) + 1} · the decision model picks "
            f"<b>{esc(pending.get('chosen'))}</b>; nothing has run yet</h2>{decision_notes(pending)}"
            "<div class='controls'><form method='post' action='/action'>"
            "<button class='primary' name='action' value='accept'>Run its pick, go to the next turn</button></form>"
            "<form method='post' action='/action'><button name='action' value='stop'>Stop the run</button></form></div>"
            f"{candidates(pending, buttons=True)}</section>"
        )
    parts.append(f"{SCROLL}{RELOAD if refresh else ''}</main></body></html>")
    return "".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="JSON written by `python -m ipyh --output`")
    p.add_argument("-o", "--html", type=Path, default=None, help="output page (default: next to the run, .html)")
    p.add_argument("--watch", action="store_true", help="rewrite the page every 3 s; the page reloads itself")
    args = p.parse_args()
    target = args.html or args.run.with_suffix(".html")
    while True:
        try:
            record = json.loads(args.run.read_text())
        except (OSError, json.JSONDecodeError):  # not written yet, or caught mid-rewrite
            record = None
        if record is not None:
            target.write_text(render(record, args.run.name, refresh=args.watch))
        if not args.watch:
            print(f"wrote {target}")
            return
        time.sleep(3)


if __name__ == "__main__":
    main()
