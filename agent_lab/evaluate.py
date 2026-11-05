"""
Runs each approach over the same job descriptions and writes what happened to results/.

    python -m agent_lab.evaluate --env path/to/.env --only loop graph crew
    python -m agent_lab.evaluate --only local
    python -m agent_lab.evaluate --trace platform      # one LangGraph run, step by step, for the portfolio's replay

The check on every result is the same validator, so frameworks can't mark their own homework.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import loop
from .common import MODEL, Meter, fact_index, gemini, load_candidate, load_jds, load_key
from .loop import text_of
from .validator import check

RESULTS = Path(__file__).resolve().parent.parent / "results"
sys.stdout.reconfigure(encoding="utf-8")


def raw(llm, jd, facts, index, config):
    """No retry and no check inside: how the model behaves on its own, measured afterwards."""
    return loop.run(llm, jd, facts, index, config=config, max_attempts=1)


def run_one(name, jd, facts, index, model):
    meter = Meter()
    config = {"callbacks": [meter]}
    if name == "raw":
        out = raw(gemini(model), jd, facts, index, config)
    elif name == "loop":
        out = loop.run(gemini(model), jd, facts, index, config=config)
    elif name == "graph":
        from . import graph
        out = graph.run(gemini(model), jd, facts, index, config=config)
    elif name == "crew":
        from . import crew
        out = crew.run(jd, facts, index, model)
    elif name == "local":
        from .local import load
        global _local
        _local = _local or load()
        out = loop.run(_local, jd, facts, index, config=config)
    else:
        raise SystemExit(f"unknown approach {name}")
    llm_calls = out.pop("llm_calls", None)
    tokens = out.pop("tokens", None)
    return {"approach": name, "jd": jd["id"], "seconds": meter.seconds, "llm_calls": llm_calls if llm_calls is not None else meter.calls, "tokens": tokens if tokens is not None else meter.tokens, **out}


_local = None


def summarise(rows):
    out = {}
    for name in dict.fromkeys(r["approach"] for r in rows):
        rs = [r for r in rows if r["approach"] == name and "error" not in r]
        n = len(rs) or 1
        out[name] = {
            "runs": len(rs),
            "errors": sum(1 for r in rows if r["approach"] == name and "error" in r),
            "first_draft_clean": sum(bool(r["first_ok"]) for r in rs),
            "final_clean": sum(bool(r["final_ok"]) for r in rs),
            "avg_llm_calls": round(sum(r["llm_calls"] for r in rs) / n, 1),
            "avg_tokens": round(sum(r["tokens"] or 0 for r in rs) / n),
            "avg_seconds": round(sum(r["seconds"] for r in rs) / n, 1),
        }
    return out


def trace(jd_id, facts, index, model):
    """One real LangGraph run, written out step by step. Messages only: there's no key in this data."""
    from . import graph
    steps, t0 = [], time.perf_counter()

    def on_step(node, update):
        step = {"node": node, "t": round(time.perf_counter() - t0, 1)}
        if node == "agent":
            msg = update["messages"][-1]
            calls = [{"name": c["name"], "args": c["args"]} for c in msg.tool_calls]
            step.update(tool_calls=calls, text=text_of(msg).strip())
        elif node == "tools":
            step["results"] = [{"name": m.name, "text": text_of(m)} for m in update["messages"]]
        elif node == "guard":
            step.update(draft=update["draft"], problems=update["problems"], attempt=update["attempts"])
        elif node == "human":
            step.update(update if isinstance(update, dict) else {})
        steps.append(step)

    jd = next(j for j in load_jds() if j["id"] == jd_id)
    result = graph.run(gemini(model), jd, facts, index, on_step=on_step)
    (RESULTS / f"trace_{jd_id}.json").write_text(json.dumps({"model": model, "jd": jd, "facts": facts, "steps": steps, "result": result}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"trace_{jd_id}.json: {len(steps)} steps, first_ok={result['first_ok']}, final_ok={result['final_ok']}, attempts={result['attempts']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", help=".env file holding GEMINI_API_KEY")
    ap.add_argument("--only", nargs="+", default=["raw", "loop", "graph", "crew"])
    ap.add_argument("--jds", nargs="+")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--trace")
    args = ap.parse_args()
    load_key(args.env)
    RESULTS.mkdir(exist_ok=True)
    facts = load_candidate()["facts"]
    index = fact_index(facts)
    if args.trace:
        return trace(args.trace, facts, index, args.model)
    jds = [j for j in load_jds() if not args.jds or j["id"] in args.jds]
    path = RESULTS / "eval.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"rows": []}
    rows = [r for r in previous["rows"] if r["approach"] not in args.only]
    for name in args.only:
        for jd in jds:
            print(f"{name} / {jd['id']} ...", end=" ", flush=True)
            try:
                row = run_one(name, jd, facts, index, args.model)
                print(f"first_ok={row['first_ok']} final_ok={row['final_ok']} calls={row['llm_calls']} {row['seconds']}s")
            except Exception as exc:  # A failure is a result too: it's recorded, not hidden
                row = {"approach": name, "jd": jd["id"], "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
                print("ERROR", row["error"][:120])
            rows.append(row)
            path.write_text(json.dumps({"model": args.model, "rows": rows, "summary": summarise(rows)}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summarise(rows), indent=1))


if __name__ == "__main__":
    main()
