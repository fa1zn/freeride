"""Can a model help when nobody spells it out?

The third half of this repo, and the first to run a model. `screen.py` measures
what a policy scores with no perception and `intent.py` what it scores with no
task; both are floors. Every model number in the README so far is quoted from
the 2023 paper. This asks the thesis question directly, of a model we run.

THREE CONDITIONS, ONE VARIABLE APART. At every decision point (a workflow cut
after k actions) the model sees the actions so far and a list of candidate
elements on the current page, and must pick one and an operation.

    blind         no task. All candidates.        The thesis test.
    told          the task. All candidates.       What being told buys.
    told_ranked   the task. Top-50 by the authors' ranker.
                  The paper's own setting, so our number sits beside theirs.

WHY `blind` MUST SEE ALL CANDIDATES. The authors' ranker scores each element by
relevance to the task text. A "blind" model handed the top-50 has been told the
task through the shortlist: 90.8% of the time the answer is in there because
the task put it there. So `blind` and `told` both get every candidate in DOM
order, and only `told_ranked` uses the shortlist. That makes `told - blind`
exactly the value of the instruction, with nothing else moving.

FLOORS PRINTED BESIDE EVERY NUMBER. `first` (the earliest candidate in page
order, which reads nothing) is the floor for each condition on the same
decision points. A model number without it is a press release.

CONFIDENCE IS FREE. Each call also returns a 0-100 confidence. Accuracy bucketed
by confidence is the "when to speak" question from `intent.py`, at no extra
cost: a model that knows when it does not know is the one you can ship.

CONTAMINATION, NAMED. Mind2Web is from 2023 and frontier models have likely
seen it. The cross-domain split is the hardest available control, not a clean
one. Treat `told` as an upper bound and `blind` as the more trustworthy of the
two, since memorised trajectories help less when the task text is absent.

Usage:
    python model.py --dry-run                     print one prompt, call nothing
    python model.py --n 300                       blind + told, 300 points
    python model.py --n 300 --cond blind
    python model.py --n 0 --models a,b,c          everything, three models
    python model.py --report                      re-read cached results

Any OpenAI-compatible endpoint. Together AI by default:
    export TOGETHER_API_KEY=...
or  export OPENAI_BASE_URL=... OPENAI_API_KEY=...
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys
import time
import urllib.request

import screen

SPLIT = "data/test_domain-*.parquet"
OUT = "results"
DEFAULT_MODELS = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
OPS = ("CLICK", "TYPE", "SELECT")

_ACT = re.compile(r"^(?P<elem>.+?)\s*->\s*(?P<op>[A-Z]+)(?::\s*(?P<val>.*))?$")


# --- data --------------------------------------------------------------------

def load_steps(pattern=SPLIT):
    """One row per action step, with that step's candidates.

    Returns {annotation_id: {"task", "reprs", "steps": {k: (pos, neg, uid)}}}.
    A decision point after k actions is scored against step k, whose candidates
    are the page the person was looking at when they took action k.
    """
    import duckdb

    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet at {pattern}; run ./fetch.sh")
    rows = duckdb.connect().execute(
        f"""SELECT annotation_id, confirmed_task, action_reprs,
                   target_action_index, pos_candidates, neg_candidates, action_uid
            FROM read_parquet('{pattern}')""").fetchall()
    wf = {}
    for ann, task, reprs, idx, pos, neg, uid in rows:
        w = wf.setdefault(ann, {"task": task, "reprs": list(reprs), "steps": {}})
        try:
            k = int(idx)
        except (TypeError, ValueError):
            continue
        w["steps"][k] = (list(pos or []), list(neg or []), uid)
    return wf


def decisions(wf, min_len=3):
    """(ann, task, prefix, truth, pos, neg, key) for every k = 1 .. n-1.

    k=0 is excluded, as in intent.py: offering before the person has done
    anything is a prompt box, not proactivity.
    """
    out = []
    for ann, w in wf.items():
        reprs = w["reprs"]
        if len(reprs) < min_len:
            continue
        for k in range(1, len(reprs)):
            if k not in w["steps"]:
                continue
            pos, neg, uid = w["steps"][k]
            if not pos:
                continue
            out.append((ann, w["task"], reprs[:k], reprs[k], pos, neg, f"{ann}_{uid}"))
    return out


def sample(ds, n, seed=0):
    """A fixed-seed sample. Same n and seed, same points, every run."""
    if not n or n >= len(ds):
        return list(ds)
    return random.Random(seed).sample(ds, n)


# --- candidates --------------------------------------------------------------

def render(c):
    """One line per candidate: [id] <tag> label text. Same labels the rule
    policies see (screen.cand_text), so model and floor read the same page."""
    d = c if isinstance(c, dict) else json.loads(c)
    tag = d.get("tag", "")
    text = " ".join(screen.cand_text(c).split())[:120]
    return f"[{screen.cand_id(c)}] <{tag}> {text}".rstrip()


def dom_order(pos, neg):
    """Every candidate by node id. Task-blind, and the order the floor uses."""
    allc = [(c, True) for c in pos] + [(c, False) for c in neg]
    allc.sort(key=lambda t: screen._nid_key(screen.cand_id(t[0])))
    return [c for c, _ in allc], {screen.cand_id(c) for c, p in allc if p}


def ranked_order(task, pos, neg, key, k=50):
    """The authors' top-k. Requires their scores file; see screen.load_xenc."""
    prepped = screen.prepare([(task, pos, neg, key.split("_")[0], key.split("_", 1)[1])])
    if not prepped:
        return [], set()
    tt, cands, gold, key2 = prepped[0]
    top = screen.rank(tt, cands, k=k, key=key2)
    by_id = {screen.cand_id(c): c for c in list(pos) + list(neg)}
    return [by_id[c["id"]] for c in top if c["id"] in by_id], gold


# --- prompt ------------------------------------------------------------------

SYSTEM = (
    "You are watching a person use a website. You will see the actions they have "
    "taken so far and the interactive elements on the page they are looking at now. "
    "Predict the single next action they will take. Answer with JSON only."
)


def prompt(task, prefix, cands, told):
    lines = []
    if told:
        lines.append(f"Their goal: {task}\n")
    else:
        lines.append("You do not know their goal.\n")
    lines.append("Actions so far, in order:")
    for i, a in enumerate(prefix, 1):
        lines.append(f"  {i}. {a}")
    lines.append(f"\nElements on the current page ({len(cands)}):")
    for c in cands:
        lines.append("  " + render(c))
    lines.append(
        "\nPredict the next action. Reply with exactly one JSON object:\n"
        '{"id": "<element id from the list>", "op": "CLICK" | "TYPE" | "SELECT", '
        '"value": "<text to type or option to select, or empty>", '
        '"confidence": <0-100, how sure you are this is the next action>}')
    return "\n".join(lines)


# --- the call ----------------------------------------------------------------

def endpoint():
    base = os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY")
    if os.environ.get("TOGETHER_API_KEY") and not base:
        base, key = "https://api.together.xyz/v1", os.environ["TOGETHER_API_KEY"]
    if not (base and key):
        raise SystemExit("set TOGETHER_API_KEY, or OPENAI_BASE_URL and OPENAI_API_KEY")
    return base.rstrip("/"), key


def call(model, text, retries=4):
    base, key = endpoint()
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": 200,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": text}],
    }).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            return d["choices"][0]["message"]["content"], d.get("usage", {})
        except Exception as e:  # rate limits, transient 5xx
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


_JSON = re.compile(r"\{.*\}", re.S)


def parse_reply(s):
    """{"id","op","value","confidence"} or None. Generous about fences and
    prose around the object; strict that an id and an op exist."""
    m = _JSON.search(s or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(d, dict) or "id" not in d or "op" not in d:
        return None
    d["id"] = str(d["id"]).strip("[] ")
    d["op"] = str(d["op"]).upper().strip()
    d["value"] = str(d.get("value") or "").strip()
    try:
        d["confidence"] = max(0, min(100, int(float(d.get("confidence", 50)))))
    except (TypeError, ValueError):
        d["confidence"] = 50
    return d


# --- scoring -----------------------------------------------------------------

def truth_parts(truth):
    m = _ACT.match(truth or "")
    if not m:
        return None, "", ""
    return m.group("elem"), m.group("op"), (m.group("val") or "").strip()


def score_one(pred, gold_ids, truth):
    """(element_right, op_right, both). Element is right if the predicted id is
    a gold id. Operation is right if it matches the recorded op. Nothing looser:
    an assistant that reached the goal a different way is marked wrong here,
    exactly as intent.py does, so these are conservative."""
    _, t_op, _ = truth_parts(truth)
    if pred is None:
        return 0, 0, 0
    e = int(pred["id"] in gold_ids)
    o = int(pred["op"] == t_op)
    return e, o, int(e and o)


def floor_first(cands, gold_ids, truth):
    """The no-model floor on this exact point: earliest candidate in page order,
    always CLICK. Reads nothing."""
    if not cands:
        return 0, 0, 0
    first = screen.cand_id(cands[0])
    _, t_op, _ = truth_parts(truth)
    e = int(first in gold_ids)
    o = int(t_op == "CLICK")
    return e, o, int(e and o)


# --- run ---------------------------------------------------------------------

def out_path(model, cond):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    return os.path.join(OUT, f"{safe}__{cond}.jsonl")


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {json.loads(l)["key"] for l in fh if l.strip()}


def run(model, cond, points, xenc_loaded, verbose=True):
    os.makedirs(OUT, exist_ok=True)
    path = out_path(model, cond)
    seen = done_keys(path)
    todo = [p for p in points if p[6] not in seen]
    if verbose:
        print(f"{model}  {cond}: {len(seen)} cached, {len(todo)} to run", file=sys.stderr)
    with open(path, "a") as fh:
        for i, (ann, task, prefix, truth, pos, neg, key) in enumerate(todo, 1):
            if cond == "told_ranked":
                if not xenc_loaded:
                    raise SystemExit("told_ranked needs data/scores_all_data.pkl; run ./fetch.sh")
                cands, gold = ranked_order(task, pos, neg, key)
            else:
                cands, gold = dom_order(pos, neg)
            if not cands:
                continue
            text = prompt(task, prefix, cands, told=(cond != "blind"))
            try:
                reply, usage = call(model, text)
            except Exception as e:
                reply, usage = f"ERROR {type(e).__name__}: {e}", {}
            pred = parse_reply(reply)
            e, o, b = score_one(pred, gold, truth)
            fe, fo, fb = floor_first(cands, gold, truth)
            rec = {"key": key, "ann": ann, "k": len(prefix), "n_cands": len(cands),
                   "gold_in_cands": int(bool(gold & {screen.cand_id(c) for c in cands})),
                   "pred": pred, "truth": truth, "elem": e, "op": o, "both": b,
                   "floor_elem": fe, "floor_op": fo, "floor_both": fb,
                   "conf": (pred or {}).get("confidence"),
                   "tokens": usage.get("total_tokens"), "raw": reply[:300]}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if verbose and i % 25 == 0:
                print(f"  {i}/{len(todo)}", file=sys.stderr)
    return path


def report(models, conds):
    print(f"\n{'model':44} {'cond':12} {'n':>5} {'elem':>7} {'op':>7} {'both':>7} "
          f"{'floor':>7} {'ceil':>7}")
    print("-" * 104)
    for model in models:
        for cond in conds:
            path = out_path(model, cond)
            if not os.path.exists(path):
                continue
            rows = [json.loads(l) for l in open(path) if l.strip()]
            rows = [r for r in rows if r.get("pred") is not None or "ERROR" not in r.get("raw", "")]
            if not rows:
                continue
            n = len(rows)
            pct = lambda k: 100 * sum(r[k] for r in rows) / n
            print(f"{model[:44]:44} {cond:12} {n:5} {pct('elem'):6.1f}% {pct('op'):6.1f}% "
                  f"{pct('both'):6.1f}% {pct('floor_elem'):6.1f}% "
                  f"{100*sum(r['gold_in_cands'] for r in rows)/n:6.1f}%")
            # the "when to speak" cut: accuracy by confidence bucket
            buckets = collections.defaultdict(list)
            for r in rows:
                c = r.get("conf")
                if c is None:
                    continue
                buckets[min(c // 25, 3)].append(r["elem"])
            if buckets:
                parts = []
                for b in sorted(buckets):
                    v = buckets[b]
                    parts.append(f"conf {b*25:>2}-{b*25+24 if b < 3 else 100:<3} "
                                 f"n={len(v):<4} elem {100*sum(v)/len(v):5.1f}%")
                print("    " + " | ".join(parts))
    print("\nfloor = first candidate in page order, always CLICK, on the same points.")
    print("ceil  = share of points where the gold element was in the list at all.")
    print("Mind2Web is from 2023. Treat `told` as an upper bound; `blind` is the "
          "more trustworthy number, since memorised trajectories help less without "
          "the task text.")


# --- selftest ----------------------------------------------------------------

def selftest():
    """The controls. Each one is a way this file could ship a fake number."""
    fails = []

    def ck(label, got, want):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'}  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
        if not ok:
            fails.append(label)

    # Real candidates carry backend_node_id at the top level AND inside
    # attributes; cand_id reads the top level. A fixture that only puts it
    # inside attributes returns None ids and every downstream check passes
    # vacuously, which is how a test suite stops being one.
    C = lambda nid, tag="a", text="": json.dumps(
        {"tag": tag, "backend_node_id": str(nid),
         "attributes": json.dumps({"backend_node_id": str(nid), "text": text})})

    # THE LEAK. A blind prompt that contains the task is the ranker bug again,
    # in prose. Assert the task string is absent, and present when told.
    task = "Book a passport appointment for Ellen Walker in Aurora"
    cands = [C(5, "a", "Home"), C(9, "input", "search")]
    ck("blind prompt withholds the task", task in prompt(task, ["[a] x -> CLICK"], cands, told=False), False)
    ck("told prompt includes the task", task in prompt(task, ["[a] x -> CLICK"], cands, told=True), True)
    ck("blind prompt says the goal is unknown",
       "do not know their goal" in prompt(task, ["[a] x -> CLICK"], cands, told=False), True)

    # dom_order sorts by node id. gold must NOT be first just because it is pos,
    # which is the pos+neg bug this repo shipped twice.
    cands_dom, gold = dom_order(pos=[C(900, "a", "late gold")], neg=[C(5), C(100)])
    ck("gold is not first merely for being pos", screen.cand_id(cands_dom[0]), "5")
    ck("gold ids collected", gold, {"900"})

    # floor_first reads position, not the label. With gold late, floor misses.
    ck("floor misses when gold is late", floor_first(cands_dom, gold, "[a] late gold -> CLICK"), (0, 1, 0))
    early, g2 = dom_order(pos=[C(1, "a", "early")], neg=[C(50), C(60)])
    ck("floor hits when gold is first", floor_first(early, g2, "[a] early -> CLICK"), (1, 1, 1))
    ck("floor op is always CLICK", floor_first(early, g2, "[a] early -> TYPE: x")[1], 0)

    # parse_reply: fences, prose, junk
    ck("parses fenced json", parse_reply('```json\n{"id": "12", "op": "click", "confidence": 80}\n```')["op"], "CLICK")
    ck("parses json inside prose", parse_reply('Sure. {"id": "[12]", "op": "TYPE", "value": "hi"} done')["id"], "12")
    ck("rejects no json", parse_reply("I think element 12"), None)
    ck("rejects missing op", parse_reply('{"id": "12"}'), None)
    ck("clamps confidence", parse_reply('{"id":"1","op":"CLICK","confidence": 250}')["confidence"], 100)
    ck("defaults confidence", parse_reply('{"id":"1","op":"CLICK"}')["confidence"], 50)

    # score_one: element and op are independent; both needs both.
    gold = {"12"}
    ck("right id right op", score_one({"id": "12", "op": "CLICK"}, gold, "[a] x -> CLICK"), (1, 1, 1))
    ck("right id wrong op", score_one({"id": "12", "op": "TYPE"}, gold, "[a] x -> CLICK"), (1, 0, 0))
    ck("wrong id right op", score_one({"id": "13", "op": "CLICK"}, gold, "[a] x -> CLICK"), (0, 1, 0))
    ck("no prediction scores zero", score_one(None, gold, "[a] x -> CLICK"), (0, 0, 0))

    # sample is fixed-seed
    ds = list(range(1000))
    ck("sample is deterministic", sample(ds, 10, 0), sample(ds, 10, 0))
    ck("sample returns all when n=0", len(sample(ds, 0)), 1000)

    print(f"\n{len(fails)} failure(s)" if fails else "\nall controls hold")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="decision points (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default=DEFAULT_MODELS, help="comma-separated")
    ap.add_argument("--cond", default="blind,told",
                    help="comma list of blind, told, told_ranked")
    ap.add_argument("--dry-run", action="store_true", help="print one prompt, call nothing")
    ap.add_argument("--report", action="store_true", help="only print cached results")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    conds = [c.strip() for c in a.cond.split(",") if c.strip()]
    for c in conds:
        if c not in ("blind", "told", "told_ranked"):
            raise SystemExit(f"unknown condition {c!r}")

    if a.selftest:
        raise SystemExit(selftest())
    if a.report:
        report(models, conds); return

    wf = load_steps()
    ds = decisions(wf)
    points = sample(ds, a.n, a.seed)
    print(f"{len(wf)} workflows, {len(ds)} decision points, {len(points)} sampled "
          f"(seed {a.seed})", file=sys.stderr)

    xenc = False
    if "told_ranked" in conds:
        p = "data/scores_all_data.pkl"
        if os.path.exists(p):
            screen.load_xenc(p); xenc = True

    if a.dry_run:
        ann, task, prefix, truth, pos, neg, key = points[0]
        cands, gold = dom_order(pos, neg)
        text = prompt(task, prefix, cands, told=False)
        print(text)
        print(f"\n--- {len(cands)} candidates, ~{len(text)//4} tokens, truth: {truth}")
        print(f"--- gold ids: {sorted(gold)}")
        return

    for model in models:
        for cond in conds:
            run(model, cond, points, xenc)
    report(models, conds)


if __name__ == "__main__":
    main()
