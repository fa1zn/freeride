"""What does a policy that never looks at the screen score on a computer-use benchmark?

Multimodal-Mind2Web gives a model a task, a page, and a candidate set, and asks
which element to act on. It ships screenshots, so it reads as a perception
benchmark. This asks how much of it is solvable with the screenshot deleted.

Three policies, none of which perceives anything and none of which calls a model:

  first      pick the first candidate in DOM order
  overlap    pick the candidate whose text overlaps the task description most
  centre     pick the candidate closest to the centre of the viewport

A benchmark whose floor is near zero is measuring what it claims. A benchmark
where one of these scores well is partly measuring something else, and every
published number on it carries that floor underneath.

The control is the no-text subset: steps whose correct element carries no text
label at all, only an icon or a bare div. `overlap` has nothing to match on
there, so its score should collapse. If it does not, the mechanism is something
other than text matching and this file is wrong about why.

    python floor.py                 # every policy, on the shard in data/
    python floor.py --selftest      # the scorer, on cases with known answers
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD = os.path.join(HERE, "data", "shard0.parquet")

WORD = re.compile(r"[a-z0-9]+")
# Words that appear in nearly every task and match nearly every element, so
# leaving them in makes `overlap` look better than it is.
STOP = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "with",
    "my", "me", "i", "is", "are", "be", "it", "this", "that", "from", "by",
    "find", "show", "get", "go", "open", "click", "select", "check", "search",
    "please", "want", "need", "would", "like", "can", "you", "your",
}


def toks(s):
    return {w for w in WORD.findall((s or "").lower()) if w not in STOP and len(w) > 2}


def cand_text(c):
    """Every LABEL a candidate carries that a text policy could match on.

    Deliberately generous about which attributes count. If the floor turns out
    high, it must not be because the parser was clever; a generous reading makes
    the number a ceiling on what text matching buys, which is the honest
    direction for this measurement.

    THE TAG NAME IS EXCLUDED, and the selftest is what caught that. With `tag`
    in the string, an icon-only `<div>` carries the token "div", so the no-text
    control was not a control at all: every element had text. It would also have
    inflated `overlap` on any task containing a word like "button" or "link".
    """
    d = c if isinstance(c, dict) else json.loads(c)
    attrs = d.get("attributes")
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except (ValueError, TypeError):
            attrs = {}
    attrs = attrs or {}
    parts = []
    for k in ("role", "class", "id", "placeholder", "aria_label", "alt",
              "title", "name", "value", "input_value", "text"):
        v = attrs.get(k)
        if isinstance(v, str):
            parts.append(v)
    return " ".join(parts)


def cand_box(c):
    """(cx, cy) of a candidate, or None when it carries no geometry."""
    d = c if isinstance(c, dict) else json.loads(c)
    attrs = d.get("attributes")
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except (ValueError, TypeError):
            return None
    rect = (attrs or {}).get("bounding_box_rect")
    if not isinstance(rect, str):
        return None
    try:
        x, y, w, h = (float(v) for v in rect.split(","))
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return x + w / 2, y + h / 2


def cand_id(c):
    d = c if isinstance(c, dict) else json.loads(c)
    return d.get("backend_node_id")


# --------------------------------------------------------------------------
# The policies. None of them sees a screenshot; none of them calls a model.
# --------------------------------------------------------------------------
def pick_first(task, cands):
    return 0 if cands else None


def pick_overlap(task, cands):
    t = toks(task)
    if not t:
        return None
    best, bi = -1, None
    for i, c in enumerate(cands):
        n = len(t & toks(cand_text(c)))
        if n > best:
            best, bi = n, i
    return bi if best > 0 else None


def pick_centre(task, cands, vw=1280.0, vh=720.0):
    best, bi = None, None
    for i, c in enumerate(cands):
        b = cand_box(c)
        if b is None:
            continue
        d = ((b[0] - vw / 2) ** 2 + (b[1] - vh / 2) ** 2) ** 0.5
        if best is None or d < best:
            best, bi = d, i
    return bi


POLICIES = {"first": pick_first, "overlap": pick_overlap, "centre": pick_centre}


def order(pos, neg, how="dom"):
    """Candidates as a policy would see them.

    THE FIRST VERSION OF THIS RETURNED `pos + neg` AND THE `first` POLICY SCORED
    100%. That was the harness handing the answer over at index 0, not a finding
    about the benchmark, and it is the single easiest way to publish a fake
    floor. Candidate order has to come from something the label does not touch.

    dom      by backend_node_id, which approximates document order. This is a
             real prior worth measuring: correct elements may genuinely cluster
             early in a page.
    shuffle  fixed-seed shuffle. Position carries nothing, so any policy that
             scores here is using content rather than order.
    """
    cands = list(pos) + list(neg)
    if how == "shuffle":
        import random
        random.Random(0).shuffle(cands)
        return cands

    def key(c):
        nid = cand_id(c)
        try:
            return (0, int(nid))
        except (TypeError, ValueError):
            return (1, 0)
    return sorted(cands, key=key)


def score(rows, policy, how="dom"):
    """(hits, n). A step counts only if the chosen candidate is the positive."""
    hits = n = 0
    for task, pos, neg in rows:
        if not pos:
            continue
        cands = order(pos, neg, how)
        if not cands:
            continue
        gold = {cand_id(c) for c in pos}
        n += 1
        i = policy(task, cands)
        if i is not None and cand_id(cands[i]) in gold:
            hits += 1
    return hits, n


def load(path, limit=None):
    import duckdb

    con = duckdb.connect()
    lim = f" LIMIT {int(limit)}" if limit else ""
    q = (f"SELECT confirmed_task, pos_candidates, neg_candidates "
         f"FROM read_parquet('{path}'){lim}")
    return con.execute(q).fetchall()


def has_text(pos):
    """Does the correct element carry any text a matcher could use?"""
    return bool(toks(" ".join(cand_text(c) for c in pos)))


def selftest() -> int:
    ok = True

    def ck(label, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {label:58s} {got!r}")

    def C(nid, tag="a", **attrs):
        return {"backend_node_id": nid, "tag": tag,
                "attributes": json.dumps(attrs)}

    ck("stopwords are dropped, so 'find the' matches nothing",
       toks("find the a of"), set())
    ck("a real word survives", toks("passport appointment"), {"passport", "appointment"})

    a = C("1", role="menuitem", **{"class": "passport-link"})
    b = C("2", role="button", **{"class": "checkout"})
    ck("overlap picks the candidate sharing a task word",
       pick_overlap("book a passport appointment", [b, a]), 1)
    # A policy that guesses when nothing matches inflates its own floor.
    ck("overlap abstains when nothing overlaps",
       pick_overlap("book a passport appointment", [b]), None)

    far = C("3", bounding_box_rect="0,0,10,10")
    near = C("4", bounding_box_rect="620,340,40,40")
    ck("centre picks the box nearest the viewport middle",
       pick_centre("t", [far, near]), 1)
    ck("centre ignores candidates with no geometry",
       pick_centre("t", [C("5"), near]), 1)
    ck("centre abstains when nothing has geometry",
       pick_centre("t", [C("6"), C("7")]), None)

    # Scoring: a hit is the POSITIVE's id, not its position.
    rows = [("book a passport appointment", [a], [b])]
    ck("a correct pick scores", score(rows, pick_overlap), (1, 1))
    ck("an abstention is a miss, never a skip",
       score([("zzz", [C("8", **{"class": "nothing"})], [])], pick_overlap), (0, 1))

    # RED: the harness must not hand the answer to a position policy. The first
    # version returned pos + neg, so `first` scored 100% on every benchmark it
    # was ever pointed at.
    lo, hi = C("10"), C("900")
    ck("candidates are ordered by node id, not positives-first",
       [cand_id(c) for c in order([hi], [lo])], ["10", "900"])
    ck("...so `first` misses when the positive is late in the DOM",
       score([("t", [hi], [lo])], pick_first), (0, 1))
    ck("...and hits when it is early", score([("t", [lo], [hi])], pick_first), (1, 1))
    ck("shuffle is deterministic, so a run is reproducible",
       [cand_id(c) for c in order([hi], [lo], "shuffle")]
       == [cand_id(c) for c in order([hi], [lo], "shuffle")], True)

    ck("no-text control: an icon-only element has nothing to match",
       has_text([C("9", tag="div")]), False)
    ck("...and a labelled one does", has_text([a]), True)

    print("\n  " + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=SHARD)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    if not os.path.exists(a.shard):
        print(f"  no shard at {a.shard}")
        return 2
    rows = load(a.shard, a.limit or None)
    print(f"  {len(rows):,} action steps\n")

    lab = [r for r in rows if has_text(r[1])]
    unl = [r for r in rows if not has_text(r[1])]
    f = lambda h, n: f"{h/n:6.1%} ({h}/{n})" if n else "      n/a"

    for how in ("dom", "shuffle"):
        print(f"  candidates in {how} order")
        print(f"  {'policy':10} {'all steps':>18}   {'text-labelled':>18}   "
              f"{'no text (control)':>20}")
        for name, fn in POLICIES.items():
            print(f"  {name:10} {f(*score(rows, fn, how)):>18}   "
                  f"{f(*score(lab, fn, how)):>18}   {f(*score(unl, fn, how)):>20}")
        print()

    print(f"  text-labelled {len(lab):,} steps, no-text control {len(unl):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
