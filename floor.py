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

    ./fetch.sh                      # the split, verified shard by shard
    python floor.py                 # every policy, over every shard in data/
    python floor.py --limit 400     # a fast pass while developing
    python floor.py --selftest      # the scorer, on cases with known answers
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Every shard of the split by default. One file is enough to develop against;
# a number worth quoting is over all of them.
SHARD = os.path.join(HERE, "data", "*.parquet")

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
def pick_first(tt, cands):
    return 0 if cands else None


def pick_overlap(tt, cands):
    if not tt:
        return None
    best, bi = 0, None
    for i, c in enumerate(cands):
        n = len(tt & c["toks"])
        if n > best:
            best, bi = n, i
    return bi


def pick_centre(tt, cands, vw=1280.0, vh=720.0):
    best, bi = None, None
    for i, c in enumerate(cands):
        b = c["box"]
        if b is None:
            continue
        d = (b[0] - vw / 2) ** 2 + (b[1] - vh / 2) ** 2
        if best is None or d < best:
            best, bi = d, i
    return bi


POLICIES = {"first": pick_first, "overlap": pick_overlap, "centre": pick_centre}


def prepare(rows):
    """Parse each candidate exactly once.

    Every policy needs the same three things from a candidate: its id, its
    label text, and its box. The first version derived them inside the scoring
    loop, so a run over the full split re-parsed the same JSON eighteen times,
    once per policy-subset-order combination, and did not finish. Parsing here
    turns that into one pass.

    Each row becomes (task_tokens, [Cand], gold_ids).
    """
    out = []
    for task, pos, neg in rows:
        if not pos:
            continue
        gold, cands = set(), []
        for c, is_pos in [(c, True) for c in pos] + [(c, False) for c in neg]:
            nid = cand_id(c)
            cands.append({"id": nid, "toks": toks(cand_text(c)), "box": cand_box(c)})
            if is_pos:
                gold.add(nid)
        if cands:
            out.append((toks(task), cands, gold))
    return out


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


def rank(tt, cands, k=50):
    """A stand-in for the ranker a real Mind2Web setup runs first.

    The published pipeline scores every element with a cross-encoder and hands
    the agent a top-k shortlist, so a floor measured over all 500 candidates
    describes a task nobody actually evaluates. This approximates that stage
    with text similarity to the task, which is the signal the cross-encoder is
    trained to approximate, and keeps the top k.

    Two things fall out and both matter:

      recall@k   how often the correct element survives ranking at all. This is
                 a ceiling on every downstream score, and it belongs next to any
                 number measured inside the shortlist.
      the floor  what a no-perception policy scores WITHIN the shortlist, which
                 is the number a real system's floor actually is.

    THE CIRCULARITY, NAMED. This ranker scores on text, so measuring the `overlap`
    policy inside its own shortlist would be marking its own homework. `first`
    and `centre` use position and geometry and are unaffected, so those are the
    ones reported in the ranked condition.
    """
    scored = sorted(cands, key=lambda c: -len(tt & c["toks"]))
    return scored[:k]


def _order_prepared(cands, how):
    # `rank` leaves the ranker's own ordering alone, so `first` becomes "take
    # the ranker's top choice". That is the floor a real system carries, and it
    # is a different question from document position.
    if how == "rank":
        return cands
    if how == "shuffle":
        import random
        c = list(cands)
        random.Random(0).shuffle(c)
        return c

    def key(c):
        try:
            return (0, int(c["id"]))
        except (TypeError, ValueError):
            return (1, 0)
    return sorted(cands, key=key)


def score(rows, policy, how="dom", k=0):
    """(hits, n) over PREPARED rows. A step counts only if the pick is a positive.

    `k` runs the ranker first and scores inside the top-k shortlist, which is
    the setting a real system evaluates in. Steps whose gold element the ranker
    drops still count as misses, because that is what they are: a shortlist that
    loses the answer makes the step unpassable for everything downstream.
    """
    hits = n = 0
    for tt, cands, gold in rows:
        n += 1
        c = rank(tt, cands, k) if k else cands
        c = _order_prepared(c, how)
        if not c:
            continue
        i = policy(tt, c)
        if i is not None and c[i]["id"] in gold:
            hits += 1
    return hits, n


def recall_at_k(rows, k=50):
    """How often the correct element survives ranking. A ceiling on everything
    measured inside the shortlist, so it is reported beside those numbers."""
    hits = 0
    for tt, cands, gold in rows:
        if any(c["id"] in gold for c in rank(tt, cands, k)):
            hits += 1
    return hits, len(rows)


def score_raw(rows, policy, how="dom"):
    """Scoring straight off unparsed rows. Used only by the selftest, where the
    fixtures are three candidates rather than two million."""
    return score(prepare(rows), policy, how)


def load(path, limit=None):
    """Rows from one parquet file or a glob over the whole split.

    The screenshot column is never selected. It is most of the 3.6 GB and no
    policy here is allowed to look at it, so reading it would only make the run
    slow and the claim weaker.
    """
    import duckdb

    con = duckdb.connect()
    lim = f" LIMIT {int(limit)}" if limit else ""
    q = (f"SELECT confirmed_task, pos_candidates, neg_candidates "
         f"FROM read_parquet('{path}'){lim}")
    return con.execute(q).fetchall()


def has_text(pos):
    """Does the correct element carry any text a matcher could use?"""
    return bool(toks(" ".join(cand_text(c) for c in pos)))


def _prep(cands):
    """Candidates in the shape the policies now take."""
    return [{"id": cand_id(c), "toks": toks(cand_text(c)), "box": cand_box(c)}
            for c in cands]


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
       pick_overlap(toks("book a passport appointment"), _prep([b, a])), 1)
    # A policy that guesses when nothing matches inflates its own floor.
    ck("overlap abstains when nothing overlaps",
       pick_overlap(toks("book a passport appointment"), _prep([b])), None)

    far = C("3", bounding_box_rect="0,0,10,10")
    near = C("4", bounding_box_rect="620,340,40,40")
    ck("centre picks the box nearest the viewport middle",
       pick_centre(toks("t"), _prep([far, near])), 1)
    ck("centre ignores candidates with no geometry",
       pick_centre(toks("t"), _prep([C("5"), near])), 1)
    ck("centre abstains when nothing has geometry",
       pick_centre(toks("t"), _prep([C("6"), C("7")])), None)

    # Scoring: a hit is the POSITIVE's id, not its position.
    rows = [("book a passport appointment", [a], [b])]
    ck("a correct pick scores", score_raw(rows, pick_overlap), (1, 1))
    ck("an abstention is a miss, never a skip",
       score_raw([("zzz", [C("8", **{"class": "nothing"})], [])], pick_overlap), (0, 1))

    # RED: the harness must not hand the answer to a position policy. The first
    # version returned pos + neg, so `first` scored 100% on every benchmark it
    # was ever pointed at.
    lo, hi = C("10"), C("900")
    ck("candidates are ordered by node id, not positives-first",
       [cand_id(c) for c in order([hi], [lo])], ["10", "900"])
    ck("...so `first` misses when the positive is late in the DOM",
       score_raw([("t", [hi], [lo])], pick_first), (0, 1))
    ck("...and hits when it is early", score_raw([("t", [lo], [hi])], pick_first), (1, 1))
    ck("shuffle is deterministic, so a run is reproducible",
       [cand_id(c) for c in order([hi], [lo], "shuffle")]
       == [cand_id(c) for c in order([hi], [lo], "shuffle")], True)

    # --- the ranking stage ------------------------------------------------
    hit = C("20", **{"class": "passport appointment"})
    miss1, miss2 = C("21", **{"class": "checkout"}), C("22", **{"class": "footer"})
    tt = toks("book a passport appointment")
    ck("the ranker keeps the candidate that matches the task",
       [c["id"] for c in rank(tt, _prep([miss1, hit, miss2]), 1)], ["20"])
    ck("...and a shortlist of k returns at most k",
       len(rank(tt, _prep([miss1, hit, miss2]), 2)), 2)
    ck("recall@k sees the gold when the ranker keeps it",
       recall_at_k([(tt, _prep([miss1, hit]), {"20"})], 1), (1, 1))
    # RED: a shortlist that drops the answer must count as a miss, not a skip.
    ck("...and misses when the ranker drops it",
       recall_at_k([(tt, _prep([hit, miss1]), {"99"})], 1), (0, 1))
    ck("a step whose gold the ranker drops is a miss for every policy",
       score([(tt, _prep([hit, miss1]), {"99"})], pick_first, "rank", 1), (0, 1))
    ck("rank order leaves the ranker's own order alone",
       [c["id"] for c in _order_prepared(_prep([hit, miss1]), "rank")], ["20", "21"])

    ck("no-text control: an icon-only element has nothing to match",
       has_text([C("9", tag="div")]), False)
    ck("...and a labelled one does", has_text([a]), True)

    print("\n  " + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=SHARD)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--topk", type=int, default=50,
                    help="shortlist size for the ranked condition")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    import glob
    files = sorted(glob.glob(a.shard)) if "*" in a.shard else [a.shard]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print(f"  no parquet at {a.shard}")
        return 2
    rows = load(a.shard, a.limit or None)
    print(f"  {len(rows):,} action steps over {len(files)} shard(s)\n")

    # Split BEFORE preparing, because `has_text` reads the raw positives, then
    # parse each subset once. Preparing the whole set twice would undo the point.
    lab_raw = [r for r in rows if r[1] and has_text(r[1])]
    unl_raw = [r for r in rows if r[1] and not has_text(r[1])]
    rows, lab, unl = prepare(rows), prepare(lab_raw), prepare(unl_raw)
    f = lambda h, n: f"{h/n:6.1%} ({h}/{n})" if n else "      n/a"

    print("  UNRANKED: every element on the page\n")
    for how in ("dom", "shuffle"):
        print(f"  candidates in {how} order")
        print(f"  {'policy':10} {'all steps':>18}   {'text-labelled':>18}   "
              f"{'no text (control)':>20}")
        for name, fn in POLICIES.items():
            print(f"  {name:10} {f(*score(rows, fn, how)):>18}   "
                  f"{f(*score(lab, fn, how)):>18}   {f(*score(unl, fn, how)):>20}")
        print()

    # The setting a real system evaluates in: rank first, then act on a
    # shortlist. `overlap` is omitted here because the ranker scores on text and
    # would be grading its own shortlist.
    k = a.topk
    rh, rn = recall_at_k(rows, k)
    print(f"  RANKED: the ranker's top {k}, which is where a real system acts")
    print(f"  recall@{k}: {rh/rn:.1%} ({rh}/{rn})  <- ceiling on everything below\n")
    print(f"  {'policy':10} {'in rank order':>18}   {'in dom order':>18}")
    for name in ("first", "centre"):
        fn = POLICIES[name]
        print(f"  {name:10} {f(*score(rows, fn, 'rank', k)):>18}   "
              f"{f(*score(rows, fn, 'dom', k)):>18}")

    print(f"\n  text-labelled {len(lab):,} steps, no-text control {len(unl):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
