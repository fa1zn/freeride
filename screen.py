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
    python screen.py                 # every policy, over every shard in data/
    python screen.py --limit 400     # a fast pass while developing
    python screen.py --selftest      # the scorer, on cases with known answers
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


def _nid_key(nid):
    """Node id as a sort key. Ids that will not parse sort last, together."""
    try:
        return (0, int(nid))
    except (TypeError, ValueError):
        return (1, 0)


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
    for row in rows:
        task, pos, neg = row[0], row[1], row[2]
        key = f"{row[3]}_{row[4]}" if len(row) >= 5 else ""
        if not pos:
            continue
        gold, cands = set(), []
        # ORDER BY NODE ID HERE, NOT LATER. This list is what `rank` sorts, and
        # Python's sort is stable, so whatever sits at index 0 wins every tie on
        # ranker score. Built as `pos + neg` that is the correct element, which
        # hands the answer to the shortlist exactly the way `order` warns about.
        # The downstream `_order_prepared(c, "dom")` re-sort cannot undo it: by
        # then the biased tie-break has already decided who made the shortlist.
        for c, is_pos in sorted(
                [(c, True) for c in pos] + [(c, False) for c in neg],
                key=lambda t: _nid_key(cand_id(t[0]))):
            nid = cand_id(c)
            cands.append({"id": nid, "toks": toks(cand_text(c)), "box": cand_box(c)})
            if is_pos:
                gold.add(nid)
        if cands:
            out.append((toks(task), cands, gold, key))
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


# The published cross-encoder's scores, when they have been loaded. Empty means
# the text stand-in is in use, and every table says which one produced it.
_XENC = {}


def load_xenc(path):
    """The Mind2Web authors' own candidate-generation scores.

    `scores_all_data.pkl` on huggingface.co/datasets/osunlp/Mind2Web is the
    output of osunlp/MindAct_CandidateGeneration_deberta-v3-base over the whole
    dataset, and it is what their action-prediction stage consumes. Using it
    beats re-running the checkpoint: it is the exact ranking the published
    numbers rest on, with no chance of drifting on input format, pair order or
    library version.

    Shape is {f"{annotation_id}_{action_uid}": {backend_node_id: score}}.
    Reproduces the `ranks` the same file ships on 99.93% of steps, the remainder
    being ties on identical scores.
    """
    import pickle

    global _XENC
    with open(path, "rb") as fh:
        _XENC = pickle.load(fh)["scores"]
    return len(_XENC)


def _ranked(tt, cands, key=""):
    """Candidates best-first, by whichever ranker is loaded.

    TIES BREAK BY NODE ID, NEVER BY POSITION IN THE INPUT LIST. Building the
    input as `pos + neg` and letting a stable sort keep it is how this repo
    shipped a fake result once already, and a tie-break is exactly where that
    hides.
    """
    if _XENC and key in _XENC:
        sc = _XENC[key]
        return sorted(cands, key=lambda c: (-sc.get(c["id"], -1e9), _nid_key(c["id"])))
    return sorted(cands, key=lambda c: (-len(tt & c["toks"]), _nid_key(c["id"])))


def rank(tt, cands, k=50, key=""):
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
    return _ranked(tt, cands, key)[:k]


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
    for tt, cands, gold, key in rows:
        n += 1
        c = rank(tt, cands, k, key) if k else cands
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
    for tt, cands, gold, key in rows:
        if any(c["id"] in gold for c in rank(tt, cands, k, key)):
            hits += 1
    return hits, len(rows)



def sweep(rows, ks):
    """The floor as a function of shortlist size, with recall@k beside it.

    The headline result rests on two points: 21.4% over the whole page and 31.4%
    inside the top 50. Two points show a difference but not a mechanism. If the
    claim is that a smaller shortlist concentrates a positional prior rather than
    removing it, the floor should rise as k falls, peak, and then fall again once
    the shortlist gets small enough to start dropping the answer. That shape is
    the claim, and this measures it.

    The ranked order is computed ONCE per step and then sliced. The top 30 is a
    prefix of the top 50, so re-ranking per k would cost k times as much and
    return the same rows.

    `recall@k` falls with k and caps every policy below it, so it is reported in
    the same table rather than in a footnote. A floor that rises while recall
    collapses is not a stronger prior, it is a shortlist that has thrown the
    answer away, and the two are only distinguishable side by side.
    """
    ks = sorted({int(k) for k in ks}, reverse=True)
    acc = {k: {"recall": 0, "dom": 0, "rank": 0, "centre": 0} for k in ks}
    n = 0
    for tt, cands, gold, key in rows:
        n += 1
        ranked = _ranked(tt, cands, key)
        for k in ks:
            sl = ranked[:k]
            if not sl:
                continue
            a = acc[k]
            if any(c["id"] in gold for c in sl):
                a["recall"] += 1
            # `first` in document order: the prior, measured inside the shortlist.
            dom = _order_prepared(sl, "dom")
            if dom[0]["id"] in gold:
                a["dom"] += 1
            # `first` in rank order: taking the ranker's own top choice, which is
            # the floor for any system built on a ranker at all.
            if sl[0]["id"] in gold:
                a["rank"] += 1
            i = pick_centre(tt, dom)
            if i is not None and dom[i]["id"] in gold:
                a["centre"] += 1
    return n, acc


def score_raw(rows, policy, how="dom"):
    """Scoring straight off unparsed rows. Used only by the selftest, where the
    fixtures are three candidates rather than two million."""
    return score(prepare(rows), policy, how)


# Element accuracy reported for the Cross-Domain split, which is this split.
# Table 2 of arXiv:2306.06070v3 (Deng et al., Mind2Web, NeurIPS 2023). Quoted
# rather than reproduced, and every row carries the shortlist size it was run
# at, because comparing a score to a floor measured at a different k is the
# easiest way to get this wrong.
PUBLISHED = [
    # (system,               Ele. Acc, top-k, note)
    ("MindAct w/ Flan-T5XL",     42.1, 50, ""),
    ("MindAct w/ Flan-T5L",      39.7, 50, ""),
    ("MindAct w/ Flan-T5B",      33.9, 50, ""),
    ("Classification (DeBERTa)", 24.5, 50, ""),
    ("MindAct w/ GPT-3.5",       21.6, 50, ""),
    ("MindAct w/ GPT-4",         37.1, 10, "50 tasks only, top-10"),
    ("Generation (Flan-T5B)",    14.2, 50, ""),
]


def macro(rows, k):
    """Floor and ceiling averaged per task, then across tasks.

    The paper macro-averages its step-wise metrics across tasks. A floor
    micro-averaged over steps is a different denominator, and quoting one
    against the other is comparing two things that share a name. This returns
    the paper's shape so the comparison is legitimate; `--sweep` keeps
    reporting micro, and both are printed side by side.
    """
    import collections
    import statistics

    hit = collections.defaultdict(list)
    rec = collections.defaultdict(list)
    for tt, cands, gold, key in rows:
        sl = _ranked(tt, cands, key)[:k]
        if not sl:
            continue
        task = key.split("_")[0] if key else ""
        dom = sorted(sl, key=lambda c: _nid_key(c["id"]))
        hit[task].append(int(dom[0]["id"] in gold))
        rec[task].append(int(any(c["id"] in gold for c in sl)))
    f = statistics.mean(sum(v) / len(v) for v in hit.values())
    c = statistics.mean(sum(v) / len(v) for v in rec.values())
    return f * 100, c * 100, len(hit)


def compare(rows):
    """Published scores against the floor and ceiling of the setting they ran in.

    This is the whole argument of the repo made concrete. A score is not a
    capability number on its own; it is a position between what a policy with no
    perception already gets and what the retrieval stage leaves reachable.

    The normalised column is (score - floor) / (ceiling - floor): the share of
    the actually-available headroom a system captured. Negative means it did
    worse than not looking.
    """
    ks = sorted({k for _, _, k, _ in PUBLISHED})
    fc = {k: macro(rows, k) for k in ks}
    ntask = fc[ks[0]][2]
    print(f"  PUBLISHED SCORES AGAINST THEIR FLOOR, macro-averaged over "
          f"{ntask} tasks as the paper does\n")
    for k in ks:
        f, c, _ = fc[k]
        print(f"  top-{k}: floor {f:.1f}  ceiling {c:.1f}  "
              f"headroom {c - f:.1f} points")
    print()
    print(f"  {'system':26} {'Ele.Acc':>8} {'floor':>7} {'vs floor':>9} "
          f"{'normalised':>11}")
    for name, acc, k, note in sorted(PUBLISHED, key=lambda r: -r[1]):
        f, c, _ = fc[k]
        norm = (acc - f) / (c - f) * 100
        flag = "  " if acc > f else " <"
        print(f"  {name:26} {acc:>8.1f} {f:>7.1f} {acc - f:>+9.1f} "
              f"{norm:>10.1f}%{flag}{note}")
    below = sum(1 for _, acc, k, _ in PUBLISHED if acc <= fc[k][0])
    print(f"\n  {below} of {len(PUBLISHED)} reported systems score at or below "
          f"a policy that never looks at the page.")
    print("  Ele. Acc from Table 2, arXiv:2306.06070v3. Floor and ceiling "
          "measured here.")


def load(path, limit=None):
    """Rows from one parquet file or a glob over the whole split.

    The screenshot column is never selected. It is most of the 3.6 GB and no
    policy here is allowed to look at it, so reading it would only make the run
    slow and the claim weaker.
    """
    import duckdb

    con = duckdb.connect()
    lim = f" LIMIT {int(limit)}" if limit else ""
    q = (f"SELECT confirmed_task, pos_candidates, neg_candidates, "
         f"annotation_id, action_uid "
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
       recall_at_k([(tt, _prep([miss1, hit]), {"20"}, "")], 1), (1, 1))
    # RED: a shortlist that drops the answer must count as a miss, not a skip.
    ck("...and misses when the ranker drops it",
       recall_at_k([(tt, _prep([hit, miss1]), {"99"}, "")], 1), (0, 1))
    ck("a step whose gold the ranker drops is a miss for every policy",
       score([(tt, _prep([hit, miss1]), {"99"}, "")], pick_first, "rank", 1), (0, 1))
    ck("rank order leaves the ranker's own order alone",
       [c["id"] for c in _order_prepared(_prep([hit, miss1]), "rank")], ["20", "21"])

    ck("no-text control: an icon-only element has nothing to match",
       has_text([C("9", tag="div")]), False)
    ck("...and a labelled one does", has_text([a]), True)


    # --- the shortlist sweep -------------------------------------------------
    # A sweep is easy to get backwards, because every column moves at once and a
    # rising floor next to a collapsing recall looks like a stronger result when
    # it is a shortlist that has thrown the answer away. These pin the shape.
    gold = C("40", **{"class": "passport"})
    filler = [C(str(50 + i), **{"class": f"junk{i}"}) for i in range(6)]
    swrows = prepare([("book a passport appointment", [gold], filler)])

    n, acc = sweep(swrows, [1, 2, 7])
    ck("sweep sees every step once", n, 1)
    # Bigger shortlist can only keep more, never less.
    ck("recall@k is monotone in k",
       [acc[k]["recall"] for k in (1, 2, 7)], sorted(acc[k]["recall"] for k in (1, 2, 7)))
    # One candidate means document order and rank order are the same list, so a
    # disagreement here means the two columns are not measuring what they say.
    ck("at k=1 dom order and rank order agree",
       acc[1]["dom"] == acc[1]["rank"], True)
    # The ranker puts the only task-matching element first, so k=1 keeps it.
    ck("the ranker keeps the gold at k=1", acc[1]["recall"], 1)
    # At k >= the candidate count the shortlist is the whole page, so the sweep
    # must reproduce the unranked number rather than a ranked one.
    ck("at k >= all candidates the sweep equals the unranked floor",
       acc[7]["dom"], score(swrows, pick_first, "dom")[0])

    # A shortlist that drops the answer is a miss in EVERY column, including the
    # positional ones. Without this a small k looks like a high floor.
    late = C("999", **{"class": "passport"})
    lead = [C(str(i), **{"class": "junk"}) for i in range(1, 6)]
    droprows = prepare([("zzz nothing matches", [late], lead)])
    _, dacc = sweep(droprows, [1])
    ck("a shortlist that loses the gold scores zero everywhere",
       (dacc[1]["recall"], dacc[1]["dom"], dacc[1]["rank"], dacc[1]["centre"]), (0, 0, 0, 0))


    # --- the published ranker ------------------------------------------------
    # Loading real scores must actually change the ordering. A path that silently
    # falls through to the text stand-in would report the stand-in's numbers
    # under the real ranker's name, which is the worst failure available here.
    # NOT `import screen`: run as a script this module is __main__, so importing
    # it by name binds a second copy and the assignment below lands on the one
    # `_ranked` is not reading. globals() is the module actually executing.
    _g = globals()
    hi = C("700", **{"class": "zzz"})       # no task words at all
    lo = C("100", **{"class": "passport"})  # the text ranker's favourite
    cs = _prep([hi, lo])
    tt2 = toks("book a passport appointment")
    ck("text stand-in ranks on words", [c["id"] for c in _ranked(tt2, cs)], ["100", "700"])
    _g["_XENC"] = {"K": {"700": 0.9, "100": 0.1}}
    ck("loaded scores override the stand-in",
       [c["id"] for c in _ranked(tt2, cs, "K")], ["700", "100"])
    ck("a step absent from the score file falls back to text",
       [c["id"] for c in _ranked(tt2, cs, "ABSENT")], ["100", "700"])
    # Equal scores must not be resolved by who was passed in first.
    _g["_XENC"] = {"K": {"700": 0.5, "100": 0.5}}
    ck("ties break by node id, not input order",
       [c["id"] for c in _ranked(tt2, _prep([hi, lo]), "K")], ["100", "700"])
    _g["_XENC"] = {}


    # --- macro averaging -----------------------------------------------------
    # The published numbers are macro-averaged across tasks and this repo
    # micro-averages across steps. Quoting one against the other compares two
    # different denominators that share a name, so the two must not silently
    # coincide in the tests.
    a1 = C("10", **{"class": "passport"})
    b1 = C("900", **{"class": "junk"})
    # Task A: two steps, one hit one miss. Task B: one step, a hit.
    rowsA = prepare([("passport", [a1], [b1], "A", "s1"),
                     ("passport", [b1], [a1], "A", "s2"),
                     ("passport", [a1], [b1], "B", "s3")])
    f, c, n = macro(rowsA, 50)
    ck("macro groups steps by task, not by step", n, 2)
    # micro would be 2/3 = 66.7%; macro is (0.5 + 1.0)/2 = 75%.
    ck("macro weights a task, not a step", round(f, 1), 75.0)
    ck("micro over the same rows differs",
       round(score(rowsA, pick_first, "dom", 50)[0] / 3 * 100, 1), 66.7)
    ck("macro ceiling is recall, same grouping", round(c, 1), 100.0)

    print("\n  " + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=SHARD)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--topk", type=int, default=50,
                    help="shortlist size for the ranked condition")
    ap.add_argument("--sweep", default="",
                    help="comma-separated shortlist sizes, e.g. 500,200,100,50,20,10,5,1")
    ap.add_argument("--xenc", default="",
                    help="path to scores_all_data.pkl, the published "
                         "cross-encoder scores; without it the text stand-in "
                         "is used and every table says so")
    ap.add_argument("--compare", action="store_true",
                    help="published element accuracy against the floor and "
                         "ceiling of the setting each was run in")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.xenc:
        print(f"  ranker: the published cross-encoder, "
              f"{load_xenc(a.xenc):,} steps scored\n")
    else:
        print("  ranker: text similarity stand-in (pass --xenc for the real one)\n")

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

    if a.sweep:
        ks = [int(x) for x in a.sweep.split(",") if x.strip()]
        n, acc = sweep(rows, ks)
        print(f"\n  SHORTLIST SWEEP: the floor as the shortlist tightens, over {n:,} steps\n")
        print(f"  {'k':>5} {'recall@k':>10} {'first (dom)':>13} {'first (rank)':>13} "
              f"{'centre':>9}")
        pk, pv = None, -1.0
        for k in sorted(acc, reverse=True):
            v = acc[k]
            d = v["dom"] / n
            if d > pv:
                pv, pk = d, k
            print(f"  {k:>5} {v['recall']/n:>9.1%} {d:>12.1%} "
                  f"{v['rank']/n:>12.1%} {v['centre']/n:>8.1%}")
        print(f"\n  first (dom) peaks at k={pk}: {pv:.1%}")
        print("  recall@k in the same table because it caps every column beside it.")

    if a.compare:
        print()
        compare(rows)

    print(f"\n  text-labelled {len(lab):,} steps, no-text control {len(unl):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
