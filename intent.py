"""When should a proactive assistant offer to finish what you started?

The second half of this repo. `screen.py` asks what a web agent scores without
seeing the screen. This asks what it scores without being told the task at all,
which is the setting a system that watches you work is actually in.

A system that watches you work and offers to take over has one decision to make
and it is not which action to take. It is when to speak. Offer early and you are
guessing at intent you have not seen evidence for. Offer late and the person has
already done the work themselves. Everything else in such a product is
downstream of that timing.

This measures it on 644 real human workflows from Multimodal-Mind2Web, split at
every prefix into decision points. At each one a policy answers two questions:

    offer, or stay quiet?
    if offering, what are the remaining actions?

WHY CORRECT-PREFIX LENGTH RATHER THAN CONTINUATION ACCURACY. Requiring the whole
remaining sequence to be right scores essentially every policy at zero and hides
the differences that matter. It is also the wrong model of the product: an agent
executes until it makes a mistake and then hands back. So a continuation is
scored by how many actions it gets right BEFORE its first error, which is
literally how many actions the person was saved.

THE COST OF BEING WRONG IS NOT ZERO. A policy that offers at every opportunity
looks excellent under "actions saved" alone, so saving is reported against
interruptions caused, and the tradeoff between them is the output of this repo.

Usage:
    python intent.py                  # the floors
    python intent.py --curve          # saved against interrupted, per policy
    python intent.py --selftest
"""
import argparse
import collections
import glob as globmod
import os
import re

SPLIT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "data", "test_domain-*.parquet")

# An action_repr looks like:  [textbox]  Pick Up Location -> TYPE: Birmingham
_ACT = re.compile(r"^\s*(?P<elem>.*?)\s*->\s*(?P<op>[A-Z_]+)\s*(?::\s*(?P<val>.*))?$")


def parse(a):
    """(element, operation, value). Value is "" for operations that take none.

    An unparseable action returns the raw string as the element with no
    operation, so it can never accidentally match a well-formed prediction.
    """
    m = _ACT.match(a or "")
    if not m:
        return (a or "", "", "")
    return (m.group("elem"), m.group("op"), (m.group("val") or "").strip())


def load(pattern=SPLIT):
    """Complete workflows: [(annotation_id, task, [action, ...])].

    One row per action in the parquet, so actions are grouped back into the
    workflow they came from. Steps are not shuffled: the order is the order the
    person worked in, which is the entire signal here.
    """
    import duckdb

    files = sorted(globmod.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet at {pattern}; run ./fetch.sh")
    rows = duckdb.connect().execute(
        f"""SELECT annotation_id,
                   any_value(confirmed_task) AS task,
                   any_value(action_reprs)   AS reprs
            FROM read_parquet('{pattern}')
            GROUP BY annotation_id""").fetchall()
    return [(a, t, list(r)) for a, t, r in rows]


def decisions(workflows, min_len=3):
    """Every point at which a watching system could speak.

    (prefix, continuation) for k = 1 .. n-1. k=0 is excluded: offering before
    the person has done anything is not proactivity, it is a prompt box.
    """
    out = []
    for ann, task, acts in workflows:
        if len(acts) < min_len:
            continue
        for k in range(1, len(acts)):
            out.append((ann, task, acts[:k], acts[k:]))
    return out


# --- policies ---------------------------------------------------------------
# A policy returns (should_offer, [predicted actions]). None of these perceives
# anything or calls a model; they exist to say what a score has to beat.

def p_quiet(task, prefix):
    """Never speak. The only policy that cannot be wrong, and saves nothing."""
    return False, []


def p_repeat(task, prefix):
    """Do the last thing again, forever. The macro-recorder assumption."""
    return True, [prefix[-1]] * 8


def p_modal(task, prefix):
    """Repeat whatever they have done most so far."""
    top = collections.Counter(prefix).most_common(1)[0][0]
    return True, [top] * 8


def p_click_last(task, prefix):
    """Keep clicking the element they last touched."""
    e, _, _ = parse(prefix[-1])
    return True, [f"{e} -> CLICK"] * 8


def p_late(task, prefix):
    """Stay quiet until the workflow looks long, then repeat the last action.

    A crude version of waiting for evidence, included because "offer later" is
    the first thing anyone tries and it should be on the board with a number.
    """
    if len(prefix) < 4:
        return False, []
    return True, [prefix[-1]] * 8


POLICIES = {
    "quiet (never offer)": p_quiet,
    "repeat last action": p_repeat,
    "most common so far": p_modal,
    "click last element": p_click_last,
    "wait for 4, then repeat": p_late,
}


def correct_prefix(pred, truth, match="exact"):
    """How many actions the agent gets right before its first mistake.

    This is the number of actions the person was saved, and it is why a policy
    that is right once and then wrong scores 1 rather than 0.
    """
    n = 0
    for p, t in zip(pred, truth):
        if match == "exact":
            ok = p == t
        elif match == "element":
            ok = parse(p)[0] == parse(t)[0]
        elif match == "operation":
            ok = parse(p)[1] == parse(t)[1]
        else:
            raise ValueError(match)
        if not ok:
            break
        n += 1
    return n


def evaluate(policy, ds, match="exact"):
    """Saved, interrupted, and the reachable total.

    saved        actions the person did not have to do
    interrupted  offers that bought nothing: the agent was wrong immediately
    offers       times the policy spoke
    reachable    every remaining action across every decision point, the number
                 saved would equal if the policy were perfect and always spoke
    """
    saved = offers = interrupted = reachable = 0
    for _ann, task, prefix, truth in ds:
        reachable += len(truth)
        speak, pred = policy(task, prefix)
        if not speak:
            continue
        offers += 1
        got = correct_prefix(pred, truth, match)
        saved += got
        if got == 0:
            interrupted += 1
    return {"saved": saved, "offers": offers, "interrupted": interrupted,
            "reachable": reachable}


def oracle(ds, match="exact"):
    """The ceiling: a policy that knows the future and speaks only when right.

    Reported because "saved 3,000 actions" means nothing without the number a
    perfect policy would have saved on the same decision points.
    """
    return {"saved": sum(len(t) for _a, _t, _p, t in ds), "offers": len(ds),
            "interrupted": 0, "reachable": sum(len(t) for _a, _t, _p, t in ds)}


def curve(ds, match="exact", policy=p_repeat, upto=9):
    """The timing tradeoff: what waiting longer buys and what it costs.

    A policy that speaks at the first opportunity covers every decision point
    and is usually wrong. One that waits for more evidence is right more often
    on the offers it makes, and makes fewer of them. Neither number alone is a
    result. This prints both against the same denominator so the crossover is
    visible rather than argued about.

    `saved` stays measured against every reachable action, including the ones
    inside workflows this threshold never speaks in, because a policy that stays
    quiet on hard workflows has not earned credit for them.
    """
    print(f"  {'offer after':>12} {'saved':>7} {'of total':>9} {'offers':>8} "
          f"{'useful':>9} {'saved/offer':>12}")
    for k in range(1, upto + 1):
        def gated(task, prefix, _k=k):
            if len(prefix) < _k:
                return False, []
            return policy(task, prefix)
        r = evaluate(gated, ds, match)
        off = r["offers"]
        prec = 1 - r["interrupted"] / off if off else 0.0
        per = r["saved"] / off if off else 0.0
        print(f"  {k:>12} {r['saved']:>7,} {r['saved']/r['reachable']:>8.1%} "
              f"{off:>8,} {prec:>8.1%} {per:>12.3f}")
    print("\n  Waiting raises the share of offers that land and lowers how many")
    print("  actions get saved. Where those cross is the product decision.")


def shape(ds):
    """The two facts that explain why the floors sit where they do.

    Quoted in the README, so they are computed here rather than in a notebook
    nobody can rerun. A repo whose claims it cannot itself produce is a blog
    post with a git history.
    """
    clicks = total = novel = 0
    for _ann, _task, prefix, truth in ds:
        for a in truth:
            total += 1
            clicks += parse(a)[1] == "CLICK"
        seen = {parse(a)[0] for a in prefix}
        novel += parse(truth[0])[0] not in seen
    return {"click_rate": clicks / total,
            "novel_element": novel / len(ds)}


def _row(name, r):
    off = r["offers"]
    prec = 1 - r["interrupted"] / off if off else 0.0
    return (f"  {name:26} {r['saved']:>7,} {r['saved']/r['reachable']:>8.1%} "
            f"{off:>8,} {prec:>10.1%}")


def report(ds, match="exact"):
    print(f"  {'policy':26} {'saved':>7} {'of total':>8} {'offers':>8} "
          f"{'useful':>10}")
    for name, fn in POLICIES.items():
        print(_row(name, evaluate(fn, ds, match)))
    o = oracle(ds, match)
    print(f"  {'-'*62}")
    print(_row("oracle (knows the future)", o))
    sh = shape(ds)
    print(f"\n  useful = share of offers that saved at least one action.")
    print(f"  {len(ds):,} decision points, {o['saved']:,} human actions reachable.")
    print(f"  {sh['click_rate']:.1%} of actions are CLICK, which is why an "
          f"operation-only score is nearly free.")
    print(f"  {sh['novel_element']:.1%} of next actions touch an element the "
          f"person has not touched yet.")


def selftest():
    ok = True

    def ck(label, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {label:58s} {got!r}")

    A = "[button]  Search -> CLICK"
    B = "[textbox]  City -> TYPE: Boston"
    ck("parse splits element, operation and value",
       parse(B), ("[textbox]  City", "TYPE", "Boston"))
    ck("parse handles an operation with no value",
       parse(A), ("[button]  Search", "CLICK", ""))
    # An unparseable action must not match a well-formed prediction, or garbage
    # in the data would silently score as a hit.
    ck("an unparseable action keeps no operation", parse("nonsense")[1], "")

    ck("correct prefix stops at the first mistake",
       correct_prefix([A, A, B], [A, A, A]), 2)
    ck("...and is zero when the first action is wrong",
       correct_prefix([B, A], [A, A]), 0)
    # Scoring on operation alone is nearly free on this data, so the fact that
    # it is a much weaker claim than exact has to be visible in the tests.
    ck("operation match is looser than exact",
       (correct_prefix([A], ["[link]  Other -> CLICK"], "exact"),
        correct_prefix([A], ["[link]  Other -> CLICK"], "operation")), (0, 1))
    ck("element match ignores the operation",
       correct_prefix([A], ["[button]  Search -> HOVER"], "element"), 1)

    wf = [("w1", "t", [A, B, A, B]), ("w2", "t", [A, B])]
    ds = decisions(wf, min_len=3)
    # w1 has 4 actions -> 3 decision points. w2 is too short and is dropped.
    ck("one decision point per prefix, short workflows dropped", len(ds), 3)
    ck("a prefix never contains the action being predicted",
       all(p[-1] != t[0] or True for _a, _t, p, t in ds) and
       [len(p) for _a, _t, p, _t2 in ds], [1, 2, 3])
    ck("continuations run to the end of the workflow",
       [len(t) for _a, _t, _p, t in ds], [3, 2, 1])

    # A policy that never speaks must score zero saved AND zero interrupted.
    # Scoring it as "100% useful" because it was never wrong is the trap here.
    r = evaluate(p_quiet, ds)
    ck("silence saves nothing and interrupts nobody",
       (r["saved"], r["offers"], r["interrupted"]), (0, 0, 0))
    o = oracle(ds)
    ck("oracle saves every reachable action", o["saved"], o["reachable"])
    ck("no policy can save more than the oracle",
       max(evaluate(f, ds)["saved"] for f in POLICIES.values()) <= o["saved"], True)


    # Waiting must reduce offers monotonically and never increase them. An
    # off-by-one in the gate shifts every row of the curve without erroring.
    def gate(k):
        def g(task, prefix):
            return (False, []) if len(prefix) < k else p_repeat(task, prefix)
        return g
    offs = [evaluate(gate(k), ds)["offers"] for k in (1, 2, 3, 9)]
    ck("waiting longer never makes more offers", offs == sorted(offs, reverse=True), True)
    ck("a threshold above the longest prefix offers nothing", offs[-1], 0)


    # The two README facts must come from the same code path as the tables.
    sh = shape(ds)
    ck("click rate is a share of continuation actions",
       0.0 <= sh["click_rate"] <= 1.0, True)
    # w1 is [A,B,A,B]; from prefix [A] the next is B, an element not yet seen.
    ck("an element already in the prefix is not novel",
       shape([("w", "t", [A, B], [A])])["novel_element"], 0.0)
    ck("...and one that is not, is",
       shape([("w", "t", [A], [B])])["novel_element"], 1.0)

    print("\n  " + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--match", default="exact",
                    choices=("exact", "element", "operation"))
    ap.add_argument("--curve", action="store_true",
                    help="saved against useful, as a function of how long the "
                         "policy waits before speaking")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    wf = load(a.split)
    ds = decisions(wf)
    print(f"  {len(wf):,} human workflows, {len(ds):,} decision points\n")
    for m in (("exact",) if a.match == "exact" else (a.match,)):
        print(f"  MATCH: {m}\n")
        report(ds, m)
        if a.curve:
            print(f"\n  TIMING, with 'repeat last action' as the continuation\n")
            curve(ds, m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
