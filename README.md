# perception-floor

What does a policy that never looks at the screen score on a computer-use
benchmark?

Multimodal-Mind2Web ships screenshots and is used to measure web agents, so a
number from it reads as a perception result. This measures how much of it is
solvable with the screenshot deleted, the task unread, and no model in the loop
at all.

```
                       all steps        text-labelled     no text (control)
candidates in dom order
  first             17.9% (63/352)     17.5% (55/314)      21.1% (8/38)
  overlap            8.0% (28/352)      8.9% (28/314)       0.0% (0/38)
  centre             1.7%  (6/352)      1.9%  (6/314)       0.0% (0/38)

candidates in shuffle order
  first              0.0%  (0/352)      0.0%  (0/314)       0.0% (0/38)
  overlap            2.6%  (9/352)      2.9%  (9/314)       0.0% (0/38)
  centre             1.1%  (4/352)      1.3%  (4/314)       0.0% (0/38)
```

**Picking the first candidate in document order scores 17.9%.** No model, no
pixels, no reading of the task.

```bash
python floor.py                 # the table above
python floor.py --selftest      # the scorer, on cases with known answers
```

## The three policies

None of them perceives anything and none of them calls a model.

**first** picks the first candidate in document order.
**overlap** picks the candidate whose label shares the most words with the task.
**centre** picks the candidate nearest the middle of the viewport.

## The two controls, which are the point

A floor number on its own proves nothing, because a policy can score for a
reason other than the one claimed. Two controls separate them.

**Shuffling the candidates** removes every positional signal and leaves content
untouched. `first` drops from 17.9% to 0.0%, which is what establishes that it
is reading position rather than anything else.

It also corrects `overlap`. Text matching appears to buy 8.0%, but 5.4 of those
points are the same positional prior arriving through tie-breaking, because the
first candidate with the best score wins. Once order carries nothing, naive text
matching is worth **2.6%**.

**The no-text subset** holds steps whose correct element carries no label at
all, only an icon or a bare div. `overlap` has nothing to match on and scores
**0.0%** there in both orders, which confirms text is what it uses. `first` is
unaffected at 21.1%, because position does not care about labels.

## What this claims, and what it does not

It does **not** claim Multimodal-Mind2Web is broken. It is a real dataset built
from real sites, and 17.9% is a long way from a solved benchmark.

It claims two narrower things.

**An element-accuracy score should be quoted against its floor.** A model
reporting 40% on this task is reporting 40%, of which some part is a positional
prior available to a policy that never looked at the page.

**The floor comes from the candidate list, not from the pixels.** Every number
here is produced with the screenshot deleted. Whatever the benchmark measures
in those 17.9%, it is not perception.

## The limitation that matters

These numbers are over the **full candidate set**, every element on the page.
The standard Mind2Web setup ranks candidates with a separate model first and
shows the agent a top-k shortlist. That ranker is itself trained on the DOM, so
the positional prior may be stronger or weaker after ranking, and this file does
not measure that. Reproducing the ranking stage is the obvious next step and
until it exists, these numbers describe the unranked task.

Measured on shard 0 of the 11-shard `test_domain` split: 370 action steps, 352
of which carry a positive candidate.

## The bug that would have shipped a fake result

The first version assembled candidates as `pos + neg`, putting the correct
element at index 0 on every step. `first` scored **100%**, which looked like an
extraordinary finding and was the harness handing over the answer.

`order()` now sorts by node id, and the selftest asserts that `first` misses
when the positive is late in the document. Any measurement of a floor is one
indexing mistake away from being a press release, so the control that catches it
belongs in the test suite rather than in someone's memory.
