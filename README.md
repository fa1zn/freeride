# blindsight

*Blindsight, in neurology, is responding correctly to something you cannot see.*

What does a policy that never looks at the screen score on a computer-use
benchmark?

Multimodal-Mind2Web ships screenshots and is used to measure web agents, so a
number from it reads as a perception result. This measures how much of it is
solvable with the screenshot deleted, the task unread, and no model in the loop
at all.

```
UNRANKED, every element on the page
                       all steps        text-labelled     no text (control)
  candidates in dom order
    first           21.4% (820/3838)   20.8% (700/3361)    25.2% (120/477)
    overlap          6.7% (257/3838)    7.6% (257/3361)     0.0%   (0/477)
    centre           1.0%  (40/3838)    1.1%  (36/3361)     0.8%   (4/477)
  candidates shuffled
    first            0.1%   (2/3838)    0.1%   (2/3361)     0.0%   (0/477)
    overlap          2.3%  (87/3838)    2.6%  (87/3361)     0.0%   (0/477)
    centre           0.8%  (32/3838)    0.8%  (28/3361)     0.8%   (4/477)

RANKED, the ranker's top 50, which is where a real system acts
  recall@50          93.6% (3592/3838)     ceiling on everything below
                       in rank order      in dom order
    first           22.7% (873/3838)   31.4% (1207/3838)
    centre           7.3% (279/3838)     7.3%  (280/3838)
```

**Inside the shortlist a real system acts on, picking the first element in
document order scores 31.4%.** No model, no pixels, no reading of the task.

```bash
pip install duckdb            # the only dependency
./fetch.sh                    # the split, verified shard by shard (~3.6 GB)
python floor.py               # the table above
python floor.py --selftest    # the scorer, on cases with known answers
```

No API key, no model, no GPU. The whole thing is a few minutes of CPU once the
split is on disk.

## Ranking raises the floor, it does not remove it

This was the obvious objection to an earlier version, and it turned out backwards.

The published Mind2Web pipeline scores every element with a cross-encoder and
hands the agent a top-k shortlist, so a floor over all ~500 candidates describes
a task nobody evaluates. The expectation was that ranking would wash the
positional prior out.

It concentrates it. Over the full page, document order scores 21.4%. Inside the
top 50 it scores **31.4%**, because the shortlist is a tenth the size and the
same prior now lands far more often. The setting real systems are measured in
carries the larger floor, not the smaller one.

**Taking the ranker's own top choice scores 22.7%**, which is the floor for any
system built on a ranker at all.

`recall@50` is 93.6%: the ranker keeps the correct element that often, which
caps every number measured inside the shortlist and belongs beside them.

## The three policies

None perceives anything and none calls a model.

**first** picks the first candidate in document order.
**overlap** picks the candidate whose label shares the most words with the task.
**centre** picks the candidate nearest the middle of the viewport.

## The controls, which are the point

A floor number alone proves nothing, because a policy can score for a reason
other than the one claimed.

**Shuffling the candidates** strips positional signal and leaves content
untouched. `first` drops from 21.4% to 0.1%, which is what establishes it reads
position and nothing else.

The same shuffle corrects `overlap`. Text matching appears to buy 6.7%, but two
thirds of that is the positional prior arriving through tie-breaking, since the
first candidate with the best score wins. Once order carries nothing, text
matching is worth **2.3%**.

**The no-text subset**, 477 steps whose correct element carries no label at all,
only an icon or a bare div. `overlap` scores **0.0%** there in both orders, which
confirms labels are what it reads. `first` is unaffected at 25.2%, because
position does not care about labels.

**`overlap` is not reported in the ranked condition.** The ranker scores on
text, so measuring a text policy inside its own shortlist would be marking its
own homework. `first` and `centre` use position and geometry and are unaffected.

## What this claims, and what it does not

It does **not** claim Multimodal-Mind2Web is broken. It is a real dataset built
from real sites, and 31.4% is a long way from a solved benchmark.

**An element-accuracy score should be quoted against its floor.** A model
reporting 40% inside a top-50 shortlist is reporting 40%, of which up to 31
points is available to a policy that never looked at the page.

**The floor comes from the candidate list, not from the pixels.** Every number
here is produced with the screenshot deleted.

## Limitations

The ranker here is text similarity to the task, standing in for the
cross-encoder the published pipeline trains. A learned ranker orders the
shortlist differently and the floor inside it would move; which way is not
measured. The direction of the effect, that a smaller shortlist concentrates a
positional prior rather than removing it, does not depend on which ranker
produces the shortlist.

No model is run anywhere in this repo, so there is no accompanying capability
number to sit the floor beside.

Measured over the whole 11-shard `test_domain` split: 4,060 action steps, 3,838
carrying a positive candidate. `fetch.sh` verifies each shard by opening it,
after one that stopped at 129 MB of 340 passed a non-empty check and failed at
read time.

## The bug that would have shipped a fake result

The first version assembled candidates as `pos + neg`, putting the correct
element at index 0 on every step. `first` scored **100%**, which looked like an
extraordinary finding and was the harness handing over the answer.

`order()` now sorts by node id, and the selftest asserts `first` misses when the
positive is late in the document. A floor measurement is one indexing mistake
away from being a press release, so the control that catches it lives in the
test suite rather than in someone's memory.
