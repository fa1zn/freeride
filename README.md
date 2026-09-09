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
    first            0.4%  (14/3838)    0.4%  (13/3361)     0.2%   (1/477)
    overlap          3.1% (120/3838)    3.6% (120/3361)     0.0%   (0/477)
    centre           0.9%  (35/3838)    0.9%  (31/3361)     0.8%   (4/477)
```

**Picking the first element in document order scores 21.4%.** No model, no
pixels, no reading of the task.

```bash
pip install duckdb            # the only dependency
./fetch.sh                    # the split, verified shard by shard (~3.6 GB)
python floor.py               # the table above
python floor.py --sweep 500,100,50,10,1
python floor.py --selftest    # the scorer, on cases with known answers
```

No API key, no model, no GPU. The whole thing is a few minutes of CPU once the
split is on disk.

## Ranking removes the floor rather than concentrating it

The published Mind2Web pipeline scores every element with a cross-encoder and
hands the agent a top-k shortlist, so a floor measured over all ~500 candidates
describes a setting nobody evaluates in. The obvious objection is that ranking
would wash the positional prior out. It does.

```
      k   recall@k   first (dom)   first (rank)   centre
    500     98.5%       21.4%         10.0%        1.1%
    200     93.3%       21.3%         10.0%        1.3%
    100     86.0%       21.1%         10.0%        2.1%
     50     74.4%       20.4%         10.0%        3.9%
     30     67.1%       20.0%         10.0%        5.7%
     20     60.0%       19.0%         10.0%        6.9%
     10     45.3%       18.1%         10.0%        7.8%
      5     31.4%       16.8%         10.0%        8.1%
      3     22.6%       15.0%         10.0%        8.7%
      1     10.0%       10.0%         10.0%       10.0%
```

The floor falls monotonically as the shortlist tightens, from 21.4% over the
whole page to 10.0% at the ranker's own top choice. **An earlier version of this
repo reported the opposite, a floor rising to 35.5% at k=10. That was a
tie-break bug, not a finding, and it is documented at the bottom.**

`first (rank)` is flat at 10.0% by construction: the ranker's top choice does
not depend on how many candidates it keeps below that. It is reported because
it is the floor for any system that acts on a ranker's first suggestion, and
because a column that fails to be flat is a bug.

`centre` moves the other way, 1.1% to 10.0%, because a shortlist small enough
converges on picking whatever is left.

## The number that changes a decision

**`recall@50` is 74.4%.** In the setting these systems are actually evaluated
in, a quarter of steps have the correct element discarded by the ranker before
the agent gets to act.

That caps element accuracy at 74.4% no matter how good the model is, and it is
a property of the retrieval stage rather than of the agent being measured. A
score reported in this setting is a score against a 74.4% ceiling and a 20.4%
floor, and neither is usually quoted.

The ranker here is text similarity, so a trained cross-encoder will do better
than 74.4%. How much better is not measured here, and it is the number worth
having.

## The three policies

None perceives anything and none calls a model.

**first** picks the first candidate in document order.
**overlap** picks the candidate whose label shares the most words with the task.
**centre** picks the candidate nearest the middle of the viewport.

## The controls, which are the point

A floor number alone proves nothing, because a policy can score for a reason
other than the one claimed.

**Shuffling the candidates** strips positional signal and leaves content
untouched. `first` drops from 21.4% to 0.4%, which is what establishes it reads
position and nothing else.

The same shuffle corrects `overlap`. Text matching appears to buy 6.7%, but half
of that is the positional prior arriving through tie-breaking, since the first
candidate with the best score wins. Once order carries nothing, text matching is
worth **3.1%**.

**The no-text subset**, 477 steps whose correct element carries no label at all,
only an icon or a bare div. `overlap` scores **0.0%** there in both orders, which
confirms labels are what it reads. `first` is unaffected at 25.2%, because
position does not care about labels.

**`overlap` is not reported in the ranked condition.** The ranker scores on
text, so measuring a text policy inside its own shortlist would be marking its
own homework. `first` and `centre` use position and geometry and are unaffected.

## What this claims, and what it does not

It does **not** claim Multimodal-Mind2Web is broken. It is a real dataset built
from real sites, and 21.4% is a long way from a solved benchmark.

**An element-accuracy score should be quoted against its floor and its ceiling.**
A model reporting 40% inside a top-50 shortlist is reporting 40% of a possible
74.4%, of which 20 points is available to a policy that never looked at the page.

**Both numbers come from the candidate list, not from the pixels.** Every figure
here is produced with the screenshot deleted.

## Limitations

The ranker is text similarity to the task, standing in for the cross-encoder the
published pipeline trains. A learned ranker keeps the correct element more often
than 74.4% and orders the shortlist differently, so both the ceiling and the
floor inside it would move. Which way is not measured, and the sweep above is
the shape for this ranker only.

Ties in ranker score are broken by document order, which is a stated prior
rather than a neutral one. A random tie-break would separate the two effects.
This matters more than it sounds like it does, and the reason is below.

No model is run anywhere in this repo, so there is no accompanying capability
number to sit the floor beside.

Measured over the whole 11-shard `test_domain` split: 4,060 action steps, 3,838
carrying a positive candidate. `fetch.sh` verifies each shard by opening it,
after one that stopped at 129 MB of 340 passed a non-empty check and failed at
read time.

## Two bugs that would have shipped a fake result

Both were the harness handing over the answer. Both were caught by a test rather
than by reading the code, and the second one had already shipped.

**The first version assembled candidates as `pos + neg`,** putting the correct
element at index 0 on every step. `first` scored **100%**, which looked like an
extraordinary finding. `order()` now sorts by node id, and the selftest asserts
`first` misses when the positive is late in the document.

**`prepare()` was then written to parse each candidate once, and rebuilt the same
`pos + neg` list.** It did not call `order()`, so it did not inherit the fix.
This was subtler, because the downstream re-sort into document order made the
unranked numbers correct and hid it. Python's sort is stable, so the correct
element still won every tie on ranker score by sitting at index 0, which bought
it preferential admission into the shortlist. Every ranked number was inflated:

```
                        shipped     correct
  recall@50              93.6%       74.4%
  first, rank order      22.7%       10.0%
  first, dom order       31.4%       20.4%
```

It also produced the headline that ranking *concentrates* the positional prior,
peaking at 35.5% inside a top-10 shortlist. That claim was entirely the bug.

What caught it was adding a shortlist sweep and asserting the invariant that a
shortlist which drops the correct element must score zero in every column. It
returned `(1, 1, 1, 0)`.

A floor measurement is one indexing mistake away from being a press release,
twice now, so the controls that catch it live in the test suite rather than in
someone's memory.
