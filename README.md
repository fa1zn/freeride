# blindsight

*Blindsight, in neurology, is responding correctly to something you cannot see.*

What does a policy that never looks at the screen score on a computer-use
benchmark?

Multimodal-Mind2Web ships screenshots and is used to measure web agents, so a
number from it reads as a perception result. This measures how much of it is
solvable with the screenshot deleted, the task unread, and no model in the loop
at all.

**Inside the top-50 shortlist the published pipeline actually hands its agent,
picking the first element in document order scores 38.4%.** No model, no pixels,
no reading of the task.

```
RANKED by the published Mind2Web cross-encoder, over 3,838 steps

      k   recall@k   first (dom)   first (rank)   centre
    500     99.6%       22.4%         24.7%        1.1%
    200     98.3%       26.7%         24.7%        2.0%
    100     95.3%       31.6%         24.7%        3.5%
     50     90.8%       38.4%         24.7%        5.9%   <- the published setting
     30     86.0%       42.9%         24.7%        7.6%
     20     81.2%       44.4%         24.7%        9.5%
     10     70.8%       44.5%         24.7%       13.0%
      5     57.2%       40.4%         24.7%       17.9%
      3     46.1%       35.5%         24.7%       21.0%
      1     24.7%       24.7%         24.7%       24.7%
```

The ranker is not a stand-in. These are `scores_all_data.pkl`, the candidate
generation output the Mind2Web authors published and their own action-prediction
stage consumes, so this is the ranking the reported numbers rest on. Re-derived
ranks match the ones shipped in that file on 99.93% of steps, the rest being
ties on identical scores.

```bash
pip install duckdb            # the only dependency
./fetch.sh                    # the split and the scores (~3.9 GB)
python floor.py --xenc data/scores_all_data.pkl --sweep 500,100,50,10,1
python floor.py               # the text stand-in, for comparison
python floor.py --selftest    # the scorer, on cases with known answers
```

No API key, no model, no GPU. A few minutes of CPU once the data is on disk.

## Ranking concentrates the positional prior

Over the whole page, document order scores 21.4%. Inside the top 50 it scores
**38.4%**, and it peaks at **44.5%** in a top-10 shortlist. The shortlist is a
tenth the size and the same prior lands far more often, so the setting real
systems are measured in carries the larger floor, not the smaller one.

**Taking the ranker's own top choice scores 24.7%**, which is the floor for any
system that acts on a ranker's first suggestion. It is flat across every k by
construction, since the top choice does not depend on how many candidates are
kept below it. A column that fails to be flat there is a bug, which is why it
is printed.

`centre` moves the other way, 1.1% to 24.7%, because a shortlist small enough
converges on picking whatever is left.

## The number that changes a decision

**`recall@50` is 90.8%.** Nine percent of steps have the correct element
discarded by the ranker before the agent acts, so element accuracy in the
standard setting is capped there no matter how good the model is.

Put beside the floor: a model reporting 40% element accuracy inside a top-50
shortlist is reporting 40% of an available 90.8%, of which 38.4 points is
reachable by a policy that never looked at the page.

## Unranked, for comparison

```
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

**`overlap` is not reported in the ranked condition.** The cross-encoder scores
on text, so measuring a text policy inside its own shortlist would be marking
its own homework. `first` and `centre` use position and geometry.

## What this claims, and what it does not

It does **not** claim Multimodal-Mind2Web is broken. It is a real dataset built
from real sites, and 38.4% is a long way from a solved benchmark.

**An element-accuracy score should be quoted against its floor and its ceiling.**
Neither is usually reported, and in the standard top-50 setting they are 38.4%
and 90.8%.

**Both numbers come from the candidate list, not from the pixels.** Every figure
here is produced with the screenshot deleted.

## Limitations

No model is run anywhere in this repo, so there is no accompanying capability
number to sit the floor beside. That is the gap worth closing next.

Ties in ranker score break by document order, which is a stated prior rather
than a neutral one. With the published scores ties are rare, since two elements
have to receive identical floats. With the text stand-in they are everywhere,
and that is most of why the stand-in gets the wrong answer.

The floor is measured on element selection alone. A full MindAct step also has
to choose an operation and a value, and this says nothing about those.

Measured over the whole 11-shard `test_domain` split: 4,060 action steps, 3,838
carrying a positive candidate. `fetch.sh` verifies each shard by opening it,
after one that stopped at 129 MB of 340 passed a non-empty check and failed at
read time.

## The text stand-in gets this wrong, which is worth knowing

Before the published scores were wired in, ranking was approximated by
bag-of-words overlap with the task. That stand-in answers the central question
backwards:

```
                        text stand-in    published cross-encoder
  recall@50                  74.4%              90.8%
  first, rank order          10.0%              24.7%
  first, dom order           20.4%              38.4%
  shape as k tightens      falls to 10%      rises, peaks 44.5%
```

The stand-in is too weak to keep the correct element, so tightening its
shortlist mostly throws the answer away, and the floor falls. A ranker good
enough to keep the answer 90.8% of the time concentrates the prior instead.
Both runs are reproducible here, `--xenc` and without.

The lesson generalises past this repo: a floor measured against an approximation
of a system component is a floor for the approximation, and it can point the
opposite way.

## Three bugs that would have shipped a fake result

All three were the harness handing over the answer, or an approximation standing
in for the thing being measured. All three were caught by a test.

**Candidates were assembled as `pos + neg`,** putting the correct element at
index 0 on every step. `first` scored **100%**. `order()` now sorts by node id,
and the selftest asserts `first` misses when the positive is late in the
document.

**`prepare()` was then written to parse each candidate once, and rebuilt the
same `pos + neg` list** without calling `order()`, so it did not inherit the
fix. The downstream re-sort into document order made the unranked numbers
correct and hid it. Python's sort is stable, so the correct element still won
every tie on ranker score by sitting at index 0, buying it preferential
admission into the shortlist. Caught by asserting that a shortlist which drops
the correct element must score zero in every column; it returned `(1, 1, 1, 0)`.

**The published-ranker path was almost tested against the wrong module.** Run as
a script this file is `__main__`, so `import floor` binds a second copy and
setting the scores on it leaves the executing module untouched. The test would
have passed while silently measuring the text stand-in under the real ranker's
name. It failed instead, because it asserts the ordering actually changes.

A floor measurement is one indexing mistake away from being a press release, so
the controls that catch it live in the test suite rather than in someone's
memory.
