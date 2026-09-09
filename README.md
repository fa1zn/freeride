# blindsight

**How much of a computer-use benchmark can you pass without doing the hard part?**

When people report that a web agent picks the right button 42% of the time, that
number sounds like it measures seeing and understanding. This repo checks. It
takes the same benchmark, deletes the screenshot, deletes the instruction,
removes the model entirely, and measures what is still solvable.

The answer is: a lot. On the standard benchmark, **five of the seven published
systems score at or below a rule that never looks at the page.**

Two questions, one method.

| | the thing removed | what is left |
|---|---|---|
| [`floor.py`](floor.py) | the screenshot, the instruction, the model | position on the page |
| [`foresight.py`](foresight.py) | the instruction, the model | what the person did so far |

The second one is the setting a proactive assistant is in: it watches you work
and has to guess what you want before you say it.

---

## What the benchmark is, and what an agent has to do

Multimodal-Mind2Web is 2,000 real tasks recorded on 137 real websites. "Book an
appointment for a new passport for one adult." A person did it, and every click
and keystroke was recorded, along with the page as it looked at the time.

To score, a system gets one page and has to pick the single right thing to click
out of roughly 500 options. That is one *step*. A whole task is five or six of
them in a row.

Because 500 options is too many to feed a model, the published pipeline first
runs a small ranking model that narrows them to the best 50, and the agent picks
from those 50. That two-stage design matters for everything below.

## What we did

We wrote rules that pick an option **without looking at anything**.

- **first** takes whichever option appears earliest in the page's HTML.
- **overlap** takes the option whose label shares the most words with the task.
- **centre** takes the option nearest the middle of the screen.

None of them opens the screenshot. None calls a model. `first` does not even
read the labels; it only knows the order things appear in the page source.

Then we scored those rules exactly the way a real system is scored, inside the
same 50-option shortlist, and compared them to what published systems get.

## What we found

**Picking the first option in page order gets 38.4%.**

```
system                      picks right   floor   difference
MindAct w/ Flan-T5XL             42.1      39.0        +3.1
MindAct w/ Flan-T5L              39.7      39.0        +0.7
MindAct w/ GPT-4                 37.1      45.4        -8.3
MindAct w/ Flan-T5B              33.9      39.0        -5.1
Classification (DeBERTa)         24.5      39.0       -14.5
MindAct w/ GPT-3.5               21.6      39.0       -17.4
Generation (Flan-T5B)            14.2      39.0       -24.8
```

The best model in the paper beats a rule that reads nothing by **3.1 points**.
Four of the others lose to it.

Two things make that table fair rather than a gotcha. The paper averages per
task and we originally averaged per step, which is a different denominator, so
we recomputed ours their way. And GPT-4 was run on a 10-option shortlist rather
than 50, where the floor is higher, so it is compared against 45.4 and not 39.0.

**Narrowing the list makes this worse, not better.** Over all 500 options, page
order gets 21.4%. Inside the top 50 it gets 38.4%, and inside the top 10 it peaks
at 44.5%. A shorter list is a smaller haystack, so the same dumb rule lands far
more often. The setting real systems are measured in carries the *bigger* floor.

**A quarter of the difficulty is not the model's fault.** The ranking model
throws away the correct option before the agent ever sees it on 9% of steps at
top-50 and 29% at top-10. Nothing downstream can recover it, so scores in this
setting sit against a ceiling of 90.8%, not 100%.

## The second question: can it guess what you want?

`foresight.py` asks the version of this that a proactive assistant faces. Nobody
tells it the task. It watches someone work and has to decide two things: **when
to speak up**, and **what to do if it does**.

We took 694 complete recorded workflows and cut each one at every point, giving
3,316 moments where a watching system could offer to take over. Then the same
trick: rules that use no model at all.

```
best no-model rule       exact action   right element   right operation
  work saved                  0.6%           1.1%            62.6%
  offers that helped          1.6%           3.4%            85.0%
```

**Guessing what kind of action to take is nearly free. Guessing what to do it to
is the entire problem.** 85.2% of recorded actions are clicks, so "it's a click"
is right almost always. Which thing to click, without a model, is 1.1%.

**Watching for repetition does not work.** The next thing a person touches is
something they have never touched before 93.2% of the time. They are not
repeating themselves, so there is no macro to record.

**Waiting helps, briefly.** Holding off until four actions have been observed
nearly doubles the share of offers that turn out useful, and costs a quarter of
the work saved. Past four, waiting buys nothing. That crossover is the shipping
decision.

## How this was built

**Nothing here approximates anything, and that turned out to matter.**

The ranking model is not our reimplementation. It is `scores_all_data.pkl`, the
file the Mind2Web authors published and their own pipeline reads. Our rankings
reproduce the ones shipped in that file on 99.93% of steps.

We know that matters because we tried the shortcut first. An earlier version
approximated the ranker with simple word overlap, and it answered the central
question **backwards**: it said narrowing the list *lowers* the floor. A ranker
too weak to hold on to the right answer throws it away as the list shrinks. A
real one keeps it and concentrates the positional advantage instead.

```
                          our approximation    the real ranker
right answer survives            74.4%              90.8%
floor at top-50                  20.4%              38.4%
as the list shrinks           falls to 10%       rises to 44.5%
```

**Every claim has a control**, because a rule can be right for a reason other
than the one you think.

- Shuffle the options and `first` collapses from 21.4% to 0.4%. That is what
  proves it is reading position and nothing else.
- The same shuffle exposes `overlap`: word matching looks worth 6.7%, but half
  of that is position sneaking in through tie-breaks. Once order is meaningless
  it is worth 3.1%.
- On the 477 steps where the right answer has no text on it at all, just an icon,
  `overlap` scores exactly 0.0%, confirming labels are what it reads.
- `overlap` is left out of the ranked results entirely, because the ranker itself
  scores on text and letting a text rule compete inside a text ranker's shortlist
  is grading its own homework.

**The metric for the proactive half has an obvious exploit and the tests close
it.** A system that never speaks is never wrong, so any scoring that rewards
precision without coverage will rank silence first. `quiet` is a first-class row
in the table, and a test asserts it scores zero saved *and* zero interrupted,
never 100% useful.

```bash
pip install duckdb            # the only dependency
./fetch.sh                    # the data and the published scores (~3.9 GB)

python floor.py --xenc data/scores_all_data.pkl --compare
python floor.py --xenc data/scores_all_data.pkl --sweep 500,100,50,10,1
python foresight.py --curve
python floor.py --selftest && python foresight.py --selftest
```

No API key, no GPU, a few minutes of CPU. 53 tests.

## What this does not say

It does not say the benchmark is broken. It is real tasks on real websites, and
38.4% is a long way from solved.

It says a score should be quoted against its floor and its ceiling, which in the
standard setting are 38.4% and 90.8%. A system reporting 42% captured about 6% of
the room that was actually available.

No model runs anywhere in this repo, so there is no capability number of our own
sitting beside the floors. Every model number here is quoted from the paper.

The proactive half scores against exactly what the person did, so an assistant
that reached the same goal a different way is marked wrong. That makes those
numbers conservative.

## Three bugs that would have shipped a fake result

All three were the harness quietly handing over the answer. All three were caught
by a test, not by reading the code.

**Options were built as `correct + wrong`**, putting the right answer first every
time. `first` scored 100%. Sorting by page position fixed it, and a test now
asserts `first` misses when the answer is late in the page.

**Then a rewrite reintroduced it.** A faster loader rebuilt the same
correct-answer-first list and skipped the sorting. The unranked numbers still
looked right, which hid it, but Python's sort is stable so the right answer won
every tie and got preferential entry into the shortlist. Every ranked number was
inflated. Caught by asserting that a shortlist which loses the answer must score
zero in every column; it returned `(1, 1, 1, 0)`.

**A test almost measured the wrong module.** Run as a script this file is
`__main__`, so `import floor` binds a second copy and setting the scores on it
leaves the running one untouched. The test would have passed while silently
measuring the old approximation under the real ranker's name.

A floor measurement is one indexing mistake away from being a press release, so
the controls that catch it live in the test suite rather than in someone's head.
