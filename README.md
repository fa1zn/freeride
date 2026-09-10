# nonchalant

**Can an agent help when nobody spells it out?**

The bet behind every proactive assistant is that AI should be able to operate
without people having to communicate everything explicitly. That is a claim you
can test, and nobody had.

So: take the standard benchmark for computer-use agents and remove the explicit
instruction. Then remove the screen. Then remove the model. Score what is still
solvable at each step.

Two answers came back. Taking the instruction away breaks it, which is the
finding for the thesis. And the control, taking *everything* away, found
something worse: **five of the seven published systems score at or below a rule
that never looks at the page.**

Two questions, one method. Both halves matter and neither is a footnote to the
other.

| | what is taken away | what is left | the setting |
|---|---|---|---|
| [`screen.py`](screen.py) | the screenshot, the instruction, the model | where things sit on the page | an agent told what to do |
| [`intent.py`](intent.py) | the instruction, the model | what the person has done so far | an assistant that has to guess |

The second is the harder one and it is where products actually live: a system
that watches you work and has to decide when to speak before you have said
anything.

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

`intent.py` asks the version of this that a proactive assistant faces. Nobody
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

## The third question: does a model do better?

Everything above is a floor or a number quoted from the 2023 paper. `model.py`
asks a model we run, on 300 of the same decision points, in two conditions
one variable apart: **blind** sees the actions so far and every element on the
page, and is not told the task. **told** sees the same and is told.

Every candidate, not the top-50, because the authors' ranker scores each
element by relevance to the task text; a blind model handed the shortlist has
been told the task through the list. The floor is the same rule as before,
first element in page order, on the same 300 points.

```
Llama-3.3-70B, cross-domain split, n=300

                       blind     told     floor (reads nothing)
  right element        15.3%    17.7%     19.7%   first in page order
  right operation      72.3%    72.3%     82.7%   always CLICK
```

**A 70B model told exactly what the person wants, shown every element on the
page, picks the right one 17.7% of the time. Picking the first element scores
19.7%.** It loses to the floor with the instruction and without it, on the
element and on the operation.

**Being told the task is worth 2.4 points.** Whatever the model uses to choose
an element, the goal is not much of it.

**Blind, the model knows when it is guessing.** Element accuracy by its own
confidence: 8.6% when it says 0-24, 20.1% when it says 75-100, and that top
bucket is the only number in the table above the floor. Told, it reports high
confidence on 295 of 300 and is wrong on 82% of them. Knowing the goal made it
confident, not correct. That is the first evidence here that *when to speak* is
readable from the model's own signal, which is the decision a proactive
assistant actually has to make.

Three things this does not say. It is one model, one split, 300 points. The
all-candidates setting is harder than the paper's top-50, so these numbers do
not sit beside the table above; `told_ranked` in `model.py` is the condition
that does, and needs the authors' scores file. And a model that picked the
other search box on a page with two is scored wrong, exactly as `intent.py`
does, so these are conservative.

Mind2Web is from 2023 and the model has likely seen it. If memorised
trajectories were helping, `told` would be the condition to show it. It is at
17.7%.

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

python screen.py --xenc data/scores_all_data.pkl --compare
python screen.py --xenc data/scores_all_data.pkl --sweep 500,100,50,10,1
python intent.py --curve
python screen.py --selftest && python intent.py --selftest
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
