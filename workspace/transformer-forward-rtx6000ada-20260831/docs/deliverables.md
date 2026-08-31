# Deliverables

What the project has to be able to *show*, and which artifact shows it. The
judging rubric is the organizing principle because it is the list of questions
that will actually be asked -- not because the goal is to game it. Every row
below is a thing that has to be true of the work first and demonstrable second.

## Where Track 3 naturally stands

The final round ranks across tracks: this operator work is compared against a
privacy filter and a shopping agent by the same panel. That comparison is not
uniform, and pretending otherwise wastes the preparation.

| Criterion | Weight | Natural position | Why |
|---|---:|---|---|
| Technical Execution | 35% | Strong | Nothing in the other tracks reads as harder |
| Innovation & Problem Insight | 20% | **Weak** | The problem statement already framed the problem |
| Impact & Relevance | 20% | **Weakest** | A kernel makes a number smaller; the next track protects users |
| Feasibility & Practicality | 15% | Underrated strength | Passing zero-tolerance correctness on every shape *is* production shape |
| Presentation | 10% | Weak | Legibility, not quality |

Half the rubric is a home game and half is not. The work is to make the strong
half unarguable and give the weak half something real to hold.

## Technical Execution -- 35%

The rubric asks for *"technical complexity [that] reflects deliberate, capable
decision-making."* Not the largest number. Evidence of judgement.

| Evidence | Where it comes from | The question it answers |
|---|---|---|
| Baseline ladder | `scripts/run_ladder.sh` | "Why not just use `torch.compile`?" |
| Precision budget | `docs/precision-budget.md` from `runs/benchmark.csv` | "How do you know it's correct?" |
| Failure record | `runs/solutions.jsonl`, `decision: reject` | "What did you try that didn't work?" |
| Reproducibility | median and p90 across runs, never best-of | "Is that number repeatable?" |

The ladder is the load-bearing one. `--compile-baseline --compile-mode
max-autotune` is a flag in the organizer's own script, so the comparison is one
command away for anyone who asks. Running it first turns the hardest question in
the room into a slide: *the thing we beat is `torch.compile max-autotune`, and
this is the net difference.*

## Innovation & Problem Insight -- 20%

The rubric asks *"how clearly the team has framed the challenge."* This is the
structural weak spot: the problem statement did the framing. Three levels of
answer, weakest first.

1. *This workload's shape profile -- six layers, d=512, 1024 tokens, ~19M
   parameters -- is a ranking or retrieval model, not an LLM. So we attacked
   launch overhead and memory round trips before peak arithmetic.*
2. *At this size the launch overhead of ~100 kernels exceeds the arithmetic. The
   problem is not making the math faster; it is stopping the GPU from idling
   between kernels.*
3. *The problem is not that this transformer is slow. The problem is that kernel
   optimization does not scale, because it requires a scarce human.*

The first two are technical insight -- only a judge who has done this work can
appreciate them. The third is problem framing, any judge on any panel can
evaluate it, and it is genuinely ours: the statement said "optimize an
operator", and restating it as "optimize the act of optimizing operators" is
where the project actually lives.

## Impact & Relevance -- 20%

The rubric's exact words: *"relevance that goes beyond solving for the hackathon
prompt alone."* That sentence names its own evidence.

| Action | Cost | What it proves |
|---|---|---|
| Run one unrelated kernel through the same loop | ~1 hour | The agent is not hard-wired to this problem |
| Converge on a second card and compare | a few GPU-hours | The method transfers across hardware |
| Name the stakeholder correctly | free | ML infrastructure teams, not "users" |

The second unrelated kernel is the highest-return hour in the entire schedule.
A softmax, a LayerNorm, a plain GEMM -- it does not matter which. One run turns
"we built an agent for this problem" into "we built an agent."

The argument that holds up: as models multiply and hardware turns over
(A100 to H100 to B200), the human cost of re-optimizing compounds. A system that
can re-derive kernels per hardware generation appreciates over time rather than
depreciating. That is a claim about a real cost curve, not a projection.

## Feasibility & Practicality -- 15%

The rubric asks for *"grounded rather than speculative."* Passing a
zero-bad-element correctness check on every shape already is that.

The detail almost nobody builds: **fallback dispatch.** *When our kernel does
not beat `torch.compile` on a shape, the system dispatches to `torch.compile`.
It is never slower than the baseline.* That is an optimization layer that only
takes over where it is confident -- ordinary production thinking, and it turns
"we lost on three shapes" from a deduction into a design decision.

`kernels/dispatch.py` implements it, and `python kernels/dispatch.py calibrate` builds the table under
a deliberately conservative rule (correctness on every case in the group, worst-
case speedup above a margin, baseline fallback otherwise).

## Presentation -- 10%

The gap to close is comprehension, not quality. Against a shopping agent, this
project is harder to understand and more impressive; the second half only counts
if the first half is solved.

Three devices that convey weight without requiring anyone to follow the
technical argument:

- **The roofline's physical ceiling.** *This line is the card's memory
  bandwidth, this one is its peak arithmetic. No software crosses either. We
  went from 11% of the roof to 74%.* True for a regulator and a compiler
  engineer alike, and it closes "could you go faster" by drawing the wall.
- **KernelBench's finding** that frontier reasoning models match the PyTorch
  baseline on well under 20% of tasks. Establishes that this is hard, from an
  external source.
- **Watching the agent fail and recover, live.** Nothing conveys "this is a
  system, not a demo" faster.

## One asset, several answers

| Asset | Feeds |
|---|---|
| Skill ablation, three bars | Innovation, Technical Execution, Presentation |
| Second unrelated kernel | Impact, Innovation, Feasibility |
| Two cards converging differently | Impact, Innovation, Presentation |
| Fallback dispatch | Feasibility, Technical Execution |
| Baseline ladder | Technical Execution, Feasibility, and it defuses the hardest question |
| Roofline migration | Presentation, Technical Execution |
| Failure record | Technical Execution, Feasibility, Q&A credibility |

If only three fit in the schedule: the ablation, the second kernel, and the two
cards. Between them they touch four of the five criteria.
