# Three Minutes

The panel is mixed -- platform engineers alongside people from outside
engineering entirely. They cannot run the benchmark and do not have the card.
Every device below exists to make the work legible to someone who cannot verify
it directly.

## The problem the bar chart has

A slide reading `2.7x` asks the room to take your word for it, and a dozen teams
will have one. Worse, the natural sentence -- *"we fused attention and softmax
into one kernel"* -- earns a polite nod and nothing else.

So the talk does not open with the kernel. It opens with the reason kernels get
written.

## Open with the method, not the operator

> Every AI company employs people whose job is making one matrix multiply 20%
> faster. It is the highest-paid, least scalable work in the industry. We
> automated it.

An engineer follows that. So does someone who has never opened a profiler. And
it upgrades the work from *an operator* to *a method*, which is what the problem
statement asked for when it made agent-written kernels a scoring criterion.

Then two external yardsticks, so nobody has to trust us to grasp the scale:

- **KernelBench** finds that frontier reasoning models match the PyTorch
  baseline on well under 20% of kernel tasks. This is hard, and that is not our
  claim.
- **HAN Lab Kernel Mafia** took 1st, 2nd and 3rd across the three NVIDIA tracks
  of the MLSys 2026 FlashInfer contest with a fully agent-driven workflow. The
  approach works, and that is not our claim either. Ours is porting it to a
  different card and a different operator family in seventy-two hours.

## The roofline is the strongest device available

Every implementation the agent tried becomes one point on a roofline plot,
coloured by search round. The audience watches the cloud migrate from the
memory-bound ramp toward the compute ceiling.

It works on this panel because the ceiling is *physical*:

> This line is the card's memory bandwidth. This one is its peak arithmetic.
> No software crosses either. The GEMM-bound shape is served at 77% of the
> measured roof.

That sentence is true for a regulator and for a compiler engineer. It also
closes "could you have gone faster" by drawing the wall -- one figure that says
both *we did a lot of engineering* and *we know where the limit is.*

Draw it against **measured** peaks -- a large GEMM and a large copy timed on the card -- never spec-sheet numbers.

## Organize the talk around one real discovery

Numbers do not produce awe. A machine finding something you did not expect does.
So the middle of the talk is one genuine result out of the search DAG.

Given 1024 tokens and roughly a hundred kernel launches per forward, the likely
shape of it: **the win was not a faster kernel, it was fewer kernels** -- CUDA
Graph capture plus deleting the four whole-tensor materializations per layer.
But the value is not the finding, it is that the agent found it and there is a
profile to prove it (nsys timelines on this box -- counters were denied, and
the report says so).

Which imposes one preparation rule from day one: **log everything, including the
branches that died.** This story can be planted in advance. It cannot be
reconstructed afterwards.

## Hand the panel the shape list

Every evaluation shape is public. Give them the list and say *pick one.* Then
run the organizer's script live:

```bash
./scripts/demo.sh <their pick>
```

`PASS` per trial, `max_abs` and `max_rel` on screen, both medians side by side.
One command proves three things at once: we did not cherry-pick the shape, the
numbers are real, and the system runs.

## The script

| Time | Beat |
|---|---|
| 0:00-0:25 | The thesis, and the two external yardsticks. *"Making one matmul 20% faster is the highest-paid, least scalable job in the industry."* Then KernelBench's under-20%, then the MLSys 2026 placements. |
| 0:25-0:55 | The method, one diagram, three stages: correct, then profile-guided optimization, then shape-group specialization. Stress that it does not guess -- it reads the profile to choose the next move. |
| 0:55-1:35 | The roofline filling in, and the discovery. Point cloud migrating up and to the right; say the counter-intuitive finding out loud. |
| 1:35-2:05 | Let them pick a shape. Run it live. Then the ladder: *"what we beat is not eager PyTorch, it is `torch.compile max-autotune`."* |
| 2:05-2:35 | The ablation, three bars. *"We didn't just build an agent -- we measured which part of it was doing the work."* |
| 2:35-3:00 | The precision budget, one limitation, one next step. *"Roughly 2% of tolerance, zero bad elements allowed. Here is where we spent it and the one place we refused to."* Limitation: *"one card, one layer family. The agent is hardware-specific by construction -- a new card means re-running the search, and that is exactly the point."* |

## Four things not to do

**No CUDA source on a slide.** Nobody reads it, and it reads as showing off
rather than explaining.

**Do not open with the speedup.** State the problem first. The number lands only
after the room cares about it.

**Never say "N times faster" without immediately saying "than what."** A bare
multiple invites the one question that can sink the talk.

**Do not downplay that the kernels were agent-written.** It is an explicit
scoring criterion and it is the only handle non-specialists have on the work.
The kernel is the output. The method is the project.

## Q&A: one question, one artifact

| They ask | You show |
|---|---|
| "Why not just use `torch.compile`?" | The baseline ladder |
| "Did the AI write this, or did you?" | The ablation and the search DAG |
| "Does this work on other models?" | The second, unrelated kernel |
| "How do you know it's correct?" | Zero-bad-element rule; let them pick a shape |
| "What's the business value?" | Cost per GPU-hour, plus the compounding hardware-generation argument |
| **"What doesn't it do?"** | The failure record. Volunteering this is worth more than being asked. |
