# Demo video script

The Devpost submission video. Target length ~1:50; recorded entirely on a
local Mac -- the GPU numbers are shown from the committed ledger, which is the
point: the evidence is the deliverable. Structure mirrors the written
description: introduce the framework, demonstrate the workflow, present the
results.

## Recording setup

- Terminal >= 18 pt, dark theme, 16:9 window; capture at 1440p or higher.
- No secrets on screen; clean shell prompt.
- Voiceover pace ~140 wpm; the verbatim script below is ~190 words.
- No third-party logos anywhere; factual text ("TikTok TechJam 2026, Track 3")
  only. No music, or CC0 only.

## Pre-flight checklist

- [ ] Fresh directory for the clone shot: `cd "$(mktemp -d)"`.
- [ ] Run `./scripts/setup_env.sh` off-camera beforehand so `smoke.sh` can run
      live during the demo segment (the on-camera onboarding rerun is
      idempotent and fast).
- [ ] Verify how `odyssey-onboarding` is invoked in a fresh clone; if the
      skill is not visible before setup, run `./scripts/setup_agent.sh`
      on camera first -- it is one line and still reads as easy.
- [ ] Rehearse the phase-prompt paste once. The workspace campaign is
      complete, so if the agent short-circuits to "already done", feed it a
      small real goal instead (e.g. "re-validate the dispatch table on the CPU
      smoke set") to generate genuine loop footage.
- [ ] Confirm the ledger freeze-frame row is on screen before recording:
      `grep seq-1024 runs/benchmark.csv | tail -1` shows
      `speedup 16.144, passed=True`.

## Part 1 -- the project (0:00-0:18)

| Shot | Screen |
|---|---|
| Title card | "The Odyssey of a Kernel" + tagline; second beat: the six loop words |

> Odyssey is an autonomous agent framework for GPU kernel optimization. The
> agent does the exploring, tuning, measuring and record-keeping; you choose
> the directions. One loop, end to end: plan, implement, measure, profile,
> record, review.

## Part 2 -- the demo (0:18-1:15, live terminal throughout)

| Shot | Screen / commands |
|---|---|
| Clone (12s) | In an empty directory: `git clone https://github.com/justkyriecai/the-odyssey-of-a-kernel.git` then `cd the-odyssey-of-a-kernel` |
| Onboarding (18s) | Open the agent (`claude`), invoke `odyssey-onboarding`; the checks scroll by (sped up), hold one second on the green passes |
| State the goal (27s) | `cd workspace/transformer-forward-opt`, paste `prompts/phase1.md`; the agent reads the README and starts planning. Speed-ramp effect: commands scrolling; cut in one real moment -- `./scripts/smoke.sh` passing on CPU in seconds |

> Using it looks like this. Clone the ship. *(pause)* Open your agent and ask
> for onboarding -- it sets up the environment, links the skills, and reads
> the rules of the voyage. *(pause)* Then the one thing a human still does:
> tell it the goal. That's it -- from here, the agent drives. Planning first.
> Then the correctness gate -- every combination of masking and padding,
> passing in seconds. Then it iterates: measure, profile, record -- again and
> again.

## Part 3 -- the results (1:15-1:45)

| Shot | Screen |
|---|---|
| The ledger (10s) | Tail of `runs/benchmark.csv`; freeze on the row `seq-1024 ... speedup 16.144 ... passed=True`; glance across one `decision: reject` node in `runs/solutions.jsonl` |
| Results card (15s) | Lines light up one by one: 13/13 official shapes -- never slower anywhere (worst 1.23x) / Ahead of the strongest numerically legal `torch.compile` config / S=100,000: 23 s, 0 bad of 3.28 B / 77% of the measured roof |
| Close card (8s) | Repo URL + closing line; hold five seconds |

> The first voyage ran on a rented RTX 6000 Ada -- and every number lives in
> the committed ledger: about four hundred measurements, each row carrying the
> benchmark's checksum and the git commit. The verdict: thirteen of thirteen
> official shapes, never slower than the baseline anywhere. Ahead of torch
> compile's strongest numerically legal configuration. The
> hundred-thousand-token shape no reference can run -- twenty-three seconds,
> zero bad elements out of three billion. And seventy-seven percent of a roof
> measured on the card itself.

> *(close)* You choose the heading. The agent makes the thousand turns.
> Everything is public.

## Editing notes

- Effects live in exactly two places: the speed-ramp after the goal is pasted,
  and the loop words animating on the title card. Everything else is a plain
  terminal.
- The FAIL discovery, the stress run and the roofline are not staged live;
  they appear only as lines on the results card. Keep it that way -- the
  ledger freeze-frame is the authenticity beat.
- YouTube: public visibility, repo link in the description, then link the
  video in the Devpost submission.
