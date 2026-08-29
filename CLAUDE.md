# Agent Instructions

This repository is the workflow and harness for TikTok TechJam 2026 Track 3
(GPU kernel for a transformer layer). Read `README.md` and
`docs/benchmark-anatomy.md` before changing anything that measures.

## Hard rules

- **Never modify `bench/official/torch_transformer_benchmark.py`.** It is the
  organizer's evaluator, vendored so that every number is traceable to a known
  file. Its md5 is recorded in `bench/official/README.md` and checked by
  `odyssey doctor`. To run a candidate through it, use
  `odyssey official <candidate>`, which patches `UserOptimizedTransformer` at
  runtime and calls the script's own `main()`.
- **Never reimplement the correctness rule or the timing protocol.** Import them
  from the script via `odyssey.official`. If a harness number ever disagrees
  with the script's own output, that is a bug in the harness, not a difference
  of opinion about the metric.
- **Target `--atol 0.001 --rtol 0.01`**, the stricter of the two readings in the
  script. Zero bad elements; non-finite values fail outright.
- **Do not run the optimization search in this repository.** Create a workspace
  under `workspaces/` (git-ignored) and run there, with `TECHJAM_BENCHMARK`
  pointing back at the vendored script.
- **English for everything repository-facing** — code, comments, docs, prompts,
  commit messages. This will be read by people who did not write it.

## Adding a candidate

Candidates live in `kernels/` and register through `odyssey.registry.register`.
The contract is the official script's, not ours:

1. `forward(x, valid_token_mask=None) -> [batch, seq_len, d_model]`.
2. Parameter names compatible with `BaselineTransformer` — `copy_model_weights`
   loads the baseline state dict with `strict=True`. Subclass it and override
   `forward`; derive fused weights lazily on the first call; register extra
   buffers with `persistent=False`.
3. Declare `requires=` honestly (`cuda`, `half`, `no-padding`, ...). The harness
   skips a candidate whose requirements a case does not meet and says so, rather
   than silently reporting a number from a fallback path.
4. Run `./scripts/smoke.sh` before anything else. It covers every combination of
   causal and padding on CPU in seconds, and that cross is where reimplemented
   attention breaks — invisibly at `--padding-ratio 0`.

**A speed/accuracy trade is a separate named candidate, never a flag on an
existing one.** `fused-safe` and `fused-sdpa` differ only in whether softmax
happens in fp32; they are two registered candidates so that calibration can
measure both on the actual hardware and admit whichever is correct *and* faster.
That pattern is the house style.

## The recording discipline

Every measurement appends to `runs/benchmark.csv`; every candidate appends a
node to `runs/solutions.jsonl` with a parent link. Record rejected branches with
`decision: reject` and the evidence that killed them — a dead end recorded on
the day it died is evidence, and the same dead end reconstructed on the last
night is a guess.

Cap each optimization direction at five iterations. If it cannot be implemented
cleanly, fails correctness, or shows no credible path after five, write down
what you learned and take the next-ranked direction.

## Reporting numbers

- Median, with p90 next to it. Never a best-of run.
- Always name the baseline. "2.7x" without "than what" is not a result.
- Run the full shape set, not the one case that looks good. Tuning on `default`
  until everything else regresses is the standard way to lose this.
- Roofline plots use **measured** ceilings from `odyssey peak`. Every entry in
  `DEVICE_PEAKS` is marked `verified=False` because it came from a spec sheet.

## Generated vs. source

`runs/`, `outputs/`, `profile/`, `workspaces/` and `ref/` are git-ignored.
Nothing there is source, and nothing there should be needed to understand the
repository. `docs/assets/` holds committed figures; regenerate them with
`odyssey roofline` and `odyssey ablation --plot`.

After editing `prompts/transformer-layer/_shared.md`, run
`python scripts/build_prompts.py` — the phase prompts each carry an expanded
copy so they stay self-contained when pasted into a session.

Commit messages describe the change and match the existing history.
