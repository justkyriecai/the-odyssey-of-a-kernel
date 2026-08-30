# Submission Checklist

Everything on this list has failed someone at 2 a.m. on a deadline night. Work
it top to bottom on D3; nothing here takes long, and several items cannot be
done at the last minute.

## Deliverables

- [ ] **YouTube video uploaded and set to Public** — not Unlisted, not Private.
      The Track 3 deliverables require it, and a video processing queue on the
      last hour is a real risk: upload early, replace later if needed.
- [ ] **Video link pasted into the Devpost description.**
- [ ] Devpost submission form complete: track selected, team members added,
      repository link present.
- [ ] Repository is **public**, and the clone-and-run path in `README.md` works
      from a fresh checkout (`./scripts/setup_env.sh && workspace/transformer-forward-opt/scripts/smoke.sh`).
- [ ] `git ls-files | grep -E "^ref/|\.ncu-rep$"` returns nothing — reference
      material and binary profiler dumps stay out of the tree; `runs/` is in it.

## Numbers

- [ ] Final hardware, final run: a fresh `--shapes official --record` sweep and
      `python kernels/dispatch.py calibrate` on the card the report quotes, so
      `runs/dispatch_table.json` was built where the numbers were.
- [ ] The gate, per shape, on the organizer's unmodified script:
      `./scripts/demo.sh <case>` for every official case — exit 0 everywhere it
      is claimed to pass.
- [ ] Baseline ladder present for the headline shapes: the number quoted is the
      margin over `torch.compile max-autotune`, not over eager.
- [ ] Every quoted latency is a **median with p90 alongside**, from
      `runs/benchmark.csv` — no best-of runs anywhere in the report or slides.
- [ ] `stress-100k` has its own story: chunked reference, memory arithmetic,
      and which card validated it (`docs/benchmark-anatomy.md` §9).

## Figures

- [ ] Ceilings measured on the final card (a large GEMM, a large copy) — the
      roofline uses those, not spec-sheet numbers.
- [ ] The roofline regenerated from the final `benchmark.csv`.
- [ ] The ablation chart regenerated after all three arms ran, with wall clock
      recorded in each arm's `meta.json`.
- [ ] The search DAG rendered from `runs/solutions.jsonl` — the version in the
      report includes the rejected branches.
- [ ] Committed figures under `docs/assets/` match the final data (no stale
      renders from development runs).

## Report

- [ ] Precision budget table filled from the final sweep (per-shape max_abs,
      budget spent, verdict) — `docs/precision-budget.md`.
- [ ] Failure record present: every direction that hit the five-iteration cap,
      with the evidence that killed it.
- [ ] Environment disclosure: GPU, driver, CUDA runtime, torch version — all
      already columns in `benchmark.csv`; quote them.
- [ ] Prior work cited (the workflow source, KernelWiki, ncu-report-skill,
      humanize, KernelBench) — cited work reads as judgment, hidden work reads
      as risk.

## Pitch

- [ ] Three-minute run-through, timed, three times (`docs/pitch.md`).
- [ ] `./scripts/demo.sh` executed once on the demo machine, on the demo
      network, before the session — the judge-picks-a-shape moment must not be
      its own first rehearsal.
- [ ] Q&A table reviewed; the failure record is ready to volunteer.
