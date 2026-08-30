# Ablation: what the harness buys

Three arms, same underlying model, same card (RTX 6000 Ada), same judge (the
unmodified organizer script at the official tolerance). Two fresh-context arms
ran ~12-13 minutes of wall-clock each, isolated from the campaign (separate
directories/worktree at the pre-campaign commit, separate Inductor caches, GPU
serialized); the full arm is this repository's recorded campaign.

**Disclosed contaminations, so the comparison is honest:** the arms are the
same model as the campaign (this ablates the HARNESS, not model capability);
both arm prompts asked for honest reporting; arm-bare's prompt included the
shape table and tolerance; arm-noskills was given one measured machine fact as
a warning (the dynamo recompile-limit trap, bitlesson BL-1) -- without it its
compiled-baseline rows would have been silently corrupted exactly as the
campaign's first attempt was. Arms could not see campaign findings, kernels
from round 1 onward, or each other.

## Headline table (speedup vs eager, the script's own verdicts)

| lane | arm-bare (13 min) | arm-noskills (12 min) | arm-full (campaign) |
|---|---|---|---|
| center | 4.09x | 2.50x | 3.9x (n=3 restated) |
| batch-1 | 12.95x | 8.31x | 12.3x |
| seq-1024 | 7.97x | 4.11x | **16.1x** |
| batch-10000 | 3.51x | not run | **3.86x** |
| wide-1024 | 1.26x | 1.09x | 1.24x (= 77% of the measured GEMM roof) |
| bf16 / fp16 | **never tested** | legal lanes, max_abs 0, 2.55-2.69x | same + the rounding-point mechanism |
| padded inputs | fallback **shipped unverified** | dev padded cases PASS | every served geometry measured |
| stress #14 (S=100k) | nothing (NO VERDICT) | not attempted | official rows to S=3072 (8.6x) + off-script 23.2 s, 0 bad of 3.28e9 |
| never-slower guarantee | none | dispatch fallback | dispatch + never-behind-the-admissible-opponent |
| evidence trail | ad-hoc logs + report | 53 CSV rows, 8 DAG nodes, calibrated table | 390+ rows, 18 nodes, roofline vs measured roofs, precision budget, profiles |

Arm artifacts: `arm-bare/` and `arm-noskills/` (their own reports, logs, CSV,
DAG, table, wall-clocks).

## Reading the table honestly

**1. A strong bare agent is fast to raw numbers on the default lanes.**
arm-bare found fused QKV + SDPA + torch.compile(reduce-overhead) in 13 minutes
and matched the campaign's launch-bound numbers. The harness's value is not
"you can't get 4x without it".

**2. The harness converts effort into ceiling and coverage.** Where knowledge
and iterations compound -- the custom flash kernel with the probed tile
(seq-1024: 16.1x vs bare's 8.0x, exactly 2x), the batch-10000 glue-fusion
hybrid, the entire stress axis -- no arm touched the full arm. And whole
correctness surfaces the bare arm never opened (reduced precision, padding)
or shipped unverified are measured lanes in the harness arms: arm-bare's own
report admits its padding fallback "was never exercised by any measured run,"
which is precisely the class of silent gap the smoke gate and variant lanes
exist to close.

**3. The discipline reproduces discoveries.** arm-noskills, with harness
prompts but no skills and no campaign knowledge, independently found in 12
minutes the campaign's central measurement insight: the max-autotune compiled
baseline fails the official tolerance against its own reference (max_abs
5.3e-3), so "beat torch.compile" must be re-anchored on the admissible
configuration. A finding produced twice by the same discipline in isolated
sessions is a property of the discipline, not luck.

**4. What the skills specifically added** (full vs noskills, both on harness
discipline): the noskills arm parked compile-inside-candidate and never
reached the flash kernel, the tile probe, or the official grid in budget; the
full arm's profiling playbook (nsys under denied counters), the kernel
research routing, and the review loop are where its remaining 4x at seq-1024
and the #14 evidence came from. The wall-clock difference is real -- the full
arm also spent hours -- but the noskills arm's own "not attempted for budget"
list is exactly the full arm's differentiating output.
