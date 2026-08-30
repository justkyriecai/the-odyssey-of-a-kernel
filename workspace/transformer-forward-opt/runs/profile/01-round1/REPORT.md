# Round 1-2 profiles -- nsys kern_sum text exports

**Tool: nsys 2024.6.2 (CUPTI timeline; hardware counters denied on this pod).
Caveats that bound every number here:** each CSV is a whole-process trace --
the eager reference model's 26 forwards are in every file (~116 kernels each,
28 cutlass GEMMs per forward) -- and CUDA graph replays are traced in "graph"
mode, so kernels executed inside a captured graph are not re-counted per
replay. These exports therefore anchor *program structure*, not per-forward
kernel counts for graphed candidates.

Protocol per trace: `--accuracy-trials 1 --warmup 5 --repeats 20
--benchmark-rounds 1` (26 forwards per model slot).

| Trace | Distinct kernels | Instances | What it anchors |
|---|---:|---:|---|
| `opponent-ro_center` | 42 | 3161 | The admissible opponent (reduce-overhead baseline): Triton-fused pointwise + cuBLAS GEMMs + cudagraph trees |
| `compiled-safe-ro_center` | 47 | 3153 | Our compiled body: near-identical instance profile to the opponent (within 8 instances of 3161) -- structurally its equal, which the timing agrees with (0.362 vs 0.368 ms, +1.7%, at the noise line) |
| `flash-tf32_seq1024` | 43 | 2677 | The flash candidate at seq-1024: attention served by the `_flash_fwd` Triton kernel (named after its jit function, not "triton_"), the S x S materialization kernels absent |

The decisive comparisons in this round were timing rows, not counters: the
compiled-vs-opponent margin lives in `runs/benchmark.csv` (notes
`center margin restatement`, n=3 fresh processes, 0.3627/0.3615/0.3627), and
the regime anatomy that motivated everything is in
`../00-regime-anatomy/REPORT.md` (116 kernels/forward eager, ~5 compiled).
