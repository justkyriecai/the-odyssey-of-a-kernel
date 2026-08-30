# Round 2 Summary

Scope: the round-1 review's 7-item CONTINUE list, executed in full, plus the
opponent-as-candidate move that closed the last performance gaps.

1. P0 dispatch recursion: fixed at resolution time; regression test green.
2. Variant lanes: 31 rows (13 padded geometries incl. base-ro's 5, bf16/fp16
   spread), all PASS; recalibrated; padded-and-dense guarantee now measured.
3. SDPA pinned (mem-efficient) inside the compiled region; re-measured
   byte-identical (batch-10000 3.12x, max_abs unchanged to the last digit).
   compiled-sdpa's claimed win set corrected (heads-16 belongs to base-ro).
4. Fidelity table filled into docs/precision-budget.md (worst row per
   candidate/dtype; compile-baseline rows excluded as in calibration).
   round-1-vs-round-3 clause: satisfied by recorded p90/median tightness;
   decision logged in the tracker.
5. center restated: 0.362 ms, n=3 fresh processes, spread 0.0012; +1.7% vs
   the opponent's 0.368 -- at the noise line, stated as parity-to-slight-edge.
6. Stress ceiling evidenced: S=3072 PASS 7.96x through the official script;
   S=4096 (~68 GB reference scores) remains arithmetic. 01-round1/REPORT.md
   written naming nsys and its whole-process/graph-mode caveats; opponent
   kern_sum anchor added (3161 vs our 3153 instances -- structural parity).
7. Bookkeeping: tracker stale lines fixed, entry counts corrected, off-script
   labeling caveat noted in REPORT and nodes.

Final state: dispatch serves every official geometry and measured variant lane;
worst case 1.232x over eager, never behind the admissible opponent anywhere it
was measured; shape #14 evidenced off-script (25.5 s, 0 bad of 3.28e9);
bf16/fp16 on graph-safe at zero budget spend, except heads-16-bf16 which falls
back by the margin rule.

## BitLesson Delta
Action: add -- BL-5: "a fallback that returns to the same code path is a
recursion, not a fallback; degrade at resolution time, never at serve time."
