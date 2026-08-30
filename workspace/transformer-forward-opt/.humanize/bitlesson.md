# BitLesson -- project lessons (specific problem -> specific solution)

- BL-1: In-process multi-case sweeps with torch.compile silently degrade to
  eager after ~8 compiles (torch._dynamo.config.cache_size_limit per code
  object). Solution: one verify.py --case per process for anything compiled,
  or raise the limit in the candidate factory and verify kern_sum.
- BL-2: A compiled reference is not a correctness reference: TF32 Triton GEMM
  templates drift ~5e-3, Inductor reduced-precision softmax fusion drifts
  0.0625 (bf16) at every mode. Judge correctness only against the uncompiled
  eager baseline; use --benchmark-on-failure for compiled-denominator timing.
- BL-3: torch tensor CPU work on tiny shapes: OMP threads = cores (192) makes
  barriers dominate (1794ms vs 0.47ms single-thread). Pin OMP/MKL threads in
  CPU gates.
- BL-4: Everything outside a compiled/captured region is paid per call; move
  mask and constant construction inside the graph and keep masks in decomposed
  broadcast form rather than materialized combinations.
- BL-5: A fallback that returns into the same code path is a recursion, not a
  fallback. Degrade at resolution time (validate names against the live
  registry when choosing), never at serve time.
- BL-6: reshape on a non-contiguous view copies silently -- a kernel that
  already takes strides should take them in the tensor's native rank. The
  flash wrapper's 3D reshape was three hidden q/k/v copies per layer.
