TPU v6e specs: C = 9.18e14 FLOPs/s (bf16), 2D torus (2 mesh axes)
	Total bidirectional ICI bandwidth/chip ≈ 8e11 bytes/s → per-axis W_ici ≈ 4e11 bytes/s

Critical per-chip batch size (1 axis):
  C / W_ici = 9.18e14 / 4e11 ≈ 2,295 tokens/chip

Using both torus axes for DP (M=2):
  2,295 / 2 ≈ 1,148 tokens/chip

Compare: v5p threshold was 2,550/axis, 850 with 3 axes.
v6e needs a SMALLER per-chip batch to stay compute-bound — its bandwidth
grew faster (relative to compute) than v5p's did, and it only has 2 torus
axes instead of 3.

---

There are TWO unrelated things both called "critical batch size" —
don't conflate them.

1. HARDWARE critical batch size (this section's topic):
   The per-chip batch size below which you become communication-bound
   instead of compute-bound. Purely a function of chip FLOPs/s vs
   network bandwidth (C/W). Nothing to do with training quality.

2. STATISTICAL critical batch size (gradient noise scale — McCandlish
   et al.):
   The batch size above which adding more data per step stops helping
   training efficiency, because gradients are already low-noise.
   Purely a function of the optimization problem, not the hardware.

Why it matters: real system design starts from #2 (scaling laws pick a
total batch size), THEN searches for a sharding scheme where #1 is
satisfied at that fixed batch size. They're solved in that order, not
simultaneously — and it's a coincidence when their numbers look similar.