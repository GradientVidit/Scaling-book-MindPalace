

# 0. MLP We Are Computing

Ignore attention and activation details.

An MLP is:

```text
X [B, D]
   │
   ▼
W_in [D, M]
   │
   ▼
H [B, M]
   │
   ▼
W_out [M, D]
   │
   ▼
Y [B, D]
```

Where:

- `B` = tokens/batch
    
- `D` = `embed_dim` / model dimension
    
- `M` = `mlp_dim`
    

Mathematically:

```text
H = X · W_in

Y = H · W_out
```

For TP=4, MaxText shards the **MLP dimension M across the 4 TPUs**. This is the key idea behind its TP formulation. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/optimization/sharding.md "maxtext/docs/guides/optimization/sharding.md at main · AI-Hypercomputer/maxtext · GitHub"))

---

# 1. Initial State

We have 4 TPUs:

```text
TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
            TP group
```

The input activation `X` is sharded along `D`:

```text
X = [B, D]

TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
X0           X1           X2           X3
[B,D/4]      [B,D/4]      [B,D/4]      [B,D/4]
```

So:

```text
X = [ X0 | X1 | X2 | X3 ]
       D dimension
```

Now the weights.

`W_in` is:

```text
W_in [D, M]
```

TP shards it along `M`:

```text
W_in = [ W_in0 | W_in1 | W_in2 | W_in3 ]

TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
W_in0        W_in1        W_in2        W_in3

[D,M/4]      [D,M/4]      [D,M/4]      [D,M/4]
```

And `W_out`:

```text
W_out [M,D]
```

is correspondingly sharded along its `M` dimension:

```text
TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
W_out0       W_out1       W_out2       W_out3

[M/4,D]      [M/4,D]      [M/4,D]      [M/4,D]
```

So the fundamental TP layout is:

```text
                 TPUs

X:       [ X0 | X1 | X2 | X3 ]       ← D sharded

W_in:    [ W0 | W1 | W2 | W3 ]       ← M sharded
              ↓
W_out:   [ W0 | W1 | W2 | W3 ]       ← M sharded
```

This corresponds to MaxText's:

```text
BE_x × EM_x
```

for the first MLP matmul after the required activation AllGather. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/optimization/sharding.md "maxtext/docs/guides/optimization/sharding.md at main · AI-Hypercomputer/maxtext · GitHub"))

---

# 2. FORWARD PASS — Prepare the Input

Each TPU currently has only:

```text
X0, X1, X2, X3
```

But `W_in0` has shape:

```text
[D, M/4]
```

and therefore requires the **entire D dimension** of `X`.

For example:

```text
TPU 0 needs:

X [B,D] · W_in0 [D,M/4]
```

But TPU 0 only has:

```text
X0 [B,D/4]
```

So the TP group performs an:

# ALL-GATHER

```text
X0 ─────┐
X1 ─────┼──→ ALL-GATHER ──→ X
X2 ─────┤
X3 ─────┘
```

After the AllGather:

```text
TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
X            X            X            X
[B,D]        [B,D]        [B,D]        [B,D]
```

Every TPU now temporarily has the complete activation.

**Important:**

The weights did **not** get gathered.

Only the activation was gathered.

That is the defining communication pattern of TP here. MaxText explicitly describes this as communicating activations rather than weights. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/optimization/sharding.md "maxtext/docs/guides/optimization/sharding.md at main · AI-Hypercomputer/maxtext · GitHub"))

---

# 3. FORWARD — First MLP Matmul

Now every TPU has:

```text
X [B,D]
```

but each TPU owns only `1/4` of `W_in`.

Therefore:

```text
TPU 0:

X  ×  W_in0
[B,D] [D,M/4]
       ↓
     H0 [B,M/4]
```

Likewise:

```text
TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
X × W0       X × W1       X × W2       X × W3
   ↓            ↓            ↓            ↓
 H0           H1           H2           H3
[B,M/4]      [B,M/4]      [B,M/4]      [B,M/4]
```

Together:

```text
H = [ H0 | H1 | H2 | H3 ]

       [B,M]
```

So the **intermediate MLP activation is naturally TP-sharded on M**.

No communication is required here.

Why?

Because each TPU owns a different slice of the output dimension `M`.

This is exactly the reason TP works so nicely with the MLP. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/optimization/sharding.md "maxtext/docs/guides/optimization/sharding.md at main · AI-Hypercomputer/maxtext · GitHub"))

---

# 4. Apply MLP Activation

Suppose:

```text
H = activation(X · W_in)
```

Because the activation function operates element-wise, each TPU can independently perform it:

```text
TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
H0           H1           H2           H3
 ↓            ↓            ↓            ↓
A0           A1           A2           A3
```

Still:

```text
A = [A0 | A1 | A2 | A3]
```

and each TPU owns:

```text
A_i [B,M/4]
```

No communication.

---

# 5. FORWARD — Second MLP Matmul

Now:

```text
W_out [M,D]
```

is sharded:

```text
W_out0 [M/4,D]
W_out1 [M/4,D]
W_out2 [M/4,D]
W_out3 [M/4,D]
```

Each TPU performs:

```text
TPU 0:

A0 [B,M/4] × W_out0 [M/4,D]
              ↓
          Y0 [B,D]
```

Similarly:

```text
TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
A0 × W0      A1 × W1      A2 × W2      A3 × W3
   ↓            ↓            ↓            ↓
Y0           Y1           Y2           Y3
[B,D]        [B,D]        [B,D]        [B,D]
```

But there is a problem.

Each `Yi` is only a **partial contribution** to the real output.

Because:

```text
Y = A · W_out
```

and we split the contracting `M` dimension:

```text
Y = A0W0 + A1W1 + A2W2 + A3W3
```

Therefore:

```text
Y0 = A0W0
Y1 = A1W1
Y2 = A2W2
Y3 = A3W3
```

and the real output is:

```text
Y = Y0 + Y1 + Y2 + Y3
```

---

# 6. REDUCE-SCATTER

We need to:

1. Sum the four partial results.
    
2. Return only `1/4` of the resulting `D` dimension to each TPU.
    

So MaxText uses:

# REDUCE-SCATTER

```text
Y0 ─────┐
Y1 ─────┤
Y2 ─────┼──→ REDUCE-SCATTER
Y3 ─────┘
```

Conceptually:

```text
                Full Y
          [ Y | Y | Y | Y ]
                │
          REDUCE-SCATTER
                │
       ┌────────┼────────┐
       ↓        ↓        ↓
      Y0       Y1       Y2       Y3
```

Final state:

```text
TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
Y0           Y1           Y2           Y3
[B,D/4]      [B,D/4]      [B,D/4]      [B,D/4]
```

Together:

```text
Y = [ Y0 | Y1 | Y2 | Y3 ]
```

So the output is again **D-sharded**.

MaxText explicitly uses ReduceScatter here rather than an AllReduce. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/optimization/sharding.md "maxtext/docs/guides/optimization/sharding.md at main · AI-Hypercomputer/maxtext · GitHub"))

---

# 7. COMPLETE FORWARD PASS

The entire MLP becomes:

```text
                 INPUT
              X [B,D] sharded
                    │
                    │
              ALL-GATHER
                    │
                    ▼
             X [B,D] replicated
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
      X·W0        X·W1        X·W2 ... X·W3
        │           │
        ▼           ▼
       H0          H1 ... H3
        │
      activation
        │
        ▼
       A0          A1 ... A3
        │           │
        ▼           ▼
     A0·Wout0    A1·Wout1 ... A3·Wout3
        │           │
        └───────────┼───────────┘
                    │
              REDUCE-SCATTER
                    │
                    ▼
              Y [B,D] sharded
```

Or the entire thing in one line:

```text
D-sharded X
    ↓
ALL-GATHER
    ↓
X replicated
    ↓
W_in M-sharded
    ↓
local GEMM
    ↓
M-sharded intermediate
    ↓
W_out M-sharded
    ↓
partial D-sized outputs
    ↓
REDUCE-SCATTER
    ↓
D-sharded output
```

This is the central TP pattern in MaxText. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/optimization/sharding.md "maxtext/docs/guides/optimization/sharding.md at main · AI-Hypercomputer/maxtext · GitHub"))

---

# 8. Why This Is Efficient

The important trick is that **the two MLP matrices complement each other**.

We have:

```text
              W_in                  W_out

             [D,M]                  [M,D]
               │                      │
               │ shard M              │ shard M
               ▼                      ▼
           [D,M/4]                 [M/4,D]
```

This means:

### First matmul

```text
X [B,D] × W_in_i [D,M/4]
              ↓
          H_i [B,M/4]
```

The output is naturally sharded.

### Second matmul

```text
H_i [B,M/4] × W_out_i [M/4,D]
                         ↓
                    partial Y [B,D]
```

The partial results need to be summed.

Therefore only two activation collectives are needed:

```text
       ALL-GATHER
            ↓
        first GEMM
            ↓
        second GEMM
            ↓
     REDUCE-SCATTER
```

The Scaling Book describes exactly this advantage: instead of an AllReduce after each matrix, TP can use an AllGather before the first matmul and a ReduceScatter after the second. ([Jax ML](https://jax-ml.github.io/scaling-book/training/ "How to Parallelize a Transformer for Training | How To Scale Your Model"))

---

# 9. BACKWARD PASS

Now suppose backward reaches this MLP.

Forward produced:

```text
X → W_in → A → W_out → Y
```

We receive:

```text
dY
```

from the next layer.

Because the forward output was D-sharded:

```text
TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
dY0          dY1          dY2          dY3
[B,D/4]      [B,D/4]      [B,D/4]      [B,D/4]
```

---

# 10. BACKWARD — All-Gather dY

To calculate the gradient of `W_out`, each TPU needs the complete `dY`.

Therefore:

```text
dY0 ─────┐
dY1 ─────┤
dY2 ─────┼──→ ALL-GATHER
dY3 ─────┘
```

Afterward:

```text
TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
dY           dY           dY           dY
[B,D]        [B,D]        [B,D]        [B,D]
```

This is the backward counterpart of the forward AllGather.

The Scaling Book explicitly gives this step as:

```text
dOut[B,D] = AllGather(dOut[B,D_Y])
```

([Jax ML](https://jax-ml.github.io/scaling-book/training/ "How to Parallelize a Transformer for Training | How To Scale Your Model"))

---

# 11. BACKWARD — Gradient of W_out

Each TPU owns a different `W_out` shard:

```text
W_out0 [M/4,D]
W_out1 [M/4,D]
...
```

The gradient is:

```text
dW_out = Aᵀ · dY
```

Therefore:

```text
TPU 0:

A0ᵀ × dY
 ↓
dW_out0
[M/4,D]
```

Likewise:

```text
TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
A0ᵀ·dY       A1ᵀ·dY       A2ᵀ·dY       A3ᵀ·dY
   ↓            ↓            ↓            ↓
dWout0       dWout1       dWout2       dWout3
```

**No gradient communication is required here.**

Why?

Because each TPU owns a different piece of `W_out`.

There is no overlap in parameter ownership.

So each TPU can independently calculate the gradient for its own parameter shard.

---

# 12. BACKWARD — Gradient Through the Second Matmul

We have:

```text
Y = A · W_out
```

Therefore:

```text
dA = dY · W_outᵀ
```

Each TPU has:

```text
dY [B,D]
W_out_i [M/4,D]
```

so it calculates:

```text
TPU 0:

dY × W_out0ᵀ
        ↓
dA0 [B,M/4]
```

and similarly:

```text
TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
dA0          dA1          dA2          dA3
[B,M/4]      [B,M/4]      [B,M/4]      [B,M/4]
```

Again:

**No communication.**

The gradient is naturally M-sharded because `W_out` is M-sharded.

---

# 13. BACKWARD — Through Activation

The activation is element-wise.

So:

```text
dA0 → dH0
dA1 → dH1
dA2 → dH2
dA3 → dH3
```

Still:

```text
dH = [dH0 | dH1 | dH2 | dH3]
```

with each TPU owning:

```text
[B,M/4]
```

No communication.

---

# 14. BACKWARD — Gradient of W_in

We have:

```text
H = X · W_in
```

Therefore:

```text
dW_in = Xᵀ · dH
```

But `X` was originally D-sharded.

Each TPU needs the full `X`.

So we need:

```text
X0 ─────┐
X1 ─────┤
X2 ─────┼──→ ALL-GATHER
X3 ─────┘
```

giving:

```text
X [B,D]
```

on every TPU.

Then:

```text
TPU 0:

Xᵀ × dH0
     ↓
dW_in0 [D,M/4]
```

and similarly:

```text
TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
dWin0        dWin1        dWin2        dWin3
[D,M/4]      [D,M/4]      [D,M/4]      [D,M/4]
```

Again, **no gradient reduction is necessary**, because each TPU owns a distinct M shard.

The Scaling Book gives this same sequence: AllGather `X`, then each TPU computes its local `dW_in`. ([Jax ML](https://jax-ml.github.io/scaling-book/training/ "How to Parallelize a Transformer for Training | How To Scale Your Model"))

---

# 15. BACKWARD — Gradient of Input

Finally:

```text
dX = dH · W_inᵀ
```

Each TPU has:

```text
dH_i [B,M/4]
W_in_i [D,M/4]
```

so:

```text
TPU 0:

dH0 × W_in0ᵀ
       ↓
partial dX0 [B,D]
```

Each TPU produces a **partial** gradient for the complete D dimension:

```text
TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
p0           p1           p2           p3
[B,D]        [B,D]        [B,D]        [B,D]
```

The real gradient is:

```text
dX = p0 + p1 + p2 + p3
```

But we want the gradient returned to the previous layer in the same D-sharded layout:

```text
dX = [dX0 | dX1 | dX2 | dX3]
```

Therefore:

# REDUCE-SCATTER

```text
p0 ─────┐
p1 ─────┤
p2 ─────┼──→ REDUCE-SCATTER
p3 ─────┘
```

Result:

```text
TPU 0        TPU 1        TPU 2        TPU 3
------       ------       ------       ------
dX0          dX1          dX2          dX3
[B,D/4]      [B,D/4]      [B,D/4]      [B,D/4]
```

This is now exactly the input layout expected by the previous layer.

---

# 16. COMPLETE BACKWARD PASS
(Assuming Activation is saved in forward pass) (Or else you would need another allgather in backward pass for recomputation of activation)

```text
                 dY
                  │
             D-sharded
                  │
             ALL-GATHER
                  │
                  ▼
             dY [B,D]
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
      dWout0    dWout1    ... dWout3
        │
        ▼
        dH [B,M/4]
        │
   activation'
        │
        ▼
        dH
        │
        ├──────────────→ dW_in
        │
        ▼
   dH · W_inᵀ
        │
        ▼
   partial dX
        │
   REDUCE-SCATTER
        │
        ▼
   dX [B,D/4]
```

---

# 17. The Full TP Training Step

Putting everything together:

```text
                    FORWARD
                       │
        X [B,D/4] on each TPU
                       │
                 ALL-GATHER
                       │
                 X [B,D]
                       │
              ┌────────┴────────┐
              │                 │
           W_in0              W_in1 ...
              │                 │
              ▼                 ▼
             H0                H1
              │                 │
          activation        activation
              │                 │
              ▼                 ▼
             A0                A1
              │                 │
           Wout0              Wout1
              │                 │
              ▼                 ▼
          partial Y         partial Y
              │                 │
              └────────┬────────┘
                       │
                 REDUCE-SCATTER
                       │
                       ▼
                 Y [B,D/4]


                    BACKWARD
                       │
                      dY
                       │
                 ALL-GATHER
                       │
                    dY [B,D]
                       │
          ┌────────────┴────────────┐
          │                         │
       dW_out0                   dW_out1 ...
          │                         │
          ▼                         ▼
         dA0                       dA1
          │                         │
      activation'              activation'
          │                         │
          ▼                         ▼
         dH0                       dH1
          │                         │
          ├──→ dW_in0              ├──→ dW_in1
          │                         │
          ▼                         ▼
       partial dX                partial dX
          │                         │
          └────────────┬────────────┘
                       │
                 REDUCE-SCATTER
                       │
                       ▼
                 dX [B,D/4]
```

---

# 18. The Most Important Mental Model

For **MaxText 1D TP**, remember this:

```text
                 M dimension
                    ↓
              SHARD WEIGHTS
                    │
                    │
X: D-sharded ──→ ALL-GATHER
                    │
                    ▼
              First GEMM
                    │
                    ▼
              M-sharded H
                    │
              Second GEMM
                    │
                    ▼
             partial D output
                    │
             REDUCE-SCATTER
                    │
                    ▼
              D-sharded Y
```

So:

> **TP shards the MLP's feature dimensions, not the batch.**

And communication is fundamentally:

```text
FORWARD:

D-sharded input
      ↓
 ALL-GATHER
      ↓
M-sharded computation
      ↓
REDUCE-SCATTER
      ↓
D-sharded output
```

and backward is essentially the transpose:

```text
BACKWARD:

D-sharded dY
      ↓
 ALL-GATHER
      ↓
M-sharded gradient computation
      ↓
REDUCE-SCATTER
      ↓
D-sharded dX
```

MaxText explicitly describes TP as sharding activations along feature dimensions and using an AllGather and ReduceScatter around the feed-forward computation. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/optimization/sharding.md "maxtext/docs/guides/optimization/sharding.md at main · AI-Hypercomputer/maxtext · GitHub"))

---

## 19. One Crucial TPU Insight

The reason TP cannot simply be scaled to arbitrarily many TPUs is now intuitive.

Each TPU does only:

```text
1/TP
```

of the MLP computation, **but the activation communication does not shrink in the same way**.

For the MaxText TP pattern:

```text
Compute ∝ M / TP

Communication ∝ B × D
```

so the operational arithmetic intensity is:

```text
AI ≈ M / TP
```

MaxText derives exactly this result. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/optimization/sharding.md "maxtext/docs/guides/optimization/sharding.md at main · AI-Hypercomputer/maxtext · GitHub"))

Therefore:

```text
TP ↑
  ↓
compute per TPU ↓
  ↓
communication becomes increasingly dominant
```

This is why TP is particularly attractive when `M` is very large and/or the per-device batch is small, and why TPU TP is normally kept within an efficient ICI topology rather than casually extending it across slower DCN links. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/optimization/custom_model.md?utm_source=chatgpt.com "maxtext/docs/guides/optimization/custom_model.md at main · AI-Hypercomputer/maxtext · GitHub"))

### The one-line intuition

> **FSDP makes the weights move; TP makes the activations move.**

For this MLP specifically:

```text
TP=4

                 TPU 0   TPU 1   TPU 2   TPU 3
                   │       │       │       │
Input D             D/4     D/4     D/4     D/4
                   │       │       │       │
                ─────── ALL-GATHER ───────
                   │       │       │       │
                 full D  full D  full D  full D
                   │       │       │       │
W_in M             M/4     M/4     M/4     M/4
                   │       │       │       │
                 local GEMMs
                   │       │       │       │
Intermediate M     M/4     M/4     M/4     M/4
                   │       │       │       │
                 local GEMMs
                   │       │       │       │
                 partial D partial D partial D partial D
                   │       │       │       │
                ──── REDUCE-SCATTER ───────
                   │       │       │       │
Output D           D/4     D/4     D/4     D/4
```

