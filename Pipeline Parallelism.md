
# 0. Model

Ignore attention details and imagine a Transformer with **8 decoder layers**:

```text
Input
  │
  ▼
L0
L1
L2
L3
L4
L5
L6
L7
  │
  ▼
Output
```

We have **4 TPUs** and choose:

```text
PP = 4
```

For simplicity:

```text
TPU 0 → layers 0,1
TPU 1 → layers 2,3
TPU 2 → layers 4,5
TPU 3 → layers 6,7
```

So:

```text
              PIPELINE STAGES

TPU 0          TPU 1          TPU 2          TPU 3
------         ------         ------         ------
L0             L2             L4             L6
L1             L3             L5             L7
```

This is exactly what MaxText means by sharding weights/computation **by layers**. A stage can contain multiple layers. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/optimization/sharding.md?utm_source=chatgpt.com "maxtext/docs/guides/optimization/sharding.md at main · AI-Hypercomputer/maxtext · GitHub"))

---

# 1. Initial State

Each TPU owns only the parameters for its layers:

```text
TPU 0             TPU 1             TPU 2             TPU 3
------             ------             ------             ------
W0, W1             W2, W3             W4, W5             W6, W7
```

Conceptually:

```text
FULL MODEL

W = [ W0 W1 | W2 W3 | W4 W5 | W6 W7 ]
          │        │        │
          ▼        ▼        ▼
        TPU 0    TPU 1    TPU 2    TPU 3
```

Unlike TP, **the model itself is divided along the layer dimension**.

There is no reason for TPU 0 to know `W4`, for example.

It only needs:

```text
W0, W1
```

to execute its part of the model.

---

# 2. Why Microbatches Are Needed

Suppose we simply send one batch through the pipeline:

```text
TPU 0 → TPU 1 → TPU 2 → TPU 3
```

Then:

```text
time →

TPU 0: [compute] [idle] [idle] [idle]
TPU 1:           [compute] [idle] [idle]
TPU 2:                     [compute] [idle]
TPU 3:                               [compute]
```

Most of the hardware is idle.

This is the:

# Pipeline Bubble

The solution is to split the global batch into **microbatches**.

Suppose:

```text
global batch = [MB0, MB1, MB2, MB3, MB4, MB5, MB6, MB7]
```

Each microbatch travels through the same pipeline.

MaxText's `num_pipeline_microbatches` controls this. The microbatch count must be a multiple of the number of pipeline stages. More microbatches reduce the bubble, but make each microbatch's computation smaller. ([MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.3/reference/core_concepts/batch_size.html?utm_source=chatgpt.com "Batch Size — MaxText documentation"))

---

# 3. Forward Pass — Microbatch 0

Start with:

```text
MB0
```

TPU 0 receives it.

It executes its layers:

```text
MB0
 │
 ▼
L0
 │
 ▼
L1
 │
 ▼
activation A1
```

So:

```text
TPU 0:

MB0
 ↓
L0 → L1
 ↓
A0
```

Then TPU 0 **sends the activation to TPU 1**:

```text
TPU 0                         TPU 1

A0 ─────────────────────────→
```

This is the fundamental PP communication:

> **Send the activation produced by one stage to the next stage.**

The Scaling Book describes PP exactly this way: execute the first stage, copy its resulting activations to the next device, and continue. ([Jax ML](https://jax-ml.github.io/scaling-book/training/?utm_source=chatgpt.com "How to Parallelize a Transformer for Training | How To Scale Your Model"))

---

# 4. TPU 1 Continues MB0

TPU 1 receives:

```text
A0
```

and runs:

```text
A0
 │
 ▼
L2
 │
 ▼
L3
 │
 ▼
A1
```

Then:

```text
A1 ─────────────────────────→ TPU 2
```

---

# 5. TPU 2

TPU 2:

```text
A1
 │
 ▼
L4
 │
 ▼
L5
 │
 ▼
A2
```

Then:

```text
A2 ─────────────────────────→ TPU 3
```

---

# 6. TPU 3

TPU 3:

```text
A2
 │
 ▼
L6
 │
 ▼
L7
 │
 ▼
Output(MB0)
```

So the complete forward path for one microbatch is:

```text
MB0
 │
 ▼
┌──────────────┐
│ TPU 0        │
│ L0 → L1      │
└──────┬───────┘
       │ A0
       ▼
┌──────────────┐
│ TPU 1        │
│ L2 → L3      │
└──────┬───────┘
       │ A1
       ▼
┌──────────────┐
│ TPU 2        │
│ L4 → L5      │
└──────┬───────┘
       │ A2
       ▼
┌──────────────┐
│ TPU 3        │
│ L6 → L7      │
└──────┬───────┘
       │
       ▼
    Output
```

---

# 7. Now Send MB1

The whole point of microbatching is that **TPU 0 doesn't wait for MB0 to finish the entire model**.

Once TPU 0 has sent MB0 to TPU 1:

```text
TPU 0
  │
  ├── MB0 → TPU 1
  │
  └── starts MB1
```

So:

```text
TPU 0:  MB0 → MB1 → MB2 → MB3 → ...
```

Meanwhile TPU 1 is processing MB0.

---

# 8. Pipeline Filling

Now the pipeline starts looking like this:

```text
Time →

          t0    t1    t2    t3    t4    t5    t6    t7

TPU 0     MB0   MB1   MB2   MB3   MB4   MB5   MB6   MB7
           ↓
TPU 1           MB0   MB1   MB2   MB3   MB4   MB5   MB6
                 ↓
TPU 2                 MB0   MB1   MB2   MB3   MB4   MB5
                       ↓
TPU 3                       MB0   MB1   MB2   MB3   MB4
```

Now multiple TPUs are doing useful work simultaneously.

This is the fundamental reason for microbatching.

---

# 9. Steady State

Once the pipeline is full:

```text
Time →

TPU 0:  MB4   MB5   MB6   MB7
TPU 1:  MB3   MB4   MB5   MB6
TPU 2:  MB2   MB3   MB4   MB5
TPU 3:  MB1   MB2   MB3   MB4
```

Every TPU is computing.

The activations continuously flow:

```text
TPU 0
  │
  │ activation
  ▼
TPU 1
  │
  │ activation
  ▼
TPU 2
  │
  │ activation
  ▼
TPU 3
```

This is literally a **pipeline**.

---

# 10. Forward Pass Complete

Eventually the last microbatch reaches TPU 3:

```text
MB7
 │
 ▼
TPU 0
 │
 ▼
TPU 1
 │
 ▼
TPU 2
 │
 ▼
TPU 3
 │
 ▼
loss
```

Now forward is finished.

But we still need backward.

---

# 11. BACKWARD PASS

The loss gives us:

```text
dL/dOutput
```

for the final microbatch.

Backward starts at the **last stage**.

So TPU 3 begins:

```text
dOutput
   │
   ▼
L7 backward
   │
   ▼
L6 backward
   │
   ▼
dA2
```

Then TPU 3 sends:

```text
dA2 ─────────────────────────→ TPU 2
```

Notice the symmetry:

### Forward

```text
activation
TPU 0 → TPU 1 → TPU 2 → TPU 3
```

### Backward

```text
gradient
TPU 3 → TPU 2 → TPU 1 → TPU 0
```

---

# 12. TPU 2 Backward

TPU 2 receives:

```text
dA2
```

and computes:

```text
dA2
 │
 ▼
L5 backward
 │
 ▼
L4 backward
 │
 ▼
dA1
```

Then:

```text
dA1 ─────────────────────────→ TPU 1
```

---

# 13. TPU 1 Backward

```text
dA1
 │
 ▼
L3 backward
 │
 ▼
L2 backward
 │
 ▼
dA0
```

Then:

```text
dA0 ─────────────────────────→ TPU 0
```

---

# 14. TPU 0 Backward

Finally:

```text
dA0
 │
 ▼
L1 backward
 │
 ▼
L0 backward
 │
 ▼
dInput
```

So one microbatch's backward pass is:

```text
TPU 3
  │
  │ gradient
  ▼
TPU 2
  │
  │ gradient
  ▼
TPU 1
  │
  │ gradient
  ▼
TPU 0
```

---

# 15. But Where Do Weight Gradients Come From?

Each stage owns its own weights.

For example TPU 2 owns:

```text
W4
W5
```

During backward for a microbatch it computes:

```text
dW4
dW5
```

locally.

There is **no need to communicate these gradients to another pipeline stage**, because no other stage owns those parameters.

So:

```text
TPU 0 → dW0, dW1
TPU 1 → dW2, dW3
TPU 2 → dW4, dW5
TPU 3 → dW6, dW7
```

This is one of the most important properties of PP.

---

# 16. What About the Microbatches?

This is where the real pipeline schedule matters.

We don't want:

```text
FORWARD ALL MICROBATCHES
        ↓
BACKWARD ALL MICROBATCHES
```

because then the stages spend a lot of time idle.

Instead, gPipe schedules forward microbatches through the stages and then performs their backward computations in a pipelined schedule.

Conceptually:

```text
TIME →

TPU 0    F0   F1   F2   F3   F4   F5   B5   B4   B3   B2   B1   B0

TPU 1         F0   F1   F2   F3   F4   B5   B4   B3   B2   B1   B0

TPU 2              F0   F1   F2   F3   B5   B4   B3   B2   B1   B0

TPU 3                   F0   F1   F2   B5   B4   B3   B2   B1   B0
```

`Fi` = forward of microbatch `i`

`Bi` = backward of microbatch `i`

The exact schedule/implementation is more complicated than this simplified picture, but this is the intuition.

---

# 17. Pipeline Bubble

There is unavoidable idle time at the beginning and end.

For 4 stages:

```text
START

TPU 0    F0
TPU 1         F0
TPU 2              F0
TPU 3                   F0

                         ← bubble
```

And when the pipeline drains:

```text
                         ← bubble

TPU 0                        B0
TPU 1                   B0
TPU 2              B0
TPU 3         B0
```

The bubble comes from the fact that information must physically travel through the stages.

MaxText gives the bubble fraction as:

```text
(num_stages - 1)
────────────────────────────────────────────
(num_pipeline_repeats × num_pipeline_microbatches
 + num_stages - 1)
```

for its pipeline configuration. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

Therefore:

```text
more microbatches
       ↓
smaller bubble
       ↓
better utilization
```

But:

```text
more microbatches
       ↓
smaller microbatch
       ↓
smaller GEMMs
       ↓
potentially worse compute efficiency
```

So there is a tradeoff. ([MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.3/reference/core_concepts/batch_size.html?utm_source=chatgpt.com "Batch Size — MaxText documentation"))

---

# 18. Complete PP Forward + Backward

Now put it together.

```text
                    FORWARD

             microbatch flows →
                     
TPU 0       L0 → L1
              │
              │ activation
              ▼
TPU 1       L2 → L3
              │
              │ activation
              ▼
TPU 2       L4 → L5
              │
              │ activation
              ▼
TPU 3       L6 → L7
              │
              ▼
             LOSS


                    BACKWARD

TPU 3       L7 ← L6
              │
              │ gradient
              ▼
TPU 2       L5 ← L4
              │
              │ gradient
              ▼
TPU 1       L3 ← L2
              │
              │ gradient
              ▼
TPU 0       L1 ← L0
```

The communication is therefore:

```text
FORWARD:

TPU 0 ──activation──→ TPU 1
TPU 1 ──activation──→ TPU 2
TPU 2 ──activation──→ TPU 3


BACKWARD:

TPU 3 ──gradient────→ TPU 2
TPU 2 ──gradient────→ TPU 1
TPU 1 ──gradient────→ TPU 0
```

---

# 19. What Is Actually Communicated?

This is the beautiful part of PP.

Suppose the activation at a stage boundary is:

```text
A [B_micro, D]
```

Only this activation needs to move:

```text
TPU 0 → TPU 1
```

You **do not move the model weights**.

You **do not AllGather the entire model**.

You **do not ReduceScatter the entire model**.

You simply transfer:

```text
activation
```

forward and:

```text
activation gradient
```

backward.

The Scaling Book describes PP as having **minimal communication: moving activations between neighboring stages**. ([Jax ML](https://jax-ml.github.io/scaling-book/training/?utm_source=chatgpt.com "How to Parallelize a Transformer for Training | How To Scale Your Model"))

---

# 20. Why PP Has Such Low Communication

Suppose:

```text
model = 100B parameters
```

The model is split:

```text
TPU 0 → 25B
TPU 1 → 25B
TPU 2 → 25B
TPU 3 → 25B
```

But the communication between stages isn't:

```text
25B parameters
```

It is roughly:

```text
microbatch × hidden_dimension
```

for the boundary activation.

So PP can be extremely communication-efficient.

This is why the Scaling Book says PP can be attractive even across lower-bandwidth interconnects. ([Jax ML](https://jax-ml.github.io/scaling-book/training/?utm_source=chatgpt.com "How to Parallelize a Transformer for Training | How To Scale Your Model"))

---

# 21. The Price: Pipeline Bubble

The tradeoff is:

```text
low communication
      +
model split by layers
      ↓
pipeline bubble
```

Imagine only one microbatch:

```text
Time →

TPU 0   ███
TPU 1      ███
TPU 2         ███
TPU 3            ███
```

Three quarters of the hardware are idle much of the time.

With many microbatches:

```text
Time →

TPU 0   ███ ███ ███ ███ ███
TPU 1      ███ ███ ███ ███ ███
TPU 2         ███ ███ ███ ███ ███
TPU 3            ███ ███ ███ ███
```

The pipeline becomes much more efficient.

---

# 22. What Does `num_layers_per_pipeline_stage` Mean?

This MaxText parameter is important.

Suppose:

```text
num_pipeline_stages = 4
num_layers_per_pipeline_stage = 2
```

Then:

```text
TPU 0 → 2 layers
TPU 1 → 2 layers
TPU 2 → 2 layers
TPU 3 → 2 layers
```

For 8 decoder layers:

```text
TPU 0: L0 L1
TPU 1: L2 L3
TPU 2: L4 L5
TPU 3: L6 L7
```

More generally, MaxText defines the decoder layer count as:

```text
num_decoder_layers
=
num_stages
× num_layers_per_pipeline_stage
× num_pipeline_repeats
```

in its pipeline configuration. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

---

# 23. `num_pipeline_microbatches`

This is different from pipeline stages.

For example:

```text
PP = 4

num_pipeline_microbatches = 8
```

means:

```text
global batch

┌──────────────────────────────────────┐
│                                      │
│             GLOBAL BATCH             │
│                                      │
└──────────────────────────────────────┘
       ↓       ↓       ↓       ↓
      MB0     MB1     MB2     MB3 ...
```

Eight pieces travel through the four-stage pipeline.

MaxText says the microbatch size is:

```text
microbatch_size
=
global_batch_size / num_pipeline_microbatches
```

and the number of microbatches must be a multiple of the total pipeline-stage count. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

---

# 24. MaxText's Physical Mesh

In MaxText, PP is represented as a physical mesh axis:

```text
stage
```

and configured through:

```text
ici_pipeline_parallelism
dcn_pipeline_parallelism
```

For our simple 4-TPU example:

```text
ici_pipeline_parallelism = 4
dcn_pipeline_parallelism = 1
```

means:

```text
4 pipeline stages
all within the ICI-connected domain
```

MaxText's mesh system explicitly has a `stage` physical axis alongside `data`, `fsdp`, `tensor`, etc. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

---

# 25. Important TPU Detail

For your 4-TPU example, you ideally want:

```text
TPU 0 ──ICI── TPU 1 ──ICI── TPU 2 ──ICI── TPU 3
```

because pipeline communication happens between stages.

On a TPU slice, ICI is the fast accelerator-to-accelerator network.

MaxText's hierarchical mesh distinguishes:

```text
ICI = fast, within slice
DCN = slower, across slices
```

and allows PP to be placed on those mesh axes. ([MaxText](https://maxtext.readthedocs.io/en/latest/guides/optimization/sharding.html "Sharding on TPUs — MaxText documentation"))

---

# 26. The Fundamental PP Mental Model

Forget all implementation details.

Think:

```text
                 MODEL

      L0 L1 | L2 L3 | L4 L5 | L6 L7
        │       │       │       │
        ▼       ▼       ▼       ▼
      TPU0   TPU1    TPU2    TPU3
        │       │       │
        └─ A ─→└─ A ─→└─ A ─→
```

The model is **cut vertically**:

```text
              LAYER DIMENSION

       ┌────────┬────────┬────────┬────────┐
       │        │        │        │        │
       │ Stage0 │ Stage1 │ Stage2 │ Stage3 │
       │        │        │        │        │
       └────────┴────────┴────────┴────────┘
```

Then data flows horizontally through those stages:

```text
Microbatch:

        ───────────────→

TPU0       TPU1       TPU2       TPU3
 │          │          │          │
 ▼          ▼          ▼          ▼
L0,L1      L2,L3      L4,L5      L6,L7
```

---

# 27. PP vs TP in One Picture

Since you're building the mental model of each technique separately, this distinction is worth one line:

```text
TP:

              SAME LAYERS
      ┌────────┬────────┬────────┬────────┐
      │ TPU0   │ TPU1   │ TPU2   │ TPU3   │
      │ shard  │ shard  │ shard  │ shard  │
      └────────┴────────┴────────┴────────┘


PP:

      DIFFERENT LAYERS
      ┌────────┬────────┬────────┬────────┐
      │ TPU0   │ TPU1   │ TPU2   │ TPU3   │
      │ L0-L1  │ L2-L3  │ L4-L5  │ L6-L7  │
      └────────┴────────┴────────┴────────┘
```

For PP:

> **Each TPU owns a contiguous portion of the model's depth.**

---

# 28. One-Line Forward/Backward Mental Model

### Forward

```text
SHARD MODEL BY LAYERS
        ↓
SPLIT BATCH INTO MICROBATCHES
        ↓
MICROBATCH 0 → STAGE 0
        ↓
ACTIVATION → STAGE 1
        ↓
ACTIVATION → STAGE 2
        ↓
ACTIVATION → STAGE 3
        ↓
LOSS
```

### Backward

```text
LOSS
 ↓
STAGE 3 BACKWARD
 ↓
GRADIENT → STAGE 2
 ↓
STAGE 2 BACKWARD
 ↓
GRADIENT → STAGE 1
 ↓
STAGE 1 BACKWARD
 ↓
GRADIENT → STAGE 0
 ↓
STAGE 0 BACKWARD
```

### Entire concept

```text
             PIPELINE PARALLELISM

              SHARD BY LAYERS
                     ↓
            ┌─────────────────┐
            │   microbatch    │
            └────────┬────────┘
                     ↓
                ┌─────────┐
                │  TPU 0  │
                │  L0-L1  │
                └────┬────┘
                     │ activation
                     ↓
                ┌─────────┐
                │  TPU 1  │
                │  L2-L3  │
                └────┬────┘
                     │ activation
                     ↓
                ┌─────────┐
                │  TPU 2  │
                │  L4-L5  │
                └────┬────┘
                     │ activation
                     ↓
                ┌─────────┐
                │  TPU 3  │
                │  L6-L7  │
                └─────────┘
                     │
                     ▼
                    LOSS

                 BACKWARD
                     ↑
              gradients flow
              in reverse
```

**The core intuition:**

> **TP divides the work inside a layer; PP divides the model itself into stages.**

And for PP specifically, the central performance tradeoff is:

```text
                 PP
                  │
       ┌──────────┴──────────┐
       ↓                     ↓
very little             pipeline
communication             bubble
       │                     │
       ↓                     ↓
good for                 need enough
large models             microbatches
```

MaxText's documentation makes the same fundamental point: PP shards computation by layers, communicates relatively small layer activations between stages, and uses microbatches to mitigate the pipeline bubble. ([MaxText](https://maxtext.readthedocs.io/en/latest/guides/optimization/sharding.html "Sharding on TPUs — MaxText documentation"))

### One MaxText-specific caveat

The explanation above is the **gPipe mental model**. Current MaxText also supports **circular pipelines**, where communication is implemented using collective-permute-style transfers and the inputs can rotate through stages. So don't interpret every MaxText PP HLO/profile literally as simple `send TPU0 → TPU1`; the conceptual model remains **layer-sharded stages + microbatches + activation movement**, while the implementation can use a more efficient SPMD schedule. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/optimization/sharding.md?utm_source=chatgpt.com "maxtext/docs/guides/optimization/sharding.md at main · AI-Hypercomputer/maxtext · GitHub"))