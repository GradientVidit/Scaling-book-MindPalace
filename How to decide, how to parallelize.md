# Flow & Intuition

```text
                              MODEL
                                │
                 ┌──────────────┼──────────────┐
                 ↓              ↓              ↓
               Memory         Batch        Dimensions
                 │              │              │
                 ↓              ↓              ↓
               FSDP?           DP?            TP?
                 │              │              │
                 └──────────────┼──────────────┘
                                ↓
                         TPU topology
                                │
                       ICI vs DCN placement
                                ↓
                       Roofline analysis
                                ↓
                    eliminate bad candidates
                                ↓
                    benchmark remaining 2–5
                                ↓
                          choose winner
```

## DP / FSDP

The Scaling Book gives the TPUv5 rule of thumb:

> DP/FSDP become communication-bound when batch per shard drops below roughly `2550 / M`, where `M` is the number of mesh axes.

### Intuition

**DP likes large batches.**

```text
large batch
    ↓
lots of computation
    ↓
DP is great
```

```text
small batch
    ↓
not enough computation
    ↓
DP eventually breaks
```

I split the batch AND use that same axis to split the model state

```text
large batch
    ↓
lots of computation
    ↓
FSDP communication can be hidden
```

```text
small batch
    ↓
not enough computation
    ↓
FSDP becomes communication-bound
```

---

## TP

### Intuition

**TP doesn't care much about batch size; it cares about how much computation exists in the model dimension you're splitting.**

So:

```text
large F
    ↓
can support more TP
```

```text
small F
    ↓
TP hits communication wall sooner
```

This is why TP is particularly useful when your batch has become too small for FSDP/DP.

---

## FSDP + TP

**FSDP + TP — intuition: "split both the batch and the layer"**

**FSDP moves weights; TP moves activations.**

```text
FSDP degree ↑
    ↓
batch per FSDP shard ↓
    ↓
FSDP communication gets harder
```

```text
TP degree ↑
    ↓
activation shard ↓
    ↓
TP communication gets harder
```

But combining them gives you a sweet spot.

The book derives:

```text
B / N > α² / (M_X M_Y F)
```

for remaining compute-bound, where `B/N` is essentially batch per device.

For TPUv5 and `F ≈ 32k`, this gives roughly **100 tokens/device**, versus roughly **850 tokens/device** for pure FSDP in the same simplified setup.

---

## PP

**PP trades communication problems for scheduling/bubble problems.**

Therefore PP wants:

> **many microbatches.**


---
---
---

## Example

Relevant Candidates for 2048 Chips / 8 v6e pods :

| Layout                       | ICI (within pod, 256-way)                                 | DCN (across pods, 8-way)     | Character                                                                          |
| ---------------------------- | --------------------------------------------------------- | ---------------------------- | ---------------------------------------------------------------------------------- |
| Pure FSDP everywhere         | `ici_fsdp_parallelism=256`                                | `dcn_fsdp_parallelism=8`     | Single 2048-way FSDP shard; weights all-gathered even over DCN                     |
| FSDP(ICI) + DP(DCN)          | `ici_fsdp_parallelism=256`                                | `dcn_data_parallelism=8`     | Classic "multi-slice" — full FSDP inside each pod, gradient all-reduce across pods |
| FSDP+TP(ICI) + DP(DCN)       | `ici_fsdp_parallelism=64`, `ici_tensor_parallelism=4`     | `dcn_data_parallelism=8`     | Adds intra-layer TP to relax batch-size pressure                                   |
| FSDP(ICI) + PP(DCN)          | `ici_fsdp_parallelism=256`                                | `dcn_pipeline_parallelism=8` | One pod = one pipeline stage; only activations cross DCN                           |
| HSDP: DP+FSDP(ICI) + DP(DCN) | `ici_data_parallelism=k`, `ici_fsdp_parallelism=256/k`    | `dcn_data_parallelism=8`     | Partial replication within pod to shrink the FSDP group                            |
| FSDP+CP(ICI) + DP(DCN)       | `ici_fsdp_parallelism=X`, `ici_context_parallelism=256/X` | `dcn_data_parallelism=8`     | For long-context variants (large `max_target_length`)                              |

In this way, after calculating relevant candidates based on theory, we can get theoretical idea of top candidates based on our requirements and objectives. In the end, we can choose final layout from top candidates via empirical testing.
