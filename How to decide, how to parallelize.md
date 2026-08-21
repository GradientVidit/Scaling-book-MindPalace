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

