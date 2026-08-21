
> **CP and SP are both ways of reducing the activation memory associated with long sequences, but they do it differently.**
> 
> In MaxText specifically, **CP shards the sequence dimension for attention**, while **SP shards the sequence dimension for the non-attention parts but changes the attention sharding to heads**. ([MaxText](https://maxtext.readthedocs.io/en/latest/guides/optimization/sharding.html?utm_source=chatgpt.com "Sharding on TPUs — MaxText documentation"))

---

# 1. First: where CP and SP fit

Think of the dimensions of a Transformer activation:

```text
Activation = [Batch, Sequence, Hidden]
              B       S       H
```

The parallelisms you've learned primarily attack different dimensions:

```text
DP       → B
FSDP     → B  (while sharding weights/state)
TP       → H / MLP dimensions
PP       → Layers
CP       → S
SP       → S  + attention heads
```

So the simplest intuition is:

```text
DP/FSDP: "Give different examples to different TPUs."

CP:      "Give different parts of the SAME sequence to different TPUs."

SP:      "Give different parts of the SAME sequence to different TPUs,
          but use the attention-head dimension to make attention efficient."
```

This is why CP/SP become particularly interesting when **sequence length gets huge**.

---

# 2. Why do we need CP at all?

Suppose:

```text
batch = 1
sequence = 128K
hidden = 8192
```

Imagine the activation:

```text
[B, S, H]

[1, 128K, 8192]
```

With ordinary DP/FSDP, you can distribute **batch**, but:

```text
B = 1
```

There isn't much batch to distribute.

TP can distribute the hidden/MLP dimensions, but the sequence dimension remains enormous.

So you can end up with:

```text
TPU 0:
[1, 128K, H/4]

TPU 1:
[1, 128K, H/4]

TPU 2:
[1, 128K, H/4]

TPU 3:
[1, 128K, H/4]
```

Each TPU still processes **128K tokens**.

CP says:

> Why not distribute the sequence itself?

With CP=4:

```text
TPU 0 → tokens 0 ... 31K
TPU 1 → tokens 31K ... 63K
TPU 2 → tokens 63K ... 95K
TPU 3 → tokens 95K ... 128K
```

Conceptually:

```text
              Sequence
                 │
       ┌─────────┼─────────┐
       ↓         ↓         ↓
      TPU0      TPU1      TPU2 ... TPU3
       │         │
      S/4       S/4
```

Therefore the **sequence dimension of activations is sharded**.

That's the fundamental idea of CP.

MaxText explicitly describes CP as similar to FSDP, except that it shards the **sequence dimension of activations rather than the batch dimension**. ([MaxText](https://maxtext.readthedocs.io/en/latest/guides/optimization/sharding.html?utm_source=chatgpt.com "Sharding on TPUs — MaxText documentation"))

---

# 3. The key difference between FSDP and CP

This is worth locking into your mental model.

### FSDP

```text
                    Batch
                      ↓
          ┌───────────┼───────────┐
          ↓           ↓           ↓
        TPU 0       TPU 1       TPU 2

        different examples
```

Each TPU owns different **batch examples**.

---

### CP

```text
                  Sequence
                     ↓
          ┌──────────┼──────────┐
          ↓          ↓          ↓
        TPU 0      TPU 1      TPU 2

        different tokens
        of the SAME examples
```

Each TPU owns different **tokens of the sequence**.

That is the core distinction.

---

# 4. But attention creates a problem

This is where CP gets interesting.

Suppose:

```text
Sequence =

A B C D E F G H
```

CP=2:

```text
TPU 0 → A B C D
TPU 1 → E F G H
```

Now consider causal attention.

For token `H`, it needs to attend to:

```text
A B C D E F G H
```

But TPU 1 only owns:

```text
E F G H
```

The required keys/values:

```text
A B C D
```

are sitting on TPU 0.

So simply splitting the sequence isn't enough.

---

# 5. CP's attention trick

For attention, MaxText shards **queries by sequence**, but the keys and values must become available across the CP group.

Conceptually:

```text
TPU 0:

Q0
K0
V0


TPU 1:

Q1
K1
V1
```

Each TPU keeps its local queries:

```text
TPU 0 → Q0
TPU 1 → Q1
```

But K/V are exchanged:

```text
        K0 V0
          ↓
TPU 0 ─────────→ TPU 1

        K1 V1
          ↓
TPU 1 ─────────→ TPU 0
```

After the communication:

```text
TPU 0:
Q0 + K0 + K1 + V0 + V1

TPU 1:
Q1 + K0 + K1 + V0 + V1
```

Then:

```text
TPU 0 computes attention for Q0

TPU 1 computes attention for Q1
```

So **the computation remains distributed over queries**, while every TPU obtains the K/V information necessary for its queries.

MaxText's current default CP strategy is `all_gather`; it also supports a ring strategy on certain GPU/TPU paths. ([MaxText](https://maxtext.readthedocs.io/en/latest/guides/optimization/sharding.html?utm_source=chatgpt.com "Sharding on TPUs — MaxText documentation"))

---

# 6. Why CP becomes particularly attractive at long context

Attention has roughly:

```text
Attention compute ∝ S²
```

So when:

```text
S = 8K
```

attention is manageable.

But:

```text
S = 128K
```

is radically different.

CP=4 gives each device only roughly:

```text
S/4
```

worth of query-side attention work.

The total work remains the same, but it is distributed across devices.

And more importantly:

> **The activation memory associated with the sequence is divided across the CP devices.**

This is exactly why CP is useful for long-context training. MaxText notes that CP allows smaller per-device batch dimensions because the sequence itself is distributed. ([MaxText](https://maxtext.readthedocs.io/en/latest/guides/optimization/sharding.html?utm_source=chatgpt.com "Sharding on TPUs — MaxText documentation"))

---

# 7. One subtle CP issue: causal attention imbalance

There is a nasty problem if you naively split the sequence.

Suppose:

```text
TPU 0 → first quarter
TPU 1 → second quarter
TPU 2 → third quarter
TPU 3 → fourth quarter
```

For causal attention:

```text
token 1 → attends to ~1 token
token 2 → attends to ~2 tokens
...
token 128K → attends to ~128K tokens
```

Therefore later sequence chunks have much more useful attention computation.

You could get:

```text
TPU 0 → ███
TPU 1 → ██████
TPU 2 → █████████
TPU 3 → █████████████
```

TPU 3 becomes the bottleneck.

MaxText therefore **stripes the sequence** rather than simply assigning contiguous chunks. ([MaxText](https://maxtext.readthedocs.io/en/latest/guides/optimization/sharding.html?utm_source=chatgpt.com "Sharding on TPUs — MaxText documentation"))

Conceptually:

```text
CP rank 0 → beginning + end
CP rank 1 → second + second-last
CP rank 2 → third + third-last
CP rank 3 → middle chunks
```

This balances the amount of causal-attention computation.

Importantly, this striping is done on the **initial input**, rather than repeatedly throughout every layer, so the overhead is small. ([MaxText](https://maxtext.readthedocs.io/en/latest/guides/optimization/sharding.html?utm_source=chatgpt.com "Sharding on TPUs — MaxText documentation"))

---

# 8. Now SP

SP is more subtle because **"Sequence Parallelism" is used differently in different systems**.

For **MaxText**, the definition is quite specific.

MaxText SP:

> **Shard layer inputs and feed-forward activations along the sequence dimension, but for attention shard Q/K/V across heads rather than sequence.** ([MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.1/guides/optimization/sharding.html?utm_source=chatgpt.com "Sharding on TPUs — MaxText documentation"))

That gives us:

```text
                    SP
                     │
          ┌──────────┴──────────┐
          │                     │
       MLP/NORM              Attention
          │                     │
      shard S              shard heads
```

This is the key thing to understand.

---

# 9. Why doesn't SP simply shard attention along sequence?

Because attention has another naturally shardable dimension:

```text
Heads
```

For example:

```text
16 attention heads
```

can naturally become:

```text
TPU 0 → heads 0-3
TPU 1 → heads 4-7
TPU 2 → heads 8-11
TPU 3 → heads 12-15
```

Attention doesn't contract across heads.

Each head can independently calculate:

```text
Attention(Q_h, K_h, V_h)
```

So heads are an extremely convenient dimension to distribute.

MaxText therefore uses:

```text
SP:

MLP activations → shard sequence
Attention       → shard heads
```

rather than making attention itself sequence-sharded. ([MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.1/guides/optimization/sharding.html?utm_source=chatgpt.com "Sharding on TPUs — MaxText documentation"))

---

# 10. So what happens when we enter attention?

This creates a layout transition.

Before attention:

```text
[Batch, Sequence/4, Hidden]
```

distributed across SP devices.

Then attention wants:

```text
[Batch, Sequence, Heads/4, HeadDim]
```

So MaxText performs an **All-to-All** to redistribute the activation.

Conceptually:

```text
          sequence-sharded
                 │
                 │ All-to-All
                 ↓
           head-sharded
                 │
                 ↓
             Attention
                 │
                 │ All-to-All
                 ↓
          sequence-sharded
```

This is the central communication pattern of MaxText SP.

The documentation explicitly describes this as transferring the sharding from sequence → heads and then heads → sequence using All-to-All communication. ([MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.1/guides/optimization/sharding.html?utm_source=chatgpt.com "Sharding on TPUs — MaxText documentation"))

---

# 11. Concrete SP example

Suppose:

```text
B = 1
S = 16
H = 1024
heads = 16

SP = 4
```

Initially:

```text
TPU 0 → tokens 0-3
TPU 1 → tokens 4-7
TPU 2 → tokens 8-11
TPU 3 → tokens 12-15
```

Each owns:

```text
[1, 4, 1024]
```

Now enter attention.

SP wants the attention heads distributed:

```text
TPU 0 → heads 0-3
TPU 1 → heads 4-7
TPU 2 → heads 8-11
TPU 3 → heads 12-15
```

So an All-to-All redistributes the activation.

After it:

```text
TPU 0 → [1, 16, 4 heads]
TPU 1 → [1, 16, 4 heads]
TPU 2 → [1, 16, 4 heads]
TPU 3 → [1, 16, 4 heads]
```

Now each TPU can independently perform attention for its heads.

After attention:

```text
head-sharded
      │
      │ All-to-All
      ↓
sequence-sharded
```

and the rest of the Transformer continues.

---

# 12. CP vs SP — the important difference

Now we can make the distinction very cleanly:

||CP|SP|
|---|---|---|
|Main thing being reduced|Sequence activation memory|Sequence activation memory|
|MLP activations|Sequence-sharded|Sequence-sharded|
|Attention Q|Sequence-sharded|Head-sharded|
|Attention K/V|All-gathered across CP|Head-sharded|
|Main attention communication|All-Gather|All-to-All|
|Attention computation|Distributed over sequence/query|Distributed over heads|
|Special requirement|Handles long sequences|Needs enough attention heads|
|MaxText TPU support|Yes|Yes|
|Best intuition|**Split the context**|**Split sequence outside attention, heads inside attention**|

MaxText's documentation explicitly notes that SP is constrained by the available KV heads because both TP and SP consume the head dimension. ([MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.1/guides/optimization/sharding.html?utm_source=chatgpt.com "Sharding on TPUs — MaxText documentation"))

---

# 13. The head-count constraint is important

Suppose:

```text
KV heads = 8
```

and:

```text
TP = 4
SP = 2
```

Then:

```text
TP × SP = 8
```

which uses all 8 KV heads.

That's okay.

But:

```text
TP = 8
SP = 2
```

would require:

```text
8 × 2 = 16
```

head partitions.

Impossible with only 8 KV heads.

MaxText therefore states the constraint roughly as:

```text
TP × SP < KV heads
```

for the applicable configuration. ([MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.1/guides/optimization/sharding.html?utm_source=chatgpt.com "Sharding on TPUs — MaxText documentation"))

This is one reason SP is less generally applicable than CP.

---

# 14. CP vs SP intuition

I would remember them like this:

### CP

> **"My sequence is too long. Split the sequence across TPUs."**

```text
S
│
├── TPU 0
├── TPU 1
├── TPU 2
└── TPU 3
```

But attention requires global K/V:

```text
Q local
K/V → AllGather
```

---

### SP

> **"My sequence is too long, but attention already has a very convenient dimension — heads."**

So:

```text
MLP:

S
│
├── TPU 0
├── TPU 1
├── TPU 2
└── TPU 3


Attention:

Heads
│
├── TPU 0
├── TPU 1
├── TPU 2
└── TPU 3
```

with:

```text
Sequence → Heads : All-to-All
Heads → Sequence : All-to-All
```

---

# 15. Why SP can be attractive on TPU

This is the particularly interesting MaxText-specific part.

CP has:

```text
Q → sequence sharded
K/V → AllGather
```

SP instead has:

```text
Q/K/V → head sharded
```

and communicates through All-to-All.

Since attention heads are naturally independent, this can be an efficient way to exploit the TPU's interconnect **when you have enough heads**.

MaxText's documentation says SP can be more efficient than CP when there are enough heads, and currently describes SP as a TPU-attention-kernel-oriented strategy; for GPUs it recommends CP instead. ([MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.1/guides/optimization/sharding.html?utm_source=chatgpt.com "Sharding on TPUs — MaxText documentation"))

---

# 16. Don't confuse MaxText SP with the common Megatron SP

This is **very important**.

When you read papers or NVIDIA/Megatron material, "Sequence Parallelism" often means something different.

The commonly discussed Megatron SP extends TP by distributing certain activations along the sequence dimension to avoid replicating them across TP ranks.

MaxText's SP is its own sharding strategy:

```text
MaxText SP:

sequence-shard MLP activations
+
head-shard attention
+
All-to-All between layouts
```

So when reading external material, **always check what they mean by SP**.

MaxText explicitly calls its head-sharding behavior fairly MaxText-specific. ([MaxText](https://maxtext.readthedocs.io/en/maxtext-v0.2.1/guides/optimization/sharding.html?utm_source=chatgpt.com "Sharding on TPUs — MaxText documentation"))

---

# 17. Where CP/SP sit in your overall mental model

You can now think of MaxText parallelism as attacking different axes:

```text
                         Transformer
                             │
              ┌──────────────┼──────────────┐
              │              │              │
             Data          Model          Layers
              │              │              │
           DP/FSDP           TP             PP
              │              │
              │              │
              └──────┐   ┌───┘
                     │   │
                  Sequence
                     │
                 ┌───┴───┐
                 │       │
                CP       SP
                 │       │
             shard S   shard S
                       outside
                      attention
                         │
                    shard heads
                    in attention
```

And the really useful memory rule is:

```text
DP    → split examples
FSDP  → split model states
TP    → split tensor computation
PP    → split layers
CP    → split context/sequence
SP    → split sequence + redistribute to heads for attention
```

---

# 18. One final distinction: CP is more general for long context

If your primary problem is:

> **"I have 128K/256K/1M context and activation memory is killing me."**

CP is the more direct tool:

```text
CP → sequence dimension directly
```

If you're on TPU and have a model with enough heads and the MaxText attention implementation supports the SP path efficiently, SP can be attractive because it avoids the CP attention pattern of globally AllGathering K/V, instead using All-to-All to change the layout.

But SP has a much stronger architectural constraint:

```text
available heads
        ↓
TP × SP
```

whereas CP fundamentally operates on the sequence dimension.

---

## The cleanest mental picture

If you remember only this:

```text
                  ACTIVATION
                [B, S, H]
                   │
        ┌──────────┼──────────┐
        │          │          │
       DP         CP         TP
        │          │          │
        B          S          H
     sharded     sharded    sharded
```

Then SP is the special case:

```text
                  SP
                   │
             ┌─────┴─────┐
             │           │
          MLP/NORM     Attention
             │           │
             S         HEADS
          sharded     sharded
             │           │
             └─────┬─────┘
                All-to-All
```

That is the conceptual foundation.

One MaxText detail worth noting for your future layout work: the current MaxText mesh actually exposes separate physical axes for `context`, `sequence`, and `tensor_sequence`, in addition to `tensor`, `fsdp`, `data`, `pipeline`, etc.; the current base config also marks **DCN sequence parallelism as "never recommended"**, while ICI sequence/context axes are available. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

So when we get to your **2048-v6e / 100B layout problem**, CP/SP aren't just theoretical concepts—they become additional axes that can be composed with your FSDP/TP/DP/PP choices.