
This gem covers two things:

1. How to actually run FSDP in MaxText today (install → config → command).
2. A deep, from-first-principles answer to "how are weights actually sharded across TPUs" — because the confusion across internet (per-layer split? physical vs logical? mesh-shaped? "units"?) comes from real, legitimate different layers of the same system, not contradictory sources. Once you see all the layers stacked, they all become the same picture.

---

## Part 1: MaxText's FSDP — what it actually is

**The one-sentence version:** in MaxText you don't implement FSDP. You describe a _mesh_ of TPUs and tell the compiler which array axes are allowed to be split across which mesh axes. The XLA compiler (via **GSPMD**, or its newer successor **Shardy**) looks at your annotations and inserts every AllGather / ReduceScatter itself. There is no `FSDP` class, no wrapping, no hooks — it's a config flag.

This is the single most important mental shift coming from PyTorch: **FSDP in MaxText is not a wrapper you apply to a model. It's a _consequence_ of how you sharded the parameter arrays.**

### Installing MaxText (current stable path, as of the latest release)


### Running a plain FSDP pretraining job

Say you have **4 TPU chips** on one host (e.g. a `v5e-4` or `v6e-4` slice). The whole "shard everything across all 4" FSDP setup is one flag: `ici_fsdp_parallelism=4`.

```bash
python3 -m MaxText.train src/MaxText/configs/base.yml \
  run_name="my-first-fsdp-run" \
  base_output_directory=gs://<your-bucket>/ \
  model_name="llama2-7b" \
  dataset_type=synthetic \
  per_device_batch_size=1 \
  ici_fsdp_parallelism=4 \
  steps=10
```

What this line means concretely:

- `ici_fsdp_parallelism=4` — carve up your **ICI** mesh (the fast on-slice interconnect between your 4 chips) into an axis of size 4, and name that axis `fsdp`.
- Every other parallelism knob (`ici_data_parallelism`, `ici_tensor_parallelism`, `ici_pipeline_parallelism`, `ici_expert_parallelism`, `ici_context_parallelism`, plus the `dcn_*` equivalents for going across multiple slices/pods) defaults to 1, so `fsdp` is the only axis actually splitting anything.
- The product of all `ici_*` and `dcn_*` values must equal your total chip count.

Because MaxText is pure JAX/XLA underneath, **this exact command is what scales**: change `ici_fsdp_parallelism=4` to `256` and rerun on a bigger slice — no code changes. That's the whole pitch of the framework.

### Combining FSDP with other axes (the realistic setup)

Pure FSDP is rarely used alone once you leave toy scale. A very common production recipe:

```bash
python3 -m MaxText.train src/MaxText/configs/base.yml \
  run_name="llama-70b-multislice" \
  base_output_directory=gs://<bucket>/ \
  model_name="llama2-70b" \
  per_device_batch_size=4 \
  ici_fsdp_parallelism=16 \
  dcn_data_parallelism=2 \
  steps=1000
```

- `ici_fsdp_parallelism=16` — shard params/grads/optimizer state across the 16 chips _within_ each slice (fast ICI network).
- `dcn_data_parallelism=2` — replicate that whole 16-way-sharded model across 2 slices, synced over the slower cross-slice network (DCN).

This is the "**FSDP within a slice, DP across slices**" pattern MaxText explicitly recommends, for the same reason ZeRO/FSDP practitioners on GPU put FSDP inside a fast NVLink domain and DP/HSDP across the slower cross-node fabric — you want the _frequent, large_ AllGather traffic on the _fastest_ wire.

For very large models you also see `ici_tensor_parallelism`, `ici_expert_parallelism` (for MoE), `ici_pipeline_parallelism` (for models too big even for FSDP alone) mixed in on top of `fsdp`. Which combination is right depends on model size, MLP width, and sequence length — MaxText's own docs walk through the "arithmetic intensity" math for choosing this, but the FSDP axis itself always works the same way described below regardless of what else you combine it with.

### Two things that specifically affect _how_ sharding plays out that you should know from day one

1. **Scanned layers.** By default MaxText stacks all your transformer decoder layers into **one array** with a leading `layers` axis, and runs the forward pass with `jax.lax.scan` over that axis (instead of Python-unrolling N separate layers). This isn't a sharding decision — it's a compile-speed decision — but it changes what "a parameter" even means, which matters a lot for the confusion in Part 2 below.
2. **Trusting the compiler.** MaxText's stated philosophy is: write plain JAX/Flax, annotate arrays with sharding _intent_, and let XLA figure out the actual collective-communication schedule (fusing AllGathers, overlapping them with compute, etc.), rather than hand-writing that scheduling like PyTorch FSDP's forward/backward hooks do. This is why there's no analogue to `auto_wrap_policy`, `backward_prefetch`, or `reshard_after_forward` in MaxText — those are exactly the decisions XLA is making for you under the hood.

---

## Part 2: How weights are _actually_ sharded — resolving the confusion

You'll read four things that sound contradictory:

- (a) "shard each layer into 4 parts, each TPU holds 1/4 of every layer"
- (b) "sharding is physical, not logical"
- (c) "sharding follows the mesh/topology of the TPUs"
- (d) "each TPU doesn't hold exactly 1/4 of everything — it holds some 'units', and the units sum to the whole model"

**All four are true. They're describing four different layers of the same stack**, from the physical hardware up to your Python code. Once you see the stack, they stop contradicting each other.

```
Layer 4:  Individual tensors      "this weight matrix: which of its
          (sharding rules)         axes gets split, on which mesh axis?"

Layer 3:  Logical axes            "activation_batch, embed, mlp, layers, heads..."
          (named, meaningful)      — human-readable names for tensor dimensions

Layer 2:  Physical mesh axes      "fsdp, data, tensor, expert, pipeline..."
          (device mesh)            — named axes of the actual device grid

Layer 1:  Physical topology       the real 3D torus/mesh wiring of ICI links
          (hardware)               between chips, plus DCN between slices
```

Let's walk it bottom-up, since that's the order it's actually decided in.

### Layer 1 → 2: Physical topology → physical mesh axes

TPU chips within a slice are wired together in a real physical topology (for `v5e`/Trillium, effectively a 2D/3D torus of point-to-point ICI links; across slices, chips talk over the slower DCN — the data-center network). This is claim (c): sharding genuinely does have to respect the hardware topology, because that's what determines communication cost.

MaxText's answer to "respect the topology" is a **device mesh**: you take your N chips and arrange them (in software) into an n-dimensional grid, and give each grid dimension a name — e.g. `data`, `fsdp`, `tensor`, `expert`, `pipeline`, `context`, `sequence` — MaxText defines up to 13 such named physical axes in its config. `mesh_utils.create_device_mesh` (or `create_hybrid_device_mesh` for the two-level ICI/DCN case) does the actual placement of chips into this grid, trying to align mesh axes with efficient physical links. `ici_fsdp_parallelism=4` is exactly "make the `fsdp` axis of this mesh 4 chips wide, placed along fast ICI links."

**This is the "physical" level people mean** in claim (b): the mesh axis names (`fsdp`, `data`, `tensor`...) are physical device-grid axes, independent of anything about your model.

### Layer 2 → 3: Physical mesh axes → logical axes

Your model's tensors don't know about `fsdp` or `tensor` — they know about `embed_dim`, `mlp_dim`, `num_heads`, `batch`, `layers`. MaxText defines a separate **logical axis rules** table that maps these meaningful names to one or more physical mesh axes. For example (simplified from MaxText's actual `base.yml`):

|Logical axis (what the tensor calls it)|Physical mesh axis(es) it maps to|
|---|---|
|`activation_batch`|`data`, `fsdp`, `fsdp_transpose`, ... (anything that shards the batch)|
|`mlp`|`tensor`, `tensor_transpose`, `expert`|
|`embed`|`fsdp`, `sequence`, `tensor_sequence`, ...|
|`layers`|usually unsharded (or `pipeline` if using pipeline parallelism)|

**This is claim (b)'s "logical" level**, and it directly answers "physical vs logical, which is it?" — it's _both_, deliberately, as two separate layers so that (1) model code can be written using meaningful names without knowing the hardware, and (2) you can retarget the same model code to a totally different mesh shape by only editing the rules table, not the model.

### Layer 3 → 4: Logical axes → the actual sharding of one tensor

Finally, each individual weight array in the model declares which of _its own_ axes correspond to which logical names — e.g. a feed-forward kernel of shape `[embed, mlp]` is tagged with logical axes `("embed", "mlp")`. The framework then looks up the rules table, resolves those to physical mesh axes, and that's the tensor's actual sharding. This is done via Flax's `nn.with_logical_partitioning` for parameters and `nn.with_logical_constraint` for activations (as hints to the compiler).

### Now, claim (a): "shard each layer into 1/4, each TPU has 1/4 of every layer"

This is **correct at the level of a single weight matrix**, with one refinement. When MaxText's `fsdp` mesh axis has size 4, a weight array like `[embed, mlp]` gets **one of its axes split into 4 contiguous chunks**, one chunk per TPU — e.g. TPU 0 literally holds rows `0:embed/4` of that matrix, TPU 1 holds `embed/4:embed/2`, etc. That's dim-0 (or whichever axis) chunking, exactly like FSDP2's per-parameter `DTensor` sharding on GPU. It is **not** splitting the matrix into 4 arbitrary or unequal "conceptual units" — it's literally `array.reshape` + `array[i*chunk:(i+1)*chunk]` per device, done by the compiler.

So per weight array, yes: with pure FSDP=4 and nothing else sharding that axis, each of the 4 TPUs holds exactly 1/4 of it. This applies independently to **every** parameter array in the model that has that logical axis (which, for FSDP, is essentially all of them) — attention projections, MLP kernels, embeddings, norms if large enough, etc.

### Now, claim (d): "not exactly 1/4 of everything, but 'units' that sum to the whole model"

This is where **scanned layers** (mentioned in Part 1) matters, and it's also where people are often unknowingly mixing in the **PyTorch FSDP mental model**, which genuinely is different.

**In PyTorch FSDP**, the sharding unit really is coarser than "one axis of one array": you `auto_wrap` whole submodules (e.g. each transformer block) into an FSDP "unit." Each unit's parameters get flattened and concatenated into one big 1D buffer (`FlatParameter`, in FSDP1) before being chunked evenly across ranks. So in PyTorch it's true that a TPU/GPU "doesn't hold exactly 1/4 of every individual weight" — it holds 1/4 of a _flattened bundle_ of many weights glued together, and the boundaries between individual tensors don't align with shard boundaries. (FSDP2 improved this — it shards each parameter individually via DTensor — but the "unit = wrapped submodule, decided by a wrap policy" framing is still how people commonly describe it.) **This is very likely the "units, cumulative sum = whole model" source you read** — it's a true and accurate description, just of the PyTorch/GPU implementation, not JAX/TPU.

**In MaxText/JAX, there is no such wrapping-unit concept.** Sharding is decided per-array, per-axis, by the logical axis rules — not by grouping submodules into flattened buffers. So the "units" framing doesn't directly apply. The closest thing MaxText has to a "unit" is the **scanned layer stack**: because all N decoder layers are stacked into one array with a leading `layers` dimension, from JAX's point of view there is only _one_ feed-forward kernel array in the whole scanned block (shape `[layers, embed, mlp]`), not N separate ones. When `fsdp` shards that array's `embed` axis, every TPU gets 1/4 of the `embed` dimension — but for **all `layers` at once**, because they're one array. So it actually is closer to "exactly 1/4 of everything, everywhere, all the time" than to "some TPUs get more units than others" — scanning makes the picture _more_ uniform, not less.

### Putting it together with a 4-TPU example

Concretely, for `ici_fsdp_parallelism=4` and nothing else sharding:

- Every weight matrix in the model has (at least) one axis chunked into 4 equal pieces, one piece per TPU.
- Because layers are scanned, this chunking applies uniformly across all layers simultaneously — TPU _i_ holds shard _i_ of the embed dimension for every layer, not "some layers on this TPU and some on that one" (that would be **pipeline parallelism**, a genuinely different strategy where whole layers, not slices of every layer, are assigned to different devices — see the Pipeline Parallelism section in MaxText's sharding docs).
- Which physical axis of the weight gets sharded (`embed` vs `mlp`) is chosen by the logical axis rules, tuned so the resulting AllGather/ReduceScatter shapes are TPU-friendly (powers of two, aligned to hardware tile sizes) — this is the "practical, not arbitrary" answer to "how exactly is it split."
- Nothing is manually all-gathered by you: `jax.jit` traces the whole `train_step`, GSPMD/Shardy sees the sharding annotations on inputs/outputs, and inserts the AllGathers (before each matmul needs the full weight) and ReduceScatters (after backward, to reshard the gradient) automatically — this is literally the same math as PyTorch FSDP's forward-AllGather / backward-AllGather / ReduceScatter pattern, just decided by the compiler instead of hand-written hooks.

### Quick glossary, mapped to what confused you

|What you read|What it actually refers to|
|---|---|
|"shard each layer, 1/4 per TPU"|Correct, per weight _array_ (dim-0/dim-1 chunking) — and applies to all scanned layers at once|
|"physical, not logical"|The **physical mesh axis** layer (`fsdp`, `data`, `tensor`, tied to real device wiring)|
|"logical" (the other half of that split)|The **logical axis** layer (`embed`, `mlp`, `batch`, tied to model semantics), mapped to physical via rules|
|"sharded by mesh/topology"|The device mesh is literally built from the chip topology (ICI torus + DCN), via `create_device_mesh` / `create_hybrid_device_mesh`|
|"not 1/4 of everything, but units"|This is the **PyTorch FSDP wrapping-unit** mental model (flattened submodule buffers) — doesn't apply to MaxText/JAX, which shards per-array|

---

## What to read next

- MaxText's own **Sharding on TPUs** guide (`guides/optimization/sharding.html`) — has the full "arithmetic intensity" derivation for FSDP, TP, EP, PP, CP, and when to combine which. This is the natural next step after this file.
