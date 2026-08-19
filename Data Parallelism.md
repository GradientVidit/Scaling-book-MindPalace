Each TPU has the SAME model weights.
Each TPU processes a DIFFERENT batch.

**For understanding Symbols and Variables : Refer [[Notations]]

### 1. General Training Loop:

```
Forward:
In  [BX, D]  ×  Win  [D, F]  =  Tmp  [BX, F]
Tmp [BX, F]  ×  Wout [F, D]  =  Out  [BX, D]


Backward:
Tmp [BX, F]^T × dOut [BX, D] = dWout [F, D]
dOut [BX, D] × Wout [F, D]^T = dTmp [BX, F]
In  [BX, D]^T × dTmp [BX, F] = dWin  [D, F]
dTmp [BX, F] × Win [D, F]^T = dIn [BX, D]
```



## 2. Pure DP Process

### 2a. Initial State across TPUs

**Parameters that remain the same across TPUs:

| Category      | Parameter              |
| ------------- | ---------------------- |
| Attention     | Wq                     |
| Attention     | Wk                     |
| Attention     | Wv                     |
| Attention     | Wo                     |
| MLP           | Wup                    |
| MLP           | Wdown                  |
| MLP           | Wgate (if applicable)  |
| Normalization | Norm scale parameters  |
| Embeddings    | Token embedding        |
| Output        | LM head (if separate)  |
| Other         | Biases (if applicable) |
 
 **Parameters that are different/sharded across TPUs:

| Category | Parameter |
| -------- | --------- |
| Data     | Batch     |


## 2b. Step by Step deep dive


```
		TPU 0                                         TPU 1
		-----                                         -----

		In[A]                                          In[B]
		[BX,D]                                        [BX,D]
	       |                                             |
	       |                                             |
	       v                                             v
  +-----------------------+                  +-----------------------+
  | Tmp[A] = In[A] * Win  |                  | Tmp[B] = In[B] * Win  |
  +-----------------------+                  +-----------------------+
	       |                                             |
	       v                                             v
  +------------------------+                 +------------------------+
  | Out[A] = Tmp[A] * Wout |                 | Out[B] = Tmp[B] * Wout |
  +------------------------+                 +------------------------+
	       |                                             |
	       v                                             v
   +----------------+                            +----------------+
   |     Loss[A]    |                            |     Loss[B]    |
   +----------------+                            +----------------+
	       |                                             |
	       |                                             |
	       +-------------------+   +---------------------+
	                           |   |
	                           v   v
	                      [ global loss ]
	                  (conceptually averaged)
		                         |
		                         |
	                    ===== BACKWARD =====
		                         |
		                         v
	  dOut[A]                                      dOut[B]
		   |                                             |
		   |                                             |
		   +------------------+   +----------------------+
							  |   |
							  v   v
				 +--------------------------+
				 | dWout_local = Tmp * dOut |
			  (Do the diff to understand better)
				 +--------------------------+
							  |
							  |
	                          +------+
                                     |
                                     | ALL-REDUCE
                                     | (across TPUs)
                                     v
                         +-----------------------+
                         | dWout = averaged     |
                         | global gradient      |
                         +-----------------------+
                            |               |
                            v               v
                         TPU 0           TPU 1
                         gets same       gets same
                         dWout           dWout
```
**All other trainable weights (`Wq`, `Wk`, `Wv`, `Win`, etc.), just like Wout above,  are obtained in same way, of the global/average gradient. 
- In practice, gradients are **grouped into buckets** rather than doing one AllReduce per weight.
- Each bucket undergoes an **AllReduce**, often asynchronously.
- Multiple AllReduces can **overlap with the backward pass**.



## Surface Level Idea

```
          TPU 0                         TPU 1
          -----                         -----

       SAME Win                       SAME Win
       SAME Wout                      SAME Wout
           |                              |
        batch A                         batch B
           |                              |
           v                              v
       forward                         forward
           |                              |
           v                              v
       backward                        backward
           |                              |
       local dW                        local dW
           |                              |
           +---------- ALL-REDUCE --------+
                          |
                          v
                   averaged dW
                      /     \
                     v       v
                  TPU 0    TPU 1
                     |       |
                 optimizer optimizer
                     |       |
                     v       v
	                 SAME W_new
                     /       \
                    /         \
                   v           v
                TPU 0       TPU 1
```


  IMPORTANT:
  - Weights are REPLICATED, not partitioned.
  - Batches are SHARDED.
  - Gradients are computed locally.
  - Weight gradients are ALL-REDUCED across TPUs.
  - The averaged gradients are identical on every TPU.
  - Therefore every optimizer replica produces the same W_new.
  - dIn is NOT all-reduced; it is needed only for earlier layers.
  - Gradient AllReduce can be overlapped asynchronously with other
    backward computation where the implementation permits it.

#### Example
```
Loss
  |
  v
dOut
  |
  +----> dWout  ---- ALL-REDUCE
  |
  v
dTmp
  |
  +----> dWin   ---- ALL-REDUCE
  |
  v
dIn
  |
  v
Embedding
  |
  +----> dEmbedding ---- ALL-REDUCE

```


---
Related:
- To understand why matrices like Tmp are transposed in backward pass, refer : [[Why Transpose things in backward pass]]
- General Approximation rule: the largest model we can train with Adam and pure data parallelism has is $$
\text{num params} = \frac{\text{HBM per device}}{20}
$$
- [[Critical batch size for v6e]]

- Pure (and FSDP) data parallelism is BULK-SYNCHRONOUS:
	- Every replica must finish forward + backward before the AllReduce
	  can complete.
	- The step doesn't advance until ALL replicas are done.
	- One slow or failed TPU stalls the entire step, for every other chip.

	Consequences:
	- Stragglers (one slow chip) cost you the whole cluster's time, not just
	  their own.
	- This is the underlying reason large-scale training needs aggressive
	  checkpointing and fast restart — a single failure wastes cluster-wide
	  compute, not just one replica's.
	- Ties into elastic training / fault tolerance (relevant to your
	  MaxText Part 13 notes).

- Danger zone: variable-length sequences / uneven microbatches (padding, packed sequences, gradient accumulation with ragged batches). Naively averaging already-averaged per-replica losses under-weights replicas that had more real tokens — this is not a hypothetical, it caused a documented training bug in 2024/2025-era LLM training code. Correct pattern: normalize by GLOBAL token count, not per-replica count — i.e. sum raw (unnormalized) losses/gradients across replicas via AllReduce, THEN divide once by the true global token count. Sum-then-divide, not average-of-averages.
