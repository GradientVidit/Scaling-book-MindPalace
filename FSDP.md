
 Example:
   4 TPUs
   Each TPU gets a different batch shard
   Model parameters are sharded 1/4 across TPUs

 Notation:
   W0,W1,W2,W3  = parameter shards
   G0,G1,G2,G3  = gradient shards
   X0,X1,X2,X3  = batch shards
   OS0,OS1,OS2,OS3 = optimizer-state shards



# 0. Initial State

```
                  MODEL PARAMETERS

                 Full parameter W
        +--------------------------------+
        | W0 | W1 | W2 | W3              |
        +--------------------------------+
          1/4  1/4  1/4  1/4

              SHARDED ACROSS TPUs
```

```
   TPU 0             TPU 1             TPU 2             TPU 3
   -------            -------            -------            -------
   W0                 W1                 W2                 W3
   OS0                OS1                OS2                OS3
```

   Each TPU stores only 1/4 of:
       - parameters
       - optimizer state

```
   Each TPU also receives 1/4 of the training batch:

   TPU 0             TPU 1             TPU 2             TPU 3
   -------            -------            -------            -------
   X0                 X1                 X2                 X3
```


# 1. FORWARD PASS — PREPARE A LAYER

 We are about to execute Layer L.

 Currently for a Layer L:
```
   TPU 0             TPU 1             TPU 2             TPU 3
   -------            -------            -------            -------
   W0                 W1                 W2                 W3
```
 
 But Layer L's computation requires the complete parameter W. So a All-gather is executed to retreive other parts of that layer.

```
                    ALL-GATHER
              "Everyone shares with everyone"


   BEFORE ALL-GATHER
   ─────────────────────────────────────────────

   TPU 0              TPU 1              TPU 2              TPU 3
   ┌─────┐             ┌─────┐             ┌─────┐             ┌─────┐
   │ W₀  │             │ W₁  │             │ W₂  │             │ W₃  │
   └──┬──┘             └──┬──┘             └──┬──┘             └──┬──┘
      │                   │                   │                   │
      └───────────────────┴───────────────────┴───────────────────┘
                              │
                              ▼
                         ALL-GATHER
                     combine all pieces


   AFTER ALL-GATHER
   ─────────────────────────────────────────────

   TPU 0              TPU 1              TPU 2              TPU 3
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │ W₀ │ W₁ │ W₂ │   │ W₀ │ W₁ │ W₂ │   │ W₀ │ W₁ │ W₂ │   │ W₀ │ W₁ │ W₂ │
   │   │     │ W₃ │   │    │    │ W₃ │   │    │    │ W₃ │   │    │    │ W₃ │
   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
          │                   │                   │                   │
          └───────────────────┴───────────────────┴───────────────────┘

                 EVERY TPU NOW HAS THE FULL W of layer L
```

# 2. FORWARD COMPUTATION

 Each TPU has the COMPLETE parameter,
 but operates on its OWN batch shard.
 
```
   TPU 0             TPU 1             TPU 2             TPU 3
   -------            -------            -------            -------
   X0 + W             X1 + W             X2 + W             X3 + W
      |                  |                  |                  |
      v                  v                  v                  v
     Y0                 Y1                 Y2                 Y3
```

 This is still DATA PARALLELISM:

       different data
             +
       same parameters


 IMPORTANT:

 The complete W does NOT need to remain resident after
 this layer's computation.


# 3. RESHARD / FREE THE TEMPORARY FULL PARAMETERS

 After the layer's computation:

   full W on every TPU
              |
              v
       discard full W

 Back to:

   TPU 0             TPU 1             TPU 2             TPU 3
   -------            -------            -------            -------
   W0                 W1                 W2                 W3

 This is where FSDP gets its memory advantage.


# 4. REPEAT FOR NEXT LAYER

 Layer L+1:

       sharded W
            |
            v
       ALL-GATHER
            |
            v
       full W temporarily
            |
            v
       forward compute
            |
            v
       discard full W

 And so on...


 COMMUNICATION IS PIPELINED
 --------------------------

 Ideally, we don't do:

       compute Layer L
              |
              v
       WAIT
              |
              v
       AllGather Layer L+1
              |
              v
       compute Layer L+1

 Instead:

       Compute Layer L
       =============================>

                    AllGather Layer L+1
                    ==================>

       Compute Layer L+1
       =============================>

 The communication for the next layer can overlap
 with computation of the current layer.


# 5. BACKWARD PASS

 Eventually backward reaches Layer L.

 We saved the necessary activation during forward.

 But the backward computation still needs W.

 Current state:
```
   TPU 0             TPU 1             TPU 2             TPU 3
   -------            -------            -------            -------
   W0                 W1                 W2                 W3
```

 So again:
```
                    ALL-GATHER

       W0 --------------+
       W1 --------------+
       W2 --------------+    -----> W
       W3 --------------+
```

 Every TPU temporarily gets:
```
   TPU 0             TPU 1             TPU 2             TPU 3
   -------            -------            -------            -------
   W                  W                  W                  W
```

# 6. BACKWARD COMPUTATION

 Each TPU uses:

       its local activation
              +
       full W
              +
       its local upstream gradient

 to calculate its LOCAL contribution to the gradient.

```
   TPU 0             TPU 1             TPU 2             TPU 3
   -------            -------            -------            -------
   dW(local)          dW(local)          dW(local)          dW(local)
```

 IMPORTANT:

 These gradients are NOT identical.

 Why?

 Because each TPU processed a different batch shard:
```
       X0 -> different gradient
       X1 -> different gradient
       X2 -> different gradient
       X3 -> different gradient
```

# 7. REDUCE-SCATTER THE GRADIENT

 We need to combine the gradient contributions from all TPUs.

 But we do NOT need the complete reduced gradient on every TPU.

 We want the final gradient to be sharded in exactly the
 same ownership pattern as the parameters.


 Before ReduceScatter:
```
   TPU 0            TPU 1               TPU 2           TPU 3
   -------          -------             -------         -------
   g0=[g00,g01,g02,g03]
			        g1=[g10,g11,g12,g13]
	                                    g2=[g20,g21,g22,g23]
		                                                g3=[g30,g31,g32,g33]
```

 REDUCE-SCATTER
       |
       +---- reduce corresponding gradient pieces
       |
       +---- scatter resulting pieces
       v


 After ReduceScatter:
```

   TPU 0             TPU 1             TPU 2             TPU 3
   -------            -------            -------            -------
   G0                 G1                 G2                 G3
```

 where:

   G0 = g00 + g10 + g20 + g30
   G1 = g01 + g11 + g21 + g31
   G2 = g02 + g12 + g22 + g32
   G3 = g03 + g13 + g23 + g33

 (or the corresponding average, depending on gradient scaling)


# 8. NOW PARAMETER + GRADIENT OWNERSHIP MATCH

```
   TPU 0             TPU 1             TPU 2             TPU 3
   -------            -------            -------            -------
   W0                 W1                 W2                 W3
   G0                 G1                 G2                 G3
```

 Therefore:

       TPU 0 updates W0 using G0
       TPU 1 updates W1 using G1
       TPU 2 updates W2 using G2
       TPU 3 updates W3 using G3


# 9. OPTIMIZER UPDATE

 Optimizer state is also sharded.

 For Adam, conceptually:
```
   TPU 0             TPU 1             TPU 2             TPU 3
   -------            -------            -------            -------
   W0                 W1                 W2                 W3
   G0                 G1                 G2                 G3
   M0,V0              M1,V1              M2,V2              M3,V3
```

 Each TPU performs the optimizer update ONLY for its
 parameter shard.

```
   TPU 0:
       W0, G0, M0, V0
              |
              v
          optimizer
              |
              v
             W0'

   TPU 1:
       W1, G1, M1, V1
              |
              v
          optimizer
              |
              v
             W1'

   TPU 2:
       W2, G2, M2, V2
              |
              v
          optimizer
              |
              v
             W2'

   TPU 3:
       W3, G3, M3, V3
              |
              v
          optimizer
              |
              v
             W3'
```

# 10. END OF TRAINING STEP

 The model is again stored sharded:
```
   TPU 0             TPU 1             TPU 2             TPU 3
   -------            -------            -------            -------
   W0'                W1'                W2'                W3'
   M0,V0              M1,V1              M2,V2              M3,V3
```
 
 The next training step starts from here.


#  ONE-LINE MENTAL MODEL

 FSDP =
```
   SHARD
      ↓
   ALL-GATHER
      ↓
   COMPUTE
      ↓
   DISCARD FULL PARAMETERS
      ↓
   BACKWARD
      ↓
   REDUCE-SCATTER GRADIENTS
      ↓
   UPDATE LOCAL PARAMETER SHARD
      ↓
   SHARD AGAIN
```
 The price for dramatically lower parameter/gradient/optimizer
 memory is additional communication.

---

Reference: [[FSDP Visual]]