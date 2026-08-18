```
             SHARDED PARAMETERS
                     |
                     v
              ┌─────────────┐
              │  ALL-GATHER │
              └──────┬──────┘
                     |
                     v
             FULL PARAMETERS
             temporarily on
              every TPU
                     |
                     v
              FORWARD PASS
                     |
                     v
              discard/reshard
                     |
                     v
              ... next layers ...
                     |
                     v
              BACKWARD PASS
                     |
                     v
              ALL-GATHER
                     |
                     v
             FULL PARAMETERS
                     |
                     v
          local gradient computation
                     |
                     v
             REDUCE-SCATTER
                     |
                     v
             SHARDED GRADIENTS
                     |
                     v
             OPTIMIZER UPDATE
                     |
                     v
             SHARDED PARAMETERS
                     |
                     └───────────────┐
                                     |
                                     v
                              NEXT TRAINING STEP


 ================================================================
                 THE CORE FSDP IDEA
 ================================================================


                 MEMORY SAVING

      DON'T KEEP THIS:

   TPU0: [ W0 W1 W2 W3 ]
   TPU1: [ W0 W1 W2 W3 ]
   TPU2: [ W0 W1 W2 W3 ]
   TPU3: [ W0 W1 W2 W3 ]


      KEEP THIS:

   TPU0: [ W0 ]
   TPU1: [ W1 ]
   TPU2: [ W2 ]
   TPU3: [ W3 ]


      TEMPORARILY DO THIS:

   TPU0: [ W0 W1 W2 W3 ]  ← AllGather
   TPU1: [ W0 W1 W2 W3 ]  ← AllGather
   TPU2: [ W0 W1 W2 W3 ]  ← AllGather
   TPU3: [ W0 W1 W2 W3 ]  ← AllGather

      THEN THROW THE FULL COPIES AWAY.


      BACKWARD:

   local gradients
          |
          v
      ReduceScatter
          |
          v
   TPU0: [ G0 ]
   TPU1: [ G1 ]
   TPU2: [ G2 ]
   TPU3: [ G3 ]


      UPDATE:

   TPU0: W0 + G0 → W0'
   TPU1: W1 + G1 → W1'
   TPU2: W2 + G2 → W2'
   TPU3: W3 + G3 → W3'


 ================================================================
 DP vs FSDP — THE KEY DIFFERENCE
 ================================================================


 DATA PARALLELISM

   TPU0: FULL W + X0
   TPU1: FULL W + X1
   TPU2: FULL W + X2
   TPU3: FULL W + X3

                 BACKWARD
                    |
                 AllReduce
                    |
                    v
             FULL reduced G
             on every TPU

                    |
                    v
          every TPU updates
             its FULL W



 FSDP

   TPU0: W0 + X0
   TPU1: W1 + X1
   TPU2: W2 + X2
   TPU3: W3 + X3

          FORWARD/BACKWARD
                 |
          AllGather when
          full W is needed
                 |
                 v
          temporary full W

                 |
          backward gradients
                 |
                 v
           ReduceScatter
                 |
                 v

   TPU0: W0 + G0
   TPU1: W1 + G1
   TPU2: W2 + G2
   TPU3: W3 + G3

                 |
                 v
       each TPU updates ONLY
       its parameter shard
   ```