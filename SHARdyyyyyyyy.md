

```
                 SHARDED MATRIX MULTIPLICATION
                              │
               A[I,J] × B[J,K] → C[I,K]
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
         CASE 1            CASE 2           CASE 3
      Contracting       One contracting   Both contracting
      dimension not       dimension          dimensions
        sharded            sharded            sharded
             │                │                │
             ▼                ▼                ▼
          NO COMM          ALL-GATHER       LOCAL MATMUL
                                               │
                                               ▼
                                          PARTIAL SUMS
                                               │
                                               ▼
                                          ALL-REDUCE
                                               │
                                               ▼
                                          COMPLETE RESULT


                         CASE 4
                           │
             Both non-contracting dimensions
                  sharded on SAME axis
                           │
                           ▼
                       INVALID
                           │
                           ▼
                  ALL-GATHER one input
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                Gather A       Gather B
```


DP
```
          GPU 0                         GPU 1
     ┌─────────────┐               ┌─────────────┐
     │ 5B weights  │               │ 5B weights  │
     │ Adam state  │               │ Adam state  │
     └──────┬──────┘               └──────┬──────┘
            │                              │
          batch A                        batch B
            │                              │
         backward                       backward
            │                              │
        gradient A                     gradient B
            │                              │
            └────────── ALL-REDUCE ────────┘
                         │
                   averaged gradient
                         │
                  ┌──────┴──────┐
                  ↓             ↓
              optimizer     optimizer
                  ↓             ↓
             same W_new      same W_new
```

FSDP
```
                  ONE LOGICAL MODEL
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
      Layer 1          Layer 2          Layer 3
        │                │                │
        ↓                ↓                ↓
    all-gather        all-gather        all-gather
        │                │                │
        ↓                ↓                ↓
    full L1           full L2           full L3
   temporarily       temporarily       temporarily
        │                │                │
        ↓                ↓                ↓
     compute           compute           compute
        │                │                │
        ↓                ↓                ↓
      reshard          reshard          reshard
```
	