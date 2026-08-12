

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