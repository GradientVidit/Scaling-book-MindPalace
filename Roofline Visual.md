
```text
                    Matrix Multiplication
                X[B,D] · Y[D,F] → Z[B,F]
                           │
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
    Compute Required                  Data Movement
        (FLOPs)                           (Bytes)
          │                                 │
          │                                 │
       ≈ 2BDF                     (BD + DF + BF) × s
          │                                 │
          └──────────────┬──────────────────┘
                         ▼
              Computation Intensity
                   FLOPs / Byte
                         │
                         │ Compare against
                         ▼
              Accelerator Intensity
      Peak FLOPs/s ÷ Peak Memory Bandwidth
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
Computation Intensity          Computation Intensity
          <                              ≥
Accelerator Intensity         Accelerator Intensity
          │                             │
          ▼                             ▼
 Communication Bound              Compute Bound
  (HBM is bottleneck)            (MXU is bottleneck)
          │                             │
          ▼                             ▼
    T_comms > T_math              T_math > T_comms
```

