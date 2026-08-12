
```
┌──────────────────────────────────────────────────────────────────────────┐
│                            TPU v6e CHIP                                  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                         HBM                                        │  │
│  │                 HIGH BANDWIDTH MEMORY                              │  │
│  │                                                                    │  │
│  │  Stores large tensors / model state:                               │  │
│  │    • Model weights       • Activations                             │  │
│  │    • Gradients            • Optimizer state                        │  │
│  │                                                                    │  │
│  │             Large capacity + high memory bandwidth                 │  │
│  └──────────────────────────────┬─────────────────────────────────────┘  │
│                                 │                                        │
│                         data movement                                    │
│                         HBM ↔ VMEM                                       │
│                                 │                                        │
│                                 ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                          TPU CORE                                  │  │
│  │                                                                    │  │
│  │  ┌──────────────────────┐       ┌───────────────────────────────┐  │  │
│  │  │        VMEM          │       │         2 x MXU               │  │  │
│  │  │                      │       │                               │  │  │
│  │  │  Vector / local      │──────►│  Matrix Multiply Unit         │  │  │
│  │  │  memory              │       │  256 × 256 systolic array     │  │  │
│  │  │                      │       │  Matrix multiply + accumulate │  │  │
│  │  │  Much smaller than   │       │                               │  │  │
│  │  │  HBM; close to       │       │  Main engine for GEMMs /      │  │  │
│  │  │  compute             │       │  matrix operations            │  │  │
│  │  └──────────────────────┘       └─────────────┬─────────────────┘  │  │
│  │                                               │                    │  │
│  │                                      accumulators / results        │  │
│  │                                               │                    │  │
│  │  ┌──────────────────────┐       ┌─────────────▼─────────────────┐  │  │
│  │  │     VECTOR UNIT      │       │        SCALAR UNIT            │  │  │
│  │  │                      │       │                               │  │  │
│  │  │ Vector / elementwise │       │ Scalar operations +           │  │  │
│  │  │ operations           │       │ control / address calculations│  │  │
│  │  └──────────────────────┘       └───────────────────────────────┘  │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                         I/O / INTERCONNECT                         │  │
│  │   ┌────────────────────┐                    ┌────────────────────┐ │  │
│  │   │    Using PCIe      │                    │     Using ICI      │ │  │
│  │   │                    │                    │                    │ │  │
│  │   │  Host CPU ↔ TPU    │                    │  TPU ↔ TPU         │ │  │
│  │   │                    │                    │  chip-to-chip      │ │  │
│  │   └─────────┬──────────┘                    └─────────┬──────────┘ │  │
│  └─────────────┼────────────────────────────────────────┼─────────────┘  │
└────────────────┼────────────────────────────────────────┼────────────────┘
                 │                                        │
                 ▼                                        ▼
            ┌───────────┐                         ┌───────────────┐
            │ CPU HOST  │                         │ OTHER TPU(s)  │
            │           │                         │               │
            │ Host-side │                         │ TPU ↔ TPU via │
            │ software  │                         │ ICI           │
            └───────────┘                         └───────────────┘
```

---
---

| Component          |               TPU v6e |
| ------------------ | --------------------: |
| TensorCores / chip |                 **1** |
| MXUs / TensorCore  |                 **2** |
| MXU size           |         **256 × 256** |
| BF16 peak          |       **918 TFLOP/s** |
| INT8 peak          |     **1,836 TFLOP/s** |
| HBM                |             **32 GB** |
| HBM bandwidth      |        **~1.64 TB/s** |
| ICI ports          |                 **4** |
| ICI                |       nearest 4 chips |
| Host DRAM          | **1536 GiB per host** |
