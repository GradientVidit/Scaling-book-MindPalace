```
						 TPU v6e SYSTEM
                              │
                              │
                    ┌─────────┴─────────┐
                    │                   │
                 POD 0               POD 1
              256 TPU chips        256 TPU chips
              16 × 16 torus        16 × 16 torus
                    │                   │
                    │                   │
              ┌─────┴─────┐       ┌─────┴─────┐
              │   SLICE   │       │   SLICE   │
              │   256     │       │   256     │
              │   TPUs    │       │   TPUs    │
              └───────────┘       └───────────┘
                    │                   │
                    └─────────┬─────────┘
                              │
                             DCN
                              │
                   INTER-SLICE COMMUNICATION
                              │
                     via HOST CPUs
```

---
# Inside one v6e Pod / 256-chip slice
```
                         v6e POD
                     256 TPU chips
                      16 × 16 torus (Mobius strip Donut)

        ┌──────────────────────────────────────┐
        │                                      │
        │  TPU ── TPU ── TPU ── TPU ── ...     │
        │   │      │      │      │             │
        │  TPU ── TPU ── TPU ── TPU ── ...     │
        │   │      │      │      │             │
        │  TPU ── TPU ── TPU ── TPU ── ...     │
        │   │      │      │      │             │
        │  ...                                 │
        │   │                                  │
        │  TPU ── TPU ── TPU ── TPU ── ...     │
        │                                      │
        │              ICI                     │
        │        TPU ↔ TPU communication       │
        │                                      │
        └──────────────────────────────────────┘

                    256 chips
                    32 hosts
                    64 × 4-chip VMs
```

---
# Host → VM → TPU relationship
```
                  PHYSICAL HOST
              ┌───────────────────┐
              │                   │
              │     HOST CPU      │
              │                   │
              │  ┌─────────────┐  │
              │  │    VM 0     │  │
              │  │             │  │
              │  │  4 TPU      │  │
              │  │  chips      │  │
              │  └─────────────┘  │
              │                   │
              │  ┌─────────────┐  │
              │  │    VM 1     │  │
              │  │             │  │
              │  │  4 TPU      │  │
              │  │  chips      │  │
              │  └─────────────┘  │
              │                   │
              └───────────────────┘
                    8 TPUs
                 per physical host
```

**Exception:** v6e also supports an 8-chip full-host VM (`v6e-8`), primarily for inference. For normal multi-host training slices, Google describes v6e slices as using 4-chip half-host VMs.

---
# Slice Sizes
```
                  v6e POD
                 256 chips
                    │
          ┌─────────┴─────────┐
          │                   │
       SLICE                SLICE
          │
          ├── 1 × 1   =   1 TPU
          │
          ├── 2 × 2   =   4 TPUs
          │
          ├── 2 × 4   =   8 TPUs
          │
          ├── 4 × 4   =  16 TPUs
          │
          ├── 4 × 8   =  32 TPUs
          │
          ├── 8 × 8   =  64 TPUs
          │
          ├── 8 × 16  = 128 TPUs
          │
          └── 16 × 16 = 256 TPUs
```

---
# Communication
```
                 SAME SLICE
                     │
                     ▼
              TPU ── ICI ── TPU
                     │
                     ▼
              TPU ── ICI ── TPU
```

```
              DIFFERENT SLICES
                     │
                     ▼
              TPU (Slice 0)
                     │
                     ▼
                  HOST CPU
                     │
                     │
                    DCN
                     │
                     ▼
                  HOST CPU
                     │
                     ▼
              TPU (Slice 1)
```

---

