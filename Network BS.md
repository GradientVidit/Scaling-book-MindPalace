
```
                         TPU v6e CHIP
                              │
              ┌───────────────┴───────────────┐
              │                               │
              │          TPU CORE             │
              │                               │
              │   ┌─────────┐  ┌──────────┐   │
              │   │ 2 x MXU │  │  Vector  │   │
              │   └────┬────┘  └──────────┘   │
              │        │                      │
              │   ┌────▼─────────────────┐    │
              │   │       VMEM           │    │
              │   │   local TPU memory   │    │
              │   └──────────┬───────────┘    │
              │              │                │
              └──────────────┼────────────────┘
                             │
                       load / store
                             │
                    ┌────────▼────────┐
                    │      HBM        │
                    │   32 GB/chip    │
                    └─────────────────┘
                             │
             ┌───────────────┼────────────────┐
             │               │                │
             │               │                │
           PCIe             ICI              DCN*
             │               │                │
             ▼               ▼                ▼
         HOST / VM       OTHER TPU        OTHER SLICE
                             │
                             │ ICI
                             ▼
                         OTHER TPU

* DCN path in Multislice:
  TPU → host → DCN → host → TPU
```

---

# Network hierarchy

```
                    v6e BANDWIDTH
              (Scaling Book convention)

    ┌──────────────────────────────┐
    │ VMEM                         │
    │ ~35 TB/s*                    │
    │ ~22× HBM bandwidth           │
    │ On-chip scratchpad           │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │ HBM                          │
    │ ~1,600 GB/s / chip           │
    │ High-bandwidth memory        │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │ ICI                          │
    │ 90 GB/s / link (one-way)     │
    │ 180 GB/s / link (bidir.)     │
    │ TPU ↔ TPU                    │
    | Within Slice                 |
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │ PCIe                         │
    │ ~32 GB/s / TPU               │
    │ TPU ↔ HOST CPU               │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │ DCN                          │
    │ ~12.5 GB/s / TPU             │
    │ HOST ↔ HOST                  │
    │ between slices               │
    └──────────────────────────────┘
```

# 256-chip v6e Pod

```
256-chip v6e Pod
│
├── HBM:        1,638 GB/s / chip
│
├── ICI:          90 GB/s / link
│                180 GB/s / link bidi
│
├── PCIe:         ~32 GB/s / TPU
│
├── DCN:          ~12.5 GB/s / TPU
│
├── Bisection:     3.2 TB/s
│
├── All-reduce:  102.4 TB/s
│
└── DCN:          25.6 Tbps / Pod
```

### One important notation caveat
Google's hardware spec also reports: **ICI = 800 GB/s bidirectional per chip**

---

#### v6e MEMORY CAPACITY

CPU RAM
├── 176 GB / 1-chip VM
├── 720 GB / 4-chip VM
└── 1,440 GB / 8-chip VM

HBM
└── 32 GB / TPU chip

VMEM
└── v6e capacity: not specified in Scaling Book
    v5e reference: 128 MiB

