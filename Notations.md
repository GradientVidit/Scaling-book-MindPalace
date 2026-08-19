| Symbol              | Meaning                                           | Shape / scope            |
| ------------------- | ------------------------------------------------- | ------------------------ |
| `In`                | Input activations to the layer                    | `[BX, D]`                |
| `BX`                | Local batch size processed by one TPU             | Per TPU                  |
| `D`                 | Input/output feature dimension                    | Model dimension          |
| `F`                 | Hidden/intermediate feature dimension             | Model dimension          |
| `Win`               | Input projection weight matrix                    | `[D, F]`                 |
| `Wout`              | Output projection weight matrix                   | `[F, D]`                 |
| `Tmp`               | Intermediate activation                           | `[BX, F]`                |
| `Out`               | Layer output activation                           | `[BX, D]`                |
| `Loss`              | Per-example loss                                  | `[BX]`                   |
| `dOut`              | Gradient of loss w.r.t. `Out`                     | `[BX, D]`                |
| `dTmp`              | Gradient of loss w.r.t. `Tmp`                     | `[BX, F]`                |
| `dWout`             | Gradient of loss w.r.t. `Wout`                    | `[F, D]`                 |
| `dWin`              | Gradient of loss w.r.t. `Win`                     | `[D, F]`                 |
| `dIn`               | Gradient of loss w.r.t. `In`                      | `[BX, D]`                |
| `A`                 | Batch processed by TPU 0                          | Local batch              |
| `B`                 | Batch processed by TPU 1                          | Local batch              |
| `local`             | Gradient computed from one TPU's local batch      | Per TPU                  |
| `ALL-REDUCE`        | Combines corresponding gradients across all TPUs  | Same result on every TPU |
| `averaged gradient` | Global-batch gradient after AllReduce + averaging | Same on every TPU        |
| `W_new`             | Updated model weights after optimizer step        | Replicated on every TPU  |
| `TPU 0 / TPU 1`     | Data-parallel replicas                            | Each holds full model    |
