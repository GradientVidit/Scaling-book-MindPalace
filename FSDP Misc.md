### A. Memory Footprint Breakdown Formula

For a model with $\Psi$ parameters trained using **FP16/BF16 mixed precision** and **FP32 Adam optimizer**:

| Component                        | Standard DDP       | ZeRO-1 (Opt State)                               | ZeRO-2 (Opt + Grad)                              | ZeRO-3 / FSDP (Full Shard)               |
| -------------------------------- | ------------------ | ------------------------------------------------ | ------------------------------------------------ | ---------------------------------------- |
| **Model Parameters (FP16/BF16)** | $2\Psi$ bytes      | $2\Psi$ bytes                                    | $2\Psi$ bytes                                    | $\frac{2\Psi}{N}$ bytes                  |
| **Gradients (FP16/BF16)**        | $2\Psi$ bytes      | $2\Psi$ bytes                                    | $\frac{2\Psi}{N}$ bytes                          | $\frac{2\Psi}{N}$ bytes                  |
| **Optimizer States (FP32 Adam)** | $12\Psi$ bytes     | $\frac{12\Psi}{N}$ bytes                         | $\frac{12\Psi}{N}$ bytes                         | $\frac{12\Psi}{N}$ bytes                 |
| **Total Model State per GPU**    | **$16\Psi$ bytes** | **$4\Psi + \frac{12\Psi}{N}$** ($\approx 4\Psi$) | **$2\Psi + \frac{14\Psi}{N}$** ($\approx 2\Psi$) | **$\frac{16\Psi}{N}$ bytes**             |
| **Transient Unsharded Buffer**   | $0$                | $0$                                              | $0$                                              | $+ 2\Psi_{\text{layer}}$ (largest layer) |

**Adam Optimizer State Math ($12\Psi$ bytes):**

- FP32 Master Parameters: $4\Psi$ bytes
- FP32 First Moment (Momentum $m_t$): $4\Psi$ bytes
- FP32 Second Moment (Variance $v_t$): $4\Psi$ bytes
- Total: $4\Psi + 4\Psi + 4\Psi = 12\Psi$ bytes.


### B. Communication Volume Derivation

Why does ZeRO-3 / FSDP incur a **+50% communication overhead** compared to DDP, ZeRO-1, and ZeRO-2?

1. **Standard DDP (Gradients AllReduce)**: $$\text{Volume} = 2 \times \left(\frac{N-1}{N}\right) \times \Psi \approx 2\Psi \text{ bytes per step}$$
2. **ZeRO-1 & ZeRO-2 (ReduceScatter Grads + AllGather Params)**: $$\text{ReduceScatter}(1\Psi) + \text{AllGather}(1\Psi) = 2\Psi \text{ bytes per step} \quad (\mathbf{0\% \text{ Communication Penalty vs DDP}})$$

3. **ZeRO-3 / FSDP (AllGather Forward + AllGather Backward + ReduceScatter Grads)**: $$\underbrace{1\Psi}_{\text{Forward AllGather}} + \underbrace{1\Psi}_{\text{Backward AllGather}} + \underbrace{1\Psi}_{\text{Backward ReduceScatter}} = \mathbf{3\Psi \text{ bytes per step}} \quad (\mathbf{+50\% \text{ Overhead vs DDP}})$$

