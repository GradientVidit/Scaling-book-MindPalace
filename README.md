# ⚡ JAX Scaling Book Notes

Personal notes covering the **JAX Scaling Book**, TPU hardware architecture, matrix sharding, and distributed training parallelism.

> 💡 **Best viewed in [Obsidian](https://obsidian.md/)** to utilize internal links (`[[Note]]`) and graph navigation.

---

## 📌 How to Read

You don't need to manually read through every file in the repository. Start with the main numbered notes in order, and navigate through the linked notes as you read:

1. [`1. Rooflines.md`](1.%20Rooflines.md) – Compute vs. memory bounds & arithmetic intensity
2. [`2. TPU.md`](2.%20TPU.md) – TPU architecture, memory hierarchy, and network topology
3. [`3. Sharding.md`](3.%20Sharding.md) – Matrix sharding primitives & GEMM rules
4. [`5. Training.md`](5.%20Training.md) – Distributed parallelism strategies (DP, FSDP, TP, PP, CP, SP)
