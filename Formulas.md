
$$
\begin{equation} T_\text{math} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}} \end{equation}
$$
$$
\begin{equation} T_\text{comms} = \frac{\text{Communication Bytes}}{\text{Network/Memory Bandwidth Bytes/s}} \end{equation}
$$

For a Accelator, the time range can be expressed as: 

$\begin{equation} T_\text{upper} = T_\text{math} + T_\text{comms} \end{equation}$

$\begin{equation} T_\text{lower}=\max(T_\text{math}, T_\text{comms}) \end{equation}$

---
$$
\begin{equation} \text{Arithmetic Intensity} = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}} \end{equation}
$$


