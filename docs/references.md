# Project Reference Data & Implementation Details

This document consolidates the key differences between our Digital Twin implementation and the original reference papers, as well as the exact physics equations utilized in the environment.

## 1. Source of Physics Equations

The core physics equations driving our Digital Twin environment (`MillingDTEnv-v1`) are sourced almost entirely from the **"DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation"** paper, with specific concepts adapted from **"Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning"** and standard manufacturing literature.

### From "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation"
*(Mo, Xia, Mei, Hu, Journal of Manufacturing Processes 156 (2025) 1006-1031)*
* **Cutting Speed ($v_c$):** $v_c = (\pi \times D \times n) / 1000$ *(Eq. 5)*
* **Feed per Minute ($f$):** $f = n \times f_r$ *(Eq. 6)*
* **Cutting Time ($t_{cut}$):** $t_{cut} = L_{cut} / f$ *(Eq. 7)*
* **Material Removal Rate (MRR):** $MRR = v_c \times a_p \times f_r$ *(Eq. 27)*
* **Tool Wear Degradation ($\Delta VB$):** The Taylor-extended wear model ($\Delta VB = C \cdot v_c^\alpha \cdot f_r^\beta \cdot a_p^\gamma \cdot t_{cut}$) is a standard manufacturing wear law, not taken from this paper. The paper's Eq. 17 is the wear *constraint* ($VBi \le VBmax$), and its virtual environment obtains the wear value from a trained data-driven model (Section 5.2) rather than from an analytic wear law.

### From "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning" / General Literature
*(Wang, Cui, Tao, Xu, Yang, Journal of Manufacturing Systems 78 (2025) 124-152)*
* **Surface Roughness ($Ra$):** The empirical regression formula we use ($Ra = k_1 f_r^2 + k_2 VB + k_3 / v_c$) is a standard traditional model that "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation" uses as a comparative baseline (Eq. 56 in their paper). "Machining parameter optimization for a batch milling system using multi-task deep reinforcement learning" utilizes a very similar mathematical structure for their baseline objectives. 

### Standard Manufacturing Engineering (Our Novelty)
* **Energy Consumption ($E$):** $E = (P_{idle} + k_c \times MRR) \times t_{cut}$. This is not explicitly in "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation". We added this standard specific cutting energy formulation to introduce the "Green Manufacturing" objective into the reward function.

---

## 2. Our Implementation vs. "DRL-based optimization for machining process: Cutting parameters dynamic optimization considering tool condition degradation"

We are remarkably close on the **Reinforcement Learning and Physics** side, but we have abstracted away the heavy **Signal Processing and Deep Learning** components.

### What we implemented identically (or very closely):
1. **The Core Digital Twin Physics:** Exact equations for calculating cutting speed, material removal rate (MRR), and the Taylor-extended tool wear degradation model.
2. **Stochastic Tool Breakage:** Probability-based catastrophic tool breakage logic (where pushing a worn tool too hard causes a sudden failure), forcing the agent to learn a safe strategy.
3. **Multi-Objective Reward:** Giving immediate penalties for violating limits ($VB_{max}$, $Ra_{max}$) and using a Hypervolume (HV) reward at the end of the workpiece to balance time and quality.
4. **Multi-Pass Logic:** The agent manages the depth of cut ($a_p$) to sequentially eat away at a total machining allowance (`kappa`), differentiating between roughing passes and the final finishing pass.

### Where we differ (What we abstracted away):
1. **No Acoustic Emission (AE) Processing:** The paper spends 10 pages building a complex "Teacher-Student CNN" to process raw audio signals from the machine to guess the tool wear. We skip this entirely; our environment acts as a perfect sensor that simply *knows* the exact tool wear based on the physics equations.
2. **Action Space (Tool Changes):** The paper uses a "Mixed Action Space" where the agent outputs continuous cutting parameters *plus* a discrete action (0 or 1) deciding whether to swap the tool. Our agent only outputs the continuous cutting parameters; we handle tool changes by simply resetting the environment (episode boundaries).
3. **Surface Roughness Model:** The paper trains a teacher-student network to estimate roughness from cutting data, with the empirical regression as its comparative baseline. We use the traditional empirical regression formula throughout.
4. **Gaussian vs Beta Distribution:** The paper writes a custom PPO algorithm using a Beta distribution to sample actions within absolute bounds. We use standard `Stable-Baselines3`, which uses a Gaussian distribution (clipped to bounds).

---

## 3. Dynamic Inference Features

Because the Digital Twin is fully parameterized in Python (`MillingDTEnv`), we can change physical or engineering parameters during inference (evaluation) without retraining to see how the agent adapts.

### The Machining Task
* `kappa_total` (default `5.0` mm): The total depth of material that needs to be removed. 
* `L_cut` (default `100.0` mm): The length of the workpiece cut.
* `D` (default `50.0` mm): The tool/workpiece diameter.

### Material & Tool Wear Physics
* `C_wear`, `alpha`, `beta`, `gamma_wear`: The Taylor-extended tool wear constants. (e.g., increasing `C_wear` simulates a softer tool).
* `k1`, `k2`, `k3`: The surface roughness regression constants (simulate different materials).

### Engineering Constraints
* `Ra_max` (default `1.6` $\mu m$): The surface roughness limit.
* `VB_max` (default `0.3` mm): The critical tool wear limit.

*(Note: Modifying machine bounds like `n_max` or `f_max` during inference is not recommended as it breaks the neural network's `[0, 1]` observation normalization bounds).*
