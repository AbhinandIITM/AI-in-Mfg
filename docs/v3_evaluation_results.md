# V3 Evaluation Results: PPO vs SAC

The following visualizations and metrics detail the performance of the trained v3 models (PPO vs SAC) over 10 evaluation batches.

## 1. Action Strategies (Roughing vs Finishing)
Both algorithms successfully learned to differentiate between roughing (bulk material removal) and finishing (surface quality pass). 

![Action Strategy Comparison](/v3_eval_actions.png)

* **Finishing (Blue)**: Both algorithms consistently dropped the feed rate to the minimum boundary ($\approx 0.05$ mm/r) and maximized spindle speed to ensure a smooth surface finish.
* **Roughing (Red)**: **PPO** aggressively maximized throughput (Feed rate $> 0.31$, Spindle Speed $\approx 7500$ RPM). **SAC** opted for a highly energy-efficient zone (Feed rate $\approx 0.28$, Spindle Speed $\approx 5600$ RPM), balancing tool wear and power consumption perfectly.

---

## 2. Constraint Adherence
The true test of the models is whether they can push the limits of the tool without violating the rigid engineering constraints: 
* Surface Roughness limit: $Ra \le 1.6 \mu m$
* Tool Wear limit: $VB \le 0.3 mm$

![Constraint Adherence](/v3_eval_constraints.png)

* **PPO**: The aggressive roughing strategy caused PPO to occasionally violate the surface roughness limit (hitting a maximum of $1.61 \mu m$) and experience one failure episode.
* **SAC**: The energy-aware SAC strategy proved to be much safer. It perfectly adhered to all constraints, keeping $Ra$ bounded under $1.50 \mu m$. Furthermore, it achieved a maximum tool wear of $0.28$ mm, effectively squeezing $93\%$ of the usable life out of the cutting tool without ever breaking it.

---

### Raw Metrics Summary

| Algorithm | Mean Roughing Feed ($f_r$) | Mean Roughing Speed ($n$) | Max Surface Roughness ($Ra$) | Max Tool Wear ($VB$) | Broken Tools |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PPO (Seed 1)** | $0.304$ mm/r | $7291$ RPM | $1.55 \mu m$ | $0.258$ mm | **1** |
| **PPO (Seed 2)** | $0.318$ mm/r | $7797$ RPM | **$1.61 \mu m$** (Violated) | $0.232$ mm | 0 |
| **SAC (Seed 1)** | $0.285$ mm/r | $5579$ RPM | $1.50 \mu m$ | $0.281$ mm | 0 |
| **SAC (Seed 2)** | $0.275$ mm/r | $5792$ RPM | $1.41 \mu m$ | $0.279$ mm | 0 |
