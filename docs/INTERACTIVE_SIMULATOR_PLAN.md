# Precision Machining & Material Removal Simulation: Digital Twin Architecture

**Project:** Energy-Aware Deep Reinforcement Learning for Dynamic CNC Milling Optimization  
**Course:** ME6324 - AI in Manufacturing (IIT Madras)  
**Deliverable:** Autonomous Digital Twin Simulator (`app.py`, `sim/machining_sim.py`)  
**Design Standard:** Anti-Slop Industrial Instrument Guidelines (`.agents/skills/antislop/`, `antislop-ui`, `frontend-ui-engineering`)  

---

## 1. Problem Statement & Anti-Slop Directive

Generic AI demos present superficial dashboards with neon purple gradients, floating cards, vague "AI score" bars, and no physics fidelity. 
This Digital Twin simulator is designed as an **authentic CNC Machine Tool Telemetry & Material Removal Workstation**:
- **Zero AI Buzzwords**: No "unlocking next-gen synergy" or "revolutionary optimization" copy. Every figure corresponds to standard ISO/CIRP manufacturing units: $\text{mm}$, $\text{RPM}$, $\text{mm/rev}$, $\mu\text{m}$, $\text{mm}^3/\text{min}$, $\text{kJ}$, and $\text{kW}$.
- **Direct Physics Linkage**: Every pass directly executes empirical Taylor-extended flank wear models, specific cutting power calculations, and surface roughness regressions parameterized from actual experimental literature.
- **Physical Material Removal Simulation (Layer by Layer)**: A visual representation of the cylindrical/billet workpiece cross-section, showing initial stock, tool pass cuts, remaining allowance $\kappa$, depth per pass $a_p$, and the resulting micro-surface topography.

---

## 2. Innovative Technical Features

### A. 2D Cross-Sectional Material Removal & Tool Engagement Simulation
- **Stock Depletion Geometry**:
  - Initial stock profile: Cylindrical billet of diameter $D$ with total machining allowance $\kappa_{\text{total}}$.
  - Pass-by-pass progressive cutout: Each cut removes a rectangular layer of depth $a_p$ and cut length $L_{\text{cut}}$.
  - The remaining stock boundary updates in real-time.
  - Interactive chip-load / engagement diagram: Visualizes the tool entry, radial engagement, and axial depth of cut.

### B. Micro-Topography & Surface Roughness Profile ($Ra$) Generator
- Generates the realistic feed-mark cusps profile based on tool nose radius $r_\epsilon$, feed per tooth $f_z$, and flank wear condition:
  $$y_{\text{cusp}}(x) \approx \frac{x^2}{8 r_\epsilon} + \text{stochastic chatter/wear perturbation}$$
- Visually shows the peak-to-valley profile alongside the ISO 1302 centerline average $Ra$, contrasting it with the $1.60\ \mu\text{m}$ ceiling.

### C. Live ISO 6983 G-Code Block Generator
- Emits real, syntax-highlighted CNC G-code for Fanuc / Siemens controllers for the exact parameter sequence calculated by the DRL agent:
  ```gcode
  %
  O1001 (PPO AUTONOMOUS MILLING ADAPTIVE RUN)
  G21 G90 G94 (METRIC, ABSOLUTE, FEED PER MIN)
  G17 G40 G49 G80
  T01 M06 (D50 CORNER MILL)
  S7500 M03 (PPO COMPUTED SPINDLE SPEED)
  G00 X0. Y0. Z5.0
  (--- PASS 1: ROUGHING PASS ---)
  G00 Z-2.500 (PASS 1 DEPTH ap = 2.50 mm)
  G01 X100.0 F2400.0 (FEED RATE fr = 0.320 mm/rev)
  ...
  ```

### D. Multi-Controller Head-to-Head Telemetry
- Simulates the exact same workpiece simultaneously under:
  1. **Dynamic PPO (Our Autonomous Policy)**
  2. **Static NSGA-II (Multi-Objective Pareto Knee)**
  3. **Rule-Based Heuristic (Industrial Feed/Speed Wear Derating)**
  4. **Constant High (Max Throughput Baseline)**
- Compares machining cycle time, total energy, final $Ra$, and tool degradation.

### E. Industrial Design Palette & Clean Human Interface (Anti-Slop)
- Strict dark-slate/industrial palette inspired by Siemens Sinumerik & Heidenhain CNC control panels:
  - Base: Deep Slate (`#0f172a`), Dark Charcoal (`#1e293b`)
  - Accent: Precision Cyan (`#06b6d4`), Safety Amber (`#f59e0b`), Alarm Crimson (`#ef4444`), Compliance Emerald (`#10b981`)
- No gratuitous animations or purple glows. Real-time updates with clear error bounds and 95% confidence bounds.

---

## 3. Architecture & Code Structure

```
G:/My Drive/IITM/AI in Mfg/

 sim/
    __init__.py
    machining_sim.py          # Trajectory simulation, surface profile and G-code synthesis

 app.py                         # Streamlit interactive application

 docs/
    INTERACTIVE_SIMULATOR_PLAN.md   # This design document
```

### Module Specifications:
1. `sim/machining_sim.py`:
   - `simulate_pass_sequence(model, env_kwargs, policy_type)`: Executes multi-pass machining and records state vectors $(VB, \kappa, n, f_r, a_p, MRR, Ra, E, P_c)$.
   - `generate_gcode(trajectory_df, workpiece_cfg)`: Converts parameter trajectory into executable CNC G-code.
   - `compute_workpiece_geometry(allowance_total, passes_ap)`: Calculates remaining stock boundary coordinates for 2D cross-sectional rendering.
   - `generate_surface_profile(passes_data)`: Generates feed-mark surface profile coordinates.

2. `app.py`:
   - Interactive input controls (workpiece diameter, cut length, total allowance, initial flank wear, green slider $w_2$).
   - Layout:
     - **Top Metrics Row**: 4 key indicators (Throughput MRR, Specific Energy, Surface Roughness $Ra$, Tool Wear $VB$).
     - **Tab 1: Physical Material Removal & Tool Path**: 2D workpiece stock depletion visualizer + dynamic cut layers.
     - **Tab 2: Surface Micro-Topography & Tool Wear**: Profile diagram + Flank wear progression vs. limit lines.
     - **Tab 3: Head-to-Head Controller Comparison**: Side-by-side performance cards and radar chart.
     - **Tab 4: Fanuc / ISO G-Code Program**: Syntax-highlighted, downloadable `.nc` file.
