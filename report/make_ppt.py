#!/usr/bin/env python3
"""
Generate PowerPoint presentation from the project content.
Run: python make_ppt.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# Colors
IIT_BLUE = RGBColor(0, 51, 102)
IIT_GOLD = RGBColor(204, 153, 0)
DARK_GRAY = RGBColor(51, 51, 51)
MED_GRAY = RGBColor(102, 102, 102)
LIGHT_GRAY = RGBColor(240, 240, 240)
WHITE = RGBColor(255, 255, 255)
RED = RGBColor(200, 0, 0)
GREEN = RGBColor(0, 128, 0)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

def add_background(slide, color):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_shape(slide, left, top, width, height, fill_color=None, line_color=None, line_width=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.background()
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width or Pt(1)
    else:
        shape.line.fill.background()
    return shape

def add_text_box(slide, left, top, width, height, text, font_size=18, bold=False, color=DARK_GRAY, alignment=PP_ALIGN.LEFT, font_name='Calibri'):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return txBox

def add_bullet_list(slide, left, top, width, height, items, font_size=16, color=DARK_GRAY, bold_first=False, spacing=Pt(6)):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = 'Calibri'
        p.space_after = spacing
        if bold_first and i == 0:
            p.font.bold = True
        p.level = 0
    return txBox

def add_two_column(slide, left, top, width, height, left_items, right_items, font_size=15, color=DARK_GRAY, title_left="", title_right=""):
    col_width = width / 2 - Inches(0.15)
    # Left column
    if title_left:
        add_text_box(slide, left, top, col_width, Inches(0.4), title_left, font_size=font_size+2, bold=True, color=IIT_BLUE)
        top_content = top + Inches(0.45)
    else:
        top_content = top
    add_bullet_list(slide, left, top_content, col_width, height - Inches(0.5), left_items, font_size=font_size, color=color)
    # Right column
    right_left = left + col_width + Inches(0.3)
    if title_right:
        add_text_box(slide, right_left, top, col_width, Inches(0.4), title_right, font_size=font_size+2, bold=True, color=IIT_BLUE)
    add_bullet_list(slide, right_left, top_content, col_width, height - Inches(0.5), right_items, font_size=font_size, color=color)

# ============================================================
# SLIDE 1: Title Slide
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
add_background(slide, IIT_BLUE)

# Gold accent line
add_shape(slide, Inches(1), Inches(2.8), Inches(11.333), Pt(4), fill_color=IIT_GOLD)

add_text_box(slide, Inches(1), Inches(1), Inches(11.333), Inches(1.5),
    "Digital Twin-Driven Deep Reinforcement Learning\nfor Energy-Aware CNC Milling Parameter Optimization",
    font_size=36, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(1), Inches(3.1), Inches(11.333), Inches(0.8),
    "ME6324 — AI in Manufacturing  |  Course Project Report",
    font_size=22, color=IIT_GOLD, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(1), Inches(4.2), Inches(11.333), Inches(0.6),
    "Abhinand  •  Department of Mechanical Engineering  •  IIT Madras",
    font_size=18, color=WHITE, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(1), Inches(5.2), Inches(11.333), Inches(0.6),
    "Git Revision: 87a4021  •  All results reproducible from committed artifacts",
    font_size=14, color=RGBColor(180, 200, 220), alignment=PP_ALIGN.CENTER)

# ============================================================
# SLIDE 2: Problem Statement
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Problem: Dynamic Cutting Parameter Optimization in CNC Face Milling",
    font_size=30, bold=True, color=WHITE)

add_text_box(slide, Inches(0.5), Inches(1.4), Inches(12), Inches(0.6),
    "Choose spindle speed (n), feed (f_r), depth of cut (a_p) pass-by-pass to balance:",
    font_size=20, color=DARK_GRAY)

items = [
    ("🎯 Productivity", "Maximize Material Removal Rate: MRR = v_c · a_p · f_r"),
    ("⚡ Energy Efficiency", "Minimize total electrical energy per part (idle + cutting)"),
    ("📏 Surface Quality", "Finished roughness R_a ≤ 1.6 μm (ISO 1302)"),
    ("🔧 Tool Life", "Flank wear V_B ≤ 0.3 mm; avoid catastrophic breakage"),
]
y = Inches(2.2)
for title, desc in items:
    box = add_shape(slide, Inches(0.8), y, Inches(11.7), Inches(1.0), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
    add_text_box(slide, Inches(1.0), y + Inches(0.1), Inches(2.5), Inches(0.8), title, font_size=18, bold=True, color=IIT_BLUE)
    add_text_box(slide, Inches(3.6), y + Inches(0.1), Inches(8.5), Inches(0.8), desc, font_size=17, color=DARK_GRAY)
    y += Inches(1.15)

add_text_box(slide, Inches(0.5), Inches(6.7), Inches(12), Inches(0.5),
    "Traditional fixed parameters from handbooks cannot adapt to tool wear, material variation, or geometry changes.",
    font_size=16, bold=True, color=RED)

# ============================================================
# SLIDE 3: Proposed Approach
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Approach: Physics-Grounded Digital Twin + Deep RL",
    font_size=30, bold=True, color=WHITE)

# Three columns
col_w = Inches(3.8)
col_gap = Inches(0.25)
titles = ["🏭 Digital Twin", "🤖 DRL Agent (PPO)", "🔬 Experimental Rigor"]
contents = [
    [
        "Speed-dependent cutting energy: k_c(v_c) = k_c0 (v_c/v_ref)^m",
        "Extended Taylor tool wear model",
        "Empirical surface roughness regression",
        "Stochastic tool breakage",
        "Domain randomization per episode",
    ],
    [
        "Gymnasium environment (8/11-D obs, 3-D action)",
        "Energy-aware reward: w₁·MRR − w₂·Energy + progress − penalties",
        "Conditioned policy: single model covers full Pareto frontier",
        "PPO with VecNormalize, 4 parallel envs, CPU training",
    ],
    [
        "100-scenario Monte Carlo with 95% CIs",
        "Ablation: 7 energy weights × 3 seeds = 21 policies",
        "5 classical baselines (incl. receding-horizon)",
        "Transfer experiment: cross-material adaptation",
        "Full reproducibility: RESULTS.md from artifacts",
    ],
]

x = Inches(0.5)
for i, (title, items) in enumerate(zip(titles, contents)):
    add_shape(slide, x, Inches(1.4), col_w, Inches(5.5), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
    add_text_box(slide, x + Inches(0.2), Inches(1.5), col_w - Inches(0.4), Inches(0.5), title, font_size=20, bold=True, color=IIT_BLUE)
    add_bullet_list(slide, x + Inches(0.2), Inches(2.1), col_w - Inches(0.4), Inches(4.5), items, font_size=16, color=DARK_GRAY)
    x += col_w + col_gap

# ============================================================
# SLIDE 4: Physics - Key Equations
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Digital Twin Physics (config/physics.py — single source of truth)",
    font_size=30, bold=True, color=WHITE)

equations = [
    ("Cutting Speed", "v_c = π D n / 1000  [m/min]"),
    ("Material Removal Rate", "MRR = v_c · a_p · f_r  [mm³/min]"),
    ("Flank Wear Rate", "ΔV_B = C_wear · v_c^α · f_r^β · a_p^γ · t_cut"),
    ("Surface Roughness", "R_a = k₁ f_r² + k₂ V_B + k₃ / v_c"),
    ("Specific Cutting Energy", "k_c(v_c) = k_c0 (v_c / v_ref)^m  ← KEY: creates trade-off"),
    ("Total Power", "P_total = P_idle + k_c(v_c) · MRR"),
    ("Energy per Pass", "E_pass = P_total · t_cut"),
]

y = Inches(1.4)
for name, eq in equations:
    add_shape(slide, Inches(0.5), y, Inches(6), Inches(0.65), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1))
    add_text_box(slide, Inches(0.7), y + Inches(0.05), Inches(1.8), Inches(0.55), name, font_size=16, bold=True, color=IIT_BLUE)
    add_text_box(slide, Inches(2.6), y + Inches(0.05), Inches(3.8), Inches(0.55), eq, font_size=16, color=DARK_GRAY)
    y += Inches(0.72)

# Right side: insight
add_shape(slide, Inches(7), Inches(1.4), Inches(5.8), Inches(5.2), fill_color=RGBColor(230, 240, 250), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(7.3), Inches(1.6), Inches(5.2), Inches(0.5), "💡 Key Insight", font_size=22, bold=True, color=IIT_GOLD)
insight_items = [
    "Without speed-dependence (m=0):",
    "  Cutting energy ∝ volume removed (fixed)",
    "  Idle energy ∝ 1/cycle_time",
    "  Both objectives push to MAX speed → no trade-off",
    "",
    "With m=0.6:",
    "  Cutting energy RISES with speed",
    "  Idle energy FALLS with speed",
    "  Part energy has INTERIOR MINIMUM",
    "  Energy weight w₂ selects operating point on this curve",
    "",
    "Validated by: test_part_energy_has_an_interior_minimum_in_speed",
]
add_bullet_list(slide, Inches(7.3), Inches(2.2), Inches(5.2), Inches(4.2), insight_items, font_size=15, color=DARK_GRAY)

# ============================================================
# SLIDE 5: Environment Design
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Environment: MillingDTEnv (Gymnasium Compliant)",
    font_size=30, bold=True, color=WHITE)

# Two columns
left_items = [
    "Two registrations:",
    "  • MillingDTEnv-v1 (base, 8-D obs)",
    "  • MillingDTEnv-w2-v1 (conditioned, 11-D obs)",
    "",
    "Observation (8 features, each onto [0,1]):",
    "  0: V_B  (tanh squashed)",
    "  1: κ_rem (tanh squashed)",
    "  2-4: prev n, f_r, a_p (machine envelope)",
    "  5: MRR / MRR_max",
    "  6: R_a (tanh squashed)",
    "  7: tool_broken flag",
    "  8-10: w₂, C_wear, k₁ (conditioned only)",
    "",
    "Action: continuous [-1,1]³ → clipped → mapped to machine envelope",
]
right_items = [
    "Reward per step:",
    "  r = 5·(w₁·MRR/MRR_max − w₂·E_pass/E_ref)",
    "       + 2·Δ(stock removed)",
    "       − 50·1[V_B > V_B_safe]   − 100·1[broken]",
    "       − 50·1[finishing pass ∧ R_a > R_a_safe]",
    "       − 0.05",
    "  +200 for a part finished in spec, +≤5 Pareto margin",
    "",
    "Domain Randomization (randomize=True):",
    "  • Workpiece: κ_total [2,15], D [40,150], L_cut [50,200]",
    "  • Material/Tool: C_wear [0.6,1.4]×, k₁,k₂,k₃ [0.8,1.2]× nominal",
    "  • Initial wear: V_B0 ~ U[0, 0.8]·V_B_max",
    "  • Same tables for training + Monte Carlo eval",
    "",
    "E_ref recomputed per scenario from its own D, L_cut, k_c",
    "The objective scale (5) lets w₂ compete with the shaping terms",
]

add_two_column(slide, Inches(0.5), Inches(1.3), Inches(12.3), Inches(5.8),
    left_items, right_items, font_size=15, color=DARK_GRAY,
    title_left="State & Action", title_right="Reward & Randomization")

# ============================================================
# SLIDE 6: RL Training Setup
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Reinforcement Learning: PPO (Stable-Baselines3)",
    font_size=30, bold=True, color=WHITE)

# Table of hyperparameters
from pptx.util import Inches, Pt
table_shape = slide.shapes.add_table(12, 2, Inches(0.5), Inches(1.4), Inches(6), Inches(5.5))
table = table_shape.table
table.columns[0].width = Inches(2.5)
table.columns[1].width = Inches(3.5)

params = [
    ("Parameter", "Value"),
    ("Policy", "MlpPolicy (2×256 hidden)"),
    ("Learning rate", "3×10⁻⁴ (linear schedule)"),
    ("n_steps", "2048"),
    ("Batch size", "64"),
    ("n_epochs", "10"),
    ("γ (discount)", "0.99"),
    ("GAE λ", "0.95"),
    ("Clip range", "0.2"),
    ("Entropy coeff", "0.02 (keeps the policy exploring)"),
    ("VecNormalize", "norm_obs=True, norm_reward=True"),
    ("Envs / Device", "4 (SubprocVecEnv) / CPU (2.5× faster than GPU)"),
]

for i, (k, v) in enumerate(params):
    cell0 = table.cell(i, 0)
    cell1 = table.cell(i, 1)
    cell0.text = k
    cell1.text = v
    for cell in (cell0, cell1):
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.size = Pt(14)
            paragraph.font.name = 'Calibri'
            if i == 0:
                paragraph.font.bold = True
                paragraph.font.color.rgb = WHITE
            else:
                paragraph.font.color.rgb = DARK_GRAY
    if i == 0:
        cell0.fill.solid()
        cell0.fill.fore_color.rgb = IIT_BLUE
        cell1.fill.solid()
        cell1.fill.fore_color.rgb = IIT_BLUE

# Right side: training runs
add_shape(slide, Inches(7), Inches(1.4), Inches(5.8), Inches(5.5), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
add_text_box(slide, Inches(7.3), Inches(1.6), Inches(5.2), Inches(0.5), "Training Runs", font_size=22, bold=True, color=IIT_BLUE)
runs = [
    "1️⃣ Primary PPO: 1M steps, seed 42, randomize=True",
    "    → models/PPO_seed42/",
    "",
    "2️⃣ Conditioned PPO: 1M steps, seed 42, on MillingDTEnv-w2-v1",
    "    → models/PPO_conditioned_seed42/",
    "",
    "3️⃣ Ablation Grid: 7 weights × 3 seeds = 21 policies",
    "    w₂ ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}",
    "    seeds {42, 43, 44}, 150k steps each",
    "    Parallel via run_experiments.py (multiprocessing + heartbeat)",
    "",
    "Heartbeat callback: queue signal every 1000 steps → stall detection",
]
add_bullet_list(slide, Inches(7.3), Inches(2.2), Inches(5.2), Inches(4.5), runs, font_size=15, color=DARK_GRAY)

# ============================================================
# SLIDE 7: Monte Carlo Results
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Results: Primary PPO — Monte Carlo Sweep (100 Scenarios)",
    font_size=30, bold=True, color=WHITE)

# Main metrics table
table_shape = slide.shapes.add_table(6, 4, Inches(0.5), Inches(1.4), Inches(7), Inches(3.2))
table = table_shape.table
table.columns[0].width = Inches(3)
table.columns[1].width = Inches(1.3)
table.columns[2].width = Inches(1.3)
table.columns[3].width = Inches(1.4)

mc_data = [
    ("Metric", "Mean", "Std", "95% CI"),
    ("MRR (mm³/min)", "1239", "621", "[1116, 1363]"),
    ("Energy/part (kJ)", "68.2", "48.7", "[58.5, 77.9]"),
    ("Cycle time (s)", "19.4", "14.1", "[16.6, 22.2]"),
    ("Max R_a (μm)", "1.475", "0.235", "[1.428, 1.521]"),
    ("Max V_B (mm)", "0.124", "0.068", "[0.111, 0.138]"),
]

for i, row in enumerate(mc_data):
    for j, val in enumerate(row):
        cell = table.cell(i, j)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(14)
            p.font.name = 'Calibri'
            if i == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
            else:
                p.font.color.rgb = DARK_GRAY
    if i == 0:
        for j in range(4):
            table.cell(i, j).fill.solid()
            table.cell(i, j).fill.fore_color.rgb = IIT_BLUE

# Constraint violations
vio_table = slide.shapes.add_table(5, 3, Inches(0.5), Inches(4.8), Inches(7), Inches(2.2))
vt = vio_table.table
vt.columns[0].width = Inches(3.5)
vt.columns[1].width = Inches(1.5)
vt.columns[2].width = Inches(2)

vio_data = [
    ("Constraint", "Rate", "95% CI"),
    ("Roughness violations (worst pass)", "39%", "[29%, 49%]"),
    ("Roughness violations (finished surface)", "0%", "[0%, 0%]"),
    ("Flank wear violations (V_B > 0.3 mm)", "0%", "[0%, 0%]"),
    ("Tool breakage", "1%", "[0%, 3%]"),
]

for i, row in enumerate(vio_data):
    for j, val in enumerate(row):
        cell = vt.cell(i, j)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.name = 'Calibri'
            if i == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
            else:
                p.font.color.rgb = DARK_GRAY
    if i == 0:
        for j in range(3):
            vt.cell(i, j).fill.solid()
            vt.cell(i, j).fill.fore_color.rgb = IIT_BLUE

# Key takeaway box
add_shape(slide, Inches(8), Inches(1.4), Inches(4.8), Inches(5.6), fill_color=RGBColor(230, 245, 230), line_color=GREEN, line_width=Pt(2))
add_text_box(slide, Inches(8.3), Inches(1.6), Inches(4.2), Inches(0.5), "✅ Key Findings", font_size=22, bold=True, color=GREEN)
takeaways = [
    "• Finished surface ALWAYS within spec (0% violations)",
    "• Flank wear NEVER exceeded (0% violations)",
    "• 99% parts completed within all constraints",
    "• Intermediate passes exceed safety margin (61%)",
    "  → policy pushes limits, recovers by finish pass",
    "• High MRR variance (std = 50% of mean)",
    "  → domain randomization works: diverse operating points",
    "• Mean operating point: 7241 RPM, 0.254 mm/r, 2.11 mm, 4 passes",
]
add_bullet_list(slide, Inches(8.3), Inches(2.2), Inches(4.2), Inches(4.5), takeaways, font_size=15, color=DARK_GRAY)

# ============================================================
# SLIDE 8: Ablation — Pareto Frontier
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Energy Weight Ablation: Empirical Pareto Frontier",
    font_size=30, bold=True, color=WHITE)

# Ablation table
table_shape = slide.shapes.add_table(9, 7, Inches(0.3), Inches(1.4), Inches(7.5), Inches(4.5))
table = table_shape.table
cols = [1.2, 1.3, 1.2, 1.0, 1.0, 1.0, 0.8]
for i, w in enumerate(cols):
    table.columns[i].width = Inches(w)

abl_data = [
    ("w₂", "MRR\n(mm³/min)", "Energy\n(kJ)", "Cycle\n(s)", "n\n(RPM)", "f_r\n(mm/r)", "Frontier"),
    ("0.0", "999.9±31", "82.8±0.7", "45.8±2", "6974", "0.177", "✗ dominated"),
    ("0.2", "990.5±137", "76.3±2.2", "31.2±6", "7307", "0.186", "✗ dominated"),
    ("0.4", "1069.0±40", "71.5±0.4", "22.9±1", "7205", "0.212", "✓"),
    ("0.5", "1026.2±57", "71.0±0.7", "23.2±0.5", "7115", "0.208", "✓"),
    ("0.6", "1008.4±23", "68.3±0.8", "21.6±1", "6775", "0.223", "✓"),
    ("0.8", "888.3±58", "65.8±1.4", "23.1±1", "6307", "0.222", "✓"),
    ("1.0", "876.6±43", "65.4±1.7", "22.7±1", "5924", "0.245", "✓"),
    ("Frontier:", "1069→877", "71.5→65.4", "", "", "", "↓18%  ↓8.5%"),
]

for i, row in enumerate(abl_data):
    for j, val in enumerate(row):
        cell = table.cell(i, j)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.font.name = 'Calibri'
            p.alignment = PP_ALIGN.CENTER
            if i == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
            elif i == 8:
                p.font.bold = True
                p.font.color.rgb = IIT_GOLD
            else:
                p.font.color.rgb = DARK_GRAY
    if i == 0:
        for j in range(7):
            table.cell(i, j).fill.solid()
            table.cell(i, j).fill.fore_color.rgb = IIT_BLUE
    elif i == 8:
        for j in range(7):
            table.cell(i, j).fill.solid()
            table.cell(i, j).fill.fore_color.rgb = RGBColor(255, 248, 220)

# Insight box
add_shape(slide, Inches(8.2), Inches(1.4), Inches(4.8), Inches(4.5), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(8.5), Inches(1.6), Inches(4.2), Inches(0.5), "📊 Frontier Analysis", font_size=22, bold=True, color=IIT_GOLD)
insights = [
    "• Asymmetric trade-off:",
    "  Large MRR sacrifice (18%) for",
    "  modest energy gain (8.5%)",
    "",
    "• Dominated points (w₂=0.0, 0.2):",
    "  HIGHER energy than w₂=0.4!",
    "  → pushing speed too high increases",
    "    cutting energy k_c(v_c) more than",
    "    idle energy is saved",
    "",
    "• Speed-dependent k_c is ESSENTIAL:",
    "  without it, frontier collapses",
    "  to a single point",
    "",
    "• Mean over 3 seeds, 50 scenarios each",
]
add_bullet_list(slide, Inches(8.5), Inches(2.2), Inches(4.2), Inches(3.5), insights, font_size=14, color=DARK_GRAY)

# ============================================================
# SLIDE 9: Conditioned Policy
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Conditioned Policy: One Model → Entire Frontier",
    font_size=30, bold=True, color=WHITE)

# Comparison table
table_shape = slide.shapes.add_table(9, 4, Inches(0.5), Inches(1.4), Inches(7), Inches(3.5))
table = table_shape.table
table.columns[0].width = Inches(1.2)
table.columns[1].width = Inches(2.0)
table.columns[2].width = Inches(2.0)
table.columns[3].width = Inches(1.8)

cond_data = [
    ("w₂", "Conditioned 1M", "Per-weight 150k", "Energy gap"),
    ("0.0", "1213.6", "999.9", "−15.9%"),
    ("0.2", "1215.3", "990.5", "−9.5%"),
    ("0.4", "1205.1", "1069.0", "−4.4%"),
    ("0.5", "1191.0", "1026.2", "−4.2%"),
    ("0.6", "1169.3", "1008.4", "−1.0%"),
    ("0.8", "1130.8", "888.3", "+2.0%"),
    ("1.0", "1096.0", "876.6", "+2.6%"),
]

for i, row in enumerate(cond_data):
    for j, val in enumerate(row):
        cell = table.cell(i, j)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(15)
            p.font.name = 'Calibri'
            p.alignment = PP_ALIGN.CENTER
            if i == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
            else:
                p.font.color.rgb = DARK_GRAY
    if i == 0:
        for j in range(4):
            table.cell(i, j).fill.solid()
            table.cell(i, j).fill.fore_color.rgb = IIT_BLUE

# Key message
add_shape(slide, Inches(8), Inches(1.4), Inches(4.8), Inches(3.5), fill_color=RGBColor(230, 245, 230), line_color=GREEN, line_width=Pt(2))
add_text_box(slide, Inches(8.3), Inches(1.6), Inches(4.2), Inches(0.5), "🎯 Result", font_size=24, bold=True, color=GREEN)
msg = [
    "One run of 1M steps beats the 21",
    "policies of 150k steps each (3.15M",
    "steps total): 13–27% more removal rate",
    "for at most 3% more energy.",
    "",
    "The weight stays a runtime query,",
    "not a retrain.",
    "",
    "Observation carries:",
    "  • Energy weight w₂ ∈ [0,1]",
    "  • Material context (C_wear, k₁)",
    "Both drawn per episode at reset().",
]
add_bullet_list(slide, Inches(8.3), Inches(2.3), Inches(4.2), Inches(2.5), msg, font_size=16, color=DARK_GRAY)

# ============================================================
# SLIDE 10: Baseline Comparison
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Baseline Comparison (50 Episodes, Nominal Condition)",
    font_size=30, bold=True, color=WHITE)

# Baseline table
table_shape = slide.shapes.add_table(7, 5, Inches(0.5), Inches(1.4), Inches(8), Inches(3.5))
table = table_shape.table
table.columns[0].width = Inches(2.3)
table.columns[1].width = Inches(1.6)
table.columns[2].width = Inches(1.5)
table.columns[3].width = Inches(1.3)
table.columns[4].width = Inches(1.3)

base_data = [
    ("Policy", "MRR (mm³/min)", "Energy (kJ)", "Cycle (s)", "Reward"),
    ("PPO (primary)", "877.7", "18.5", "5.7", "+204.1"),
    ("Receding-Horizon", "981.7", "18.0", "4.8", "+203.5"),
    ("NSGA-II Static", "732.9", "18.8", "6.4", "−46.4"),
    ("Rule-Based", "719.8", "18.7", "6.6", "−46.5"),
    ("Constant-Safe", "1.6", "1506.8", "3000.0", "−139.3"),
    ("Constant-High", "1099.6", "17.8", "4.3", "−46.1 ⚠ violates"),
]

for i, row in enumerate(base_data):
    for j, val in enumerate(row):
        cell = table.cell(i, j)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(14)
            p.font.name = 'Calibri'
            p.alignment = PP_ALIGN.CENTER
            if i == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
            elif i == 6 and j == 0:
                p.font.bold = True
                p.font.color.rgb = RED
            else:
                p.font.color.rgb = DARK_GRAY
    if i == 0:
        for j in range(5):
            table.cell(i, j).fill.solid()
            table.cell(i, j).fill.fore_color.rgb = IIT_BLUE
    elif i == 6:
        for j in range(5):
            table.cell(i, j).fill.solid()
            table.cell(i, j).fill.fore_color.rgb = RGBColor(255, 230, 230)

# Analysis
add_shape(slide, Inches(0.5), Inches(5.2), Inches(8), Inches(2), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
add_text_box(slide, Inches(0.8), Inches(5.3), Inches(7.5), Inches(0.4), "Analysis", font_size=20, bold=True, color=IIT_BLUE)
analysis = [
    "• Same 50 episodes of the nominal condition for all six policies",
    "• The four static policies end outside the roughness safety limit (reward −46); only the controller stays inside",
    "• PPO +204.1 vs controller +203.5: level on the reward, controller ahead on MRR, energy and cycle time",
    "• No DRL advantage yet: next step is to initialize PPO from the controller → fine-tune",
]
add_bullet_list(slide, Inches(0.8), Inches(5.8), Inches(7.5), Inches(1.2), analysis, font_size=15, color=DARK_GRAY)

# ============================================================
# SLIDE 11: Transfer Experiment
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Transfer Experiment: Cross-Material Adaptation",
    font_size=30, bold=True, color=WHITE)

# Target material box
add_shape(slide, Inches(0.5), Inches(1.4), Inches(6), Inches(2.5), fill_color=RGBColor(255, 240, 240), line_color=RED, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(0.4), "🎯 Target Material (unseen during source training)", font_size=18, bold=True, color=RED)
target_items = [
    "• Harder alloy: k₁ × 1.25, k₂ × 1.30",
    "• Faster-wearing tool: C_wear × 2.0",
    "• Same machine constraints",
    "• Source policy has NEVER seen this material",
]
add_bullet_list(slide, Inches(0.8), Inches(2.0), Inches(5.5), Inches(1.8), target_items, font_size=16, color=DARK_GRAY)

# Three arms
arm_data = [
    ("1. Source Policy\n   Zero-Shot", "183.0", "100.0%", "93.3%"),
    ("2. Source Policy\n   Fine-Tuned (50k)", "182.9", "100.0%", "93.3%"),
    ("3. Target from\n   Scratch (50k)", "196.7", "0.0%", "100.0%"),
]

table_shape = slide.shapes.add_table(4, 4, Inches(7), Inches(1.4), Inches(5.8), Inches(2.5))
table = table_shape.table
table.columns[0].width = Inches(1.8)
table.columns[1].width = Inches(1.3)
table.columns[2].width = Inches(1.3)
table.columns[3].width = Inches(1.4)

arm_header = ["Arm", "Mean Reward", "Ra violations", "Parts done"]
for i, row in enumerate([arm_header] + arm_data):
    for j, val in enumerate(row):
        cell = table.cell(i, j)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(14)
            p.font.name = 'Calibri'
            p.alignment = PP_ALIGN.CENTER
            if i == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
            elif i == 2:
                p.font.bold = True
                p.font.color.rgb = GREEN
            else:
                p.font.color.rgb = DARK_GRAY
    if i == 0:
        for j in range(4):
            table.cell(i, j).fill.solid()
            table.cell(i, j).fill.fore_color.rgb = IIT_BLUE
    elif i == 2:
        for j in range(4):
            table.cell(i, j).fill.solid()
            table.cell(i, j).fill.fore_color.rgb = RGBColor(230, 245, 230)

# Conclusion
add_shape(slide, Inches(0.5), Inches(4.2), Inches(12.3), Inches(2.8), fill_color=RGBColor(255, 240, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(4.3), Inches(11.5), Inches(0.4), "⚠ Fine tuning is the wrong tool here", font_size=22, bold=True, color=IIT_GOLD)
conc_items = [
    "• Fine tuning moves the reward from 183.0 to 182.9 and removes no violation at all",
    "• Training on the harder material from scratch removes every violation (196.7)",
    "• Zero-shot on the harder alloy violates the roughness limit in 100% of scenarios",
    "• The remedy is the one this project adopted: train across a distribution of materials",
    "• Domain randomization is not a nicety here, it is what makes the policy transfer at all",
]
add_bullet_list(slide, Inches(0.8), Inches(4.8), Inches(11.5), Inches(2), conc_items, font_size=15, color=DARK_GRAY)

# ============================================================
# SLIDE 12: Interactive Workstation
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Interactive Digital Twin Workstation (Streamlit)",
    font_size=30, bold=True, color=WHITE)

features = [
    ("📐 2D Stock Depletion", "Cylindrical billet cross-section; pass-by-pass cutout; remaining κ, a_p; tool engagement diagram"),
    ("📊 Micro-Topography", "Feed-mark cusp profile: y_cusp(x) ≈ x²/(8r_ε) + noise; ISO 1302 R_a overlay with 1.6 μm ceiling"),
    ("📝 G-Code Generator", "ISO 6983 Fanuc/Siemens compliant; emits exact PPO parameter sequence; tool/diameter aware"),
    ("⚔️ Multi-Controller", "Side-by-side: PPO vs NSGA-II vs Rule-Based vs Constant-High; live telemetry overlay"),
    ("⚡ Cached Loading", "@st.cache_resource loads 1M-step PPO model once per session"),
]

y = Inches(1.4)
for title, desc in features:
    add_shape(slide, Inches(0.5), y, Inches(12.3), Inches(1.0), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1))
    add_text_box(slide, Inches(0.8), y + Inches(0.1), Inches(2.5), Inches(0.8), title, font_size=18, bold=True, color=IIT_BLUE)
    add_text_box(slide, Inches(3.5), y + Inches(0.1), Inches(9), Inches(0.8), desc, font_size=16, color=DARK_GRAY)
    y += Inches(1.1)

add_text_box(slide, Inches(0.5), Inches(6.8), Inches(12), Inches(0.5),
    "Launch:  streamlit run app.py",
    font_size=20, bold=True, color=IIT_BLUE, alignment=PP_ALIGN.CENTER)

# ============================================================
# SLIDE 13: Software Engineering & Reproducibility
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Software Engineering: Reproducibility & Quality",
    font_size=30, bold=True, color=WHITE)

left_items = [
    "Architecture:",
    "  config/  →  physics, scenarios, paths (single source)",
    "  env/     →  Gymnasium env, 2 registrations",
    "  agent/   →  train, eval, ablation, transfer, pipeline",
    "  baselines/ →  5 classical + receding-horizon",
    "  sim/     →  stock depletion, micro-topography, G-code",
    "  viz/     →  shared plotting utilities",
    "",
    "Reproducibility:",
    "  • Global seeding (Python, NumPy, PyTorch, CUDA)",
    "  • Shared scenario tables (train = eval)",
    "  • summarize_results.py regenerates RESULTS.md",
    "  • All randomness deterministic given seed",
    "  • Centralized paths (config/paths.py), env IDs (env/__init__.py)",
]
right_items = [
    "Testing (26 tests):",
    "  • 19 env tests: clipping, truncation, soft norm,",
    "    physics constants, overrides, energy norm,",
    "    Gym checker, randomization, scenario scales,",
    "    cutting energy speed dependence, part energy",
    "    interior minimum, shared physics, shared tables",
    "  • 2 reward tests: golden-value (1e-9), finishing penalty",
    "  • 3 integration: train→eval roundtrip, conditioned obs,",
    "    baseline artifact",
    "  • 3 callback: heartbeat delivery, failure report, interval",
    "",
    "Code Quality:",
    "  • No unused imports (AST verified)",
    "  • Conventional commits, clean git history",
    "  • Lit citations in physics_constants.py",
]

add_two_column(slide, Inches(0.5), Inches(1.3), Inches(12.3), Inches(5.8),
    left_items, right_items, font_size=14, color=DARK_GRAY,
    title_left="Architecture & Reproducibility", title_right="Test Coverage & Quality")

# ============================================================
# SLIDE 14: Discussion
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Discussion: What Worked, Limitations, Future Work",
    font_size=30, bold=True, color=WHITE)

# Three columns
col_w = Inches(3.9)
titles = ["✅ What Worked", "⚠️ Limitations", "🚀 Future Work"]
contents = [
    [
        "Speed-dependent k_c creates genuine trade-off",
        "Domain randomization → 99% completion, 0% finished violations",
        "Conditioned policy: 1 run replaces 21 (beats them at 1/3 the steps)",
        "Receding-horizon = strong, fair baseline (same physics)",
        "Test suite catches physics & reward regressions",
        "Full reproducibility: one script regenerates all tables",
    ],
    [
        "PPO is level with the receding-horizon, not ahead of it",
        "Ablation uses 150k steps/weight (vs 1M primary)",
        "The reward charges a pass, the metric is a part",
        "Micro-topography is simplified (no chatter/vibration)",
        "Sim-only — no real machine calibration yet",
    ],
    [
        "Controller-initialized RL (BC + fine-tune)",
        "Sim-to-real: calibrate on machine data, add noise/delay",
        "True multi-objective RL (Pareto-conditioned, hypervolume)",
        "Online adaptation: runtime w₂ optimizer for energy budget",
        "Full 3D Digital Twin: toolpath viz, collision checking",
        "Deploy on edge hardware for real-time control",
    ],
]

x = Inches(0.4)
for title, items in zip(titles, contents):
    add_shape(slide, x, Inches(1.3), col_w, Inches(5.8), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
    add_text_box(slide, x + Inches(0.2), Inches(1.4), col_w - Inches(0.4), Inches(0.5), title, font_size=20, bold=True, color=IIT_BLUE)
    add_bullet_list(slide, x + Inches(0.2), Inches(2.0), col_w - Inches(0.4), Inches(4.8), items, font_size=15, color=DARK_GRAY, spacing=Pt(8))
    x += col_w + Inches(0.15)

# ============================================================
# SLIDE 15: Conclusion
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, IIT_BLUE)
add_shape(slide, Inches(0.5), Inches(0.8), Inches(12.3), Pt(4), fill_color=IIT_GOLD)

add_text_box(slide, Inches(1), Inches(1.2), Inches(11.333), Inches(1),
    "Conclusion",
    font_size=40, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

conclusions = [
    "🏭 Physics-grounded Digital Twin with speed-dependent cutting energy, tool wear, roughness, stochastic breakage",
    "🤖 Gymnasium environment with domain randomization, conditioned observations, energy-aware reward",
    "📊 PPO policies with rigorous evaluation: Monte Carlo (100 scenarios, 95% CIs), ablation frontier, conditioned policy",
    "⚔️ Strong classical baselines including receding-horizon controller (re-optimizes every pass)",
    "🔄 Transfer experiment: fine tuning does not adapt, training across materials does",
    "🖥️ Interactive workstation: 2D stock depletion, micro-topography, G-code, multi-controller comparison",
    "🔁 Full reproducibility: every result table generated from committed artifacts by one script",
]

y = Inches(2.4)
for c in conclusions:
    add_text_box(slide, Inches(1), y, Inches(11.333), Inches(0.55), c, font_size=20, color=WHITE, alignment=PP_ALIGN.LEFT)
    y += Inches(0.62)

add_shape(slide, Inches(1), Inches(6.8), Inches(11.333), Pt(4), fill_color=IIT_GOLD)
add_text_box(slide, Inches(1), Inches(6.9), Inches(11.333), Inches(0.5),
    "Key Scientific Finding: The speed-dependent specific cutting energy creates a genuine energy/throughput trade-off, and DRL learns to navigate it pass-by-pass while respecting constraints. Across the measured frontier the removal rate falls 18% while the energy of the part falls 8.5%.",
    font_size=18, bold=True, color=IIT_GOLD, alignment=PP_ALIGN.CENTER)

# ============================================================
# SLIDE 16: Reproducibility Commands
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Reproducibility: All Commands Run from Project Root",
    font_size=30, bold=True, color=WHITE)

cmds = [
    ("Tests", "pytest tests/ -v"),
    ("Train Primary PPO", "python agent/train_rl.py --algo PPO --steps 1000000 --seed 42 --randomize"),
    ("Train Conditioned", "python agent/train_rl_conditioned.py --steps 1000000 --seed 42"),
    ("Monte Carlo Eval", "python agent/evaluate.py"),
    ("Ablation Study", "python agent/train_rl_ablation.py --steps 150000 --scenarios 50 --slots 7"),
    ("Classical Baselines", "python baselines/run_baselines.py"),
    ("Transfer Experiment", "python agent/transfer_experiment.py"),
    ("Regenerate Plots", "python viz/plot_eval.py && python viz/plot_results.py && python viz/plot_baselines_comparison.py"),
    ("Regenerate RESULTS.md", "python agent/summarize_results.py"),
    ("Launch Workstation", "streamlit run app.py"),
    ("Full Pipeline", "python agent/run_pipeline.py --steps 1000000 --ablation_steps 100000 --scenarios 50 --slots 7"),
]

y = Inches(1.4)
for name, cmd in cmds:
    add_shape(slide, Inches(0.5), y, Inches(12.3), Inches(0.45), fill_color=LIGHT_GRAY if cmds.index((name, cmd)) % 2 == 0 else WHITE)
    add_text_box(slide, Inches(0.7), y + Inches(0.02), Inches(2.5), Inches(0.4), name, font_size=15, bold=True, color=IIT_BLUE)
    add_text_box(slide, Inches(3.3), y + Inches(0.02), Inches(9.3), Inches(0.4), cmd, font_size=14, color=DARK_GRAY, font_name='Consolas')
    y += Inches(0.5)

# ============================================================
# SLIDE 17: Thank You / Questions
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, IIT_BLUE)
add_shape(slide, Inches(1), Inches(2.5), Inches(11.333), Pt(4), fill_color=IIT_GOLD)

add_text_box(slide, Inches(1), Inches(1.5), Inches(11.333), Inches(1),
    "Thank You",
    font_size=50, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(1), Inches(2.8), Inches(11.333), Inches(0.8),
    "Questions & Discussion",
    font_size=36, color=IIT_GOLD, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(1), Inches(4.0), Inches(11.333), Inches(0.6),
    "Repository: G:/My Drive/IITM/AI in Mfg",
    font_size=20, color=WHITE, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(1), Inches(4.8), Inches(11.333), Inches(0.6),
    "All code, data, results, and documentation in the repository",
    font_size=18, color=RGBColor(180, 200, 220), alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(1), Inches(5.8), Inches(11.333), Inches(0.6),
    "ME6324 — AI in Manufacturing  •  IIT Madras",
    font_size=18, color=IIT_GOLD, alignment=PP_ALIGN.CENTER)

# Save
output_path = r"G:\My Drive\IITM\AI in Mfg\report\presentation.pptx"
prs.save(output_path)
print(f"Presentation saved to {output_path}")
print(f"Slides: {len(prs.slides)}")