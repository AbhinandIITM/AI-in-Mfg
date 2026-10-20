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

def add_two_column(slide, left, top, width, height, left_items, right_items, font_size=15, color=DARK_GRAY, title_left="", title_right="", spacing=Pt(6)):
    col_width = width / 2 - Inches(0.15)
    # Left column
    if title_left:
        add_text_box(slide, left, top, col_width, Inches(0.4), title_left, font_size=font_size+2, bold=True, color=IIT_BLUE)
        top_content = top + Inches(0.45)
    else:
        top_content = top
    add_bullet_list(slide, left, top_content, col_width, height - Inches(0.5), left_items, font_size=font_size, color=color, spacing=spacing)
    # Right column
    right_left = left + col_width + Inches(0.3)
    if title_right:
        add_text_box(slide, right_left, top, col_width, Inches(0.4), title_right, font_size=font_size+2, bold=True, color=IIT_BLUE)
    add_bullet_list(slide, right_left, top_content, col_width, height - Inches(0.5), right_items, font_size=font_size, color=color, spacing=spacing)

def style_table(table, header_rows=1, font_size=13, header_size=None, center=True, highlight_rows=()):
    """Apply the deck's table look: blue header, dark body text."""
    header_size = header_size or font_size
    for i, row in enumerate(table.rows):
        for j, cell in enumerate(row.cells):
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(header_size if i < header_rows else font_size)
                p.font.name = 'Calibri'
                if center:
                    p.alignment = PP_ALIGN.CENTER
                if i < header_rows:
                    p.font.bold = True
                    p.font.color.rgb = WHITE
                elif i in highlight_rows:
                    p.font.bold = True
                    p.font.color.rgb = IIT_BLUE
                else:
                    p.font.color.rgb = DARK_GRAY
            if i < header_rows:
                cell.fill.solid()
                cell.fill.fore_color.rgb = IIT_BLUE
            elif i in highlight_rows:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(255, 248, 220)

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
    "ME6324, AI in Manufacturing  |  Course Project Report",
    font_size=22, color=IIT_GOLD, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(1), Inches(4.2), Inches(11.333), Inches(0.6),
    "Abhinand  •  Department of Mechanical Engineering  •  IIT Madras",
    font_size=18, color=WHITE, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(1), Inches(5.2), Inches(11.333), Inches(0.6),
    "Git Revision: d76603b  •  All results reproducible from committed artifacts",
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
        "One Gymnasium environment (11-D obs, 3-D action)",
        "Energy-aware reward: 5·(w₁·MRR − w₂·Energy) + progress − penalties",
        "Conditioned policy: one model covers the full frontier",
        "PPO with VecNormalize, 4 parallel envs, CPU training",
    ],
    [
        "100-scenario Monte Carlo with 95% CIs",
        "Ablation: 7 energy weights × 3 seeds, weight pinned",
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
# SLIDE 4: Related Work
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Positioning Against the Reference Papers",
    font_size=30, bold=True, color=WHITE)

rel_data = [
    ("Capability", "Mo et al. (2025)\nJ. Manuf. Process 156", "Wang et al. (2025a)\nJ. Manuf. Syst. 78", "This Work"),
    ("Dynamic optimization", "✓", "✓", "✓"),
    ("Tool wear in the state", "✓", "✗", "✓"),
    ("Roughness constraint", "✓", "✗", "✓"),
    ("Energy as an objective", "✗", "✗", "✓"),
    ("Speed-dependent k_c", "✗", "✗", "✓"),
    ("Domain randomization", "✗", "✗", "✓"),
    ("Conditioned policy", "✗", "✗", "✓"),
    ("Transfer experiment", "✗", "✗", "✓"),
    ("Receding-horizon baseline", "✗", "✗", "✓"),
    ("Monte Carlo (100 scenarios)", "✗", "✗", "✓"),
    ("Interactive workstation", "✗", "✗", "✓"),
]

rel_table = slide.shapes.add_table(len(rel_data), 4, Inches(0.5), Inches(1.35), Inches(12.3), Inches(3.6)).table
for j, w in enumerate([4.7, 2.6, 2.7, 2.3]):
    rel_table.columns[j].width = Inches(w)
for i, row in enumerate(rel_data):
    for j, val in enumerate(row):
        rel_table.cell(i, j).text = val
style_table(rel_table, font_size=13)
for j in range(4):
    rel_table.cell(0, j).fill.solid()
    rel_table.cell(0, j).fill.fore_color.rgb = IIT_BLUE

add_text_box(slide, Inches(0.5), Inches(5.15), Inches(12.3), Inches(0.9),
    "Mo et al.: PPO over a mixed action space (tool change + parameters); the state carries V_B, breakage and R_a; objectives are quality and efficiency.\n"
    "Wang et al.: MATD3 for batch milling; objectives are dimensional quality and machining time.",
    font_size=14, color=DARK_GRAY)

add_shape(slide, Inches(0.5), Inches(6.15), Inches(12.3), Inches(0.95), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(6.25), Inches(11.7), Inches(0.75),
    "Key differentiator: neither reference treats energy as an objective. The speed-dependent k_c is what creates a genuine "
    "energy/throughput trade-off, and one conditioned policy covers the whole frontier in a single run.",
    font_size=15, bold=True, color=IIT_GOLD)

# ============================================================
# SLIDE 5: Physics - Key Equations
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Digital Twin Physics (MillingDT/config/physics.py, single source of truth)",
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
# SLIDE 6: Environment Design
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "One Gymnasium Environment: MillingDTEnv-v1",
    font_size=30, bold=True, color=WHITE)

# Two columns
left_items = [
    "One registration:",
    "  • MillingDTEnv-v1 (env.ENV_ID), 200-step limit",
    "  • The weight and the material are inputs, not tasks",
    "",
    "Observation (11-D, always the same shape):",
    "  0-7: V_B, κ_rem, prev n, f_r, a_p, MRR/MRR_max,",
    "       R_a, tool-broken flag",
    "  8-10: the context: w₂, C_wear [0.6,1.4], k₁ [0.8,1.2]",
    "",
    "Action: continuous [-1,1]³ → clipped → machine envelope",
]
right_items = [
    "Reward per step:",
    "  r = 5·(w₁·MRR/MRR_max − w₂·E_pass/E_ref)",
    "       + 2·Δ(stock removed)",
    "       − 50·1[V_B > V_B_safe] − 100·1[broken]",
    "       − 50·1[finishing pass ∧ R_a > R_a_safe] − 0.05",
    "  +200 for a part finished in spec, +≤5 Pareto margin",
    "  w₁ = 1 − w₂; safety limits 0.255 mm and 1.36 μm",
    "  breakage p = 0.3·wear factor·(load factor)² above 0.15 mm",
    "",
    "Domain Randomization (--randomize):",
    "  • Workpiece: κ_total [2,15], D [40,150], L_cut [50,200]",
    "  • Material/Tool: C_wear [0.6,1.4]×, k₁,k₂,k₃ [0.8,1.2]× nominal",
    "  • Initial wear: V_B0 ~ U[0, 0.8]·V_B_max",
    "  • Energy weight drawn U[0,1] per episode",
    "  • Same tables for training + Monte Carlo eval",
    "",
    "E_ref recomputed per scenario from its own D, L_cut, k_c",
]

add_two_column(slide, Inches(0.5), Inches(1.3), Inches(12.3), Inches(5.9),
    left_items, right_items, font_size=14, color=DARK_GRAY, spacing=Pt(4),
    title_left="State & Action", title_right="Reward & Randomization")

# ============================================================
# SLIDE 7: RL Training Setup
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
    "1️⃣ Production policy: 1M steps, seed 42",
    "    --randomize --randomize-weight",
    "    → results/run_PPO_seed42/",
    "",
    "2️⃣ Ablation grid: 7 weights × 3 seeds = 21 policies",
    "    w₂ ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}",
    "    seeds {42, 43, 44}, 150k steps each",
    "    weight PINNED at training, evaluated at that",
    "    same weight",
    "",
    "Heartbeat callback: queue signal every 1000 steps",
    "→ stall detection in run_experiments.py",
]
add_bullet_list(slide, Inches(7.3), Inches(2.2), Inches(5.2), Inches(4.5), runs, font_size=15, color=DARK_GRAY)

# ============================================================
# SLIDE 8: Monte Carlo Results
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Results: Production Policy, Monte Carlo Sweep (100 Scenarios)",
    font_size=30, bold=True, color=WHITE)

# Main metrics table
table_shape = slide.shapes.add_table(6, 4, Inches(0.35), Inches(1.4), Inches(6.6), Inches(2.0))
table = table_shape.table
for j, w in enumerate([2.9, 1.2, 1.2, 1.3]):
    table.columns[j].width = Inches(w)

mc_data = [
    ("Metric", "Mean", "Std", "95% CI"),
    ("MRR (mm³/min)", "1205.2", "585.7", "[1088.9, 1321.4]"),
    ("Energy/part (kJ)", "68.4", "48.3", "[58.8, 77.9]"),
    ("Cycle time (s)", "20.3", "17.1", "[16.9, 23.6]"),
    ("Max R_a (μm)", "1.379", "0.350", "[1.309, 1.448]"),
    ("Max V_B (mm)", "0.1244", "0.0679", "[0.1109, 0.1378]"),
]

for i, row in enumerate(mc_data):
    for j, val in enumerate(row):
        table.cell(i, j).text = val
style_table(table, font_size=13)

# Constraint violations
vio_table = slide.shapes.add_table(8, 3, Inches(7.1), Inches(1.4), Inches(5.9), Inches(2.6)).table
for j, w in enumerate([3.4, 1.0, 1.5]):
    vio_table.columns[j].width = Inches(w)

vio_data = [
    ("Constraint", "Rate", "95% CI"),
    ("R_a viol., worst pass", "38.0%", "[28.5, 47.5]%"),
    ("R_a viol., finished surface", "0.0%", "[0, 0]%"),
    ("R_a above safety limit, worst", "56.0%", "[46.3, 65.7]%"),
    ("R_a above safety limit, finished", "0.0%", "[0, 0]%"),
    ("Flank wear violations (V_B > 0.3)", "0.0%", "[0, 0]%"),
    ("Tool breakage", "1.0%", "[0, 3]%"),
    ("Parts completed in spec", "99.0%", ""),
]

for i, row in enumerate(vio_data):
    for j, val in enumerate(row):
        vio_table.cell(i, j).text = val
style_table(vio_table, font_size=13)

# Key takeaway box
add_shape(slide, Inches(0.5), Inches(4.35), Inches(12.3), Inches(2.6), fill_color=RGBColor(230, 245, 230), line_color=GREEN, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(4.5), Inches(11.7), Inches(0.5), "✅ Key Findings", font_size=22, bold=True, color=GREEN)
takeaways = [
    "• Finished surface ALWAYS within spec (0% violations), the part quality holds",
    "• Flank wear NEVER exceeded (0% violations); 99% of the parts completed",
    "• 56% of the passes exceed the roughness safety margin (1.36 μm): the policy pushes and recovers by the finish pass",
    "• High MRR variance (std = 49% of mean) → domain randomization works: diverse operating points",
    "• Mean operating point: 7199 RPM, 0.247 mm/r, 2.11 mm, 4.0 passes",
]
add_bullet_list(slide, Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.7), takeaways, font_size=15, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE 9: Ablation, Pareto Frontier
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Energy Weight Ablation: Empirical Pareto Frontier",
    font_size=30, bold=True, color=WHITE)

# Ablation table
table_shape = slide.shapes.add_table(8, 7, Inches(0.5), Inches(1.4), Inches(12.3), Inches(2.7))
table = table_shape.table
cols = [1.1, 2.3, 2.0, 1.7, 1.7, 1.5, 2.0]
for i, w in enumerate(cols):
    table.columns[i].width = Inches(w)

abl_data = [
    ("w₂", "MRR (mm³/min)", "Energy (kJ)", "Cycle (s)", "n (RPM)", "f_r (mm/r)", "Frontier"),
    ("0.0", "946.3±63.4", "82.9±1.4", "44.8", "7094", "0.166", "✗ dominated"),
    ("0.2", "1007.8±22.7", "77.1±1.1", "31.3", "7471", "0.184", "✗ dominated"),
    ("0.4", "1038.0±67.6", "72.7±0.5", "24.8", "7256", "0.203", "✓"),
    ("0.5", "1035.8±43.8", "70.3±1.2", "22.0", "7080", "0.214", "✓"),
    ("0.6", "1023.9±27.4", "70.8±0.6", "22.1", "7148", "0.209", "✗ dominated"),
    ("0.8", "971.1±43.2", "66.8±1.2", "21.9", "6538", "0.228", "✓"),
    ("1.0", "951.3±54.7", "65.8±0.6", "20.9", "6490", "0.229", "✓"),
]

for i, row in enumerate(abl_data):
    for j, val in enumerate(row):
        table.cell(i, j).text = val
style_table(table, font_size=13)

# Insight box
add_shape(slide, Inches(0.5), Inches(4.35), Inches(12.3), Inches(2.6), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(4.5), Inches(11.7), Inches(0.5), "📊 Frontier Analysis", font_size=22, bold=True, color=IIT_GOLD)
insights = [
    "• From w₂ = 0.4 to w₂ = 1.0 the removal rate falls 8.4% (1038 → 951 mm³/min) and the energy falls 9.5% (72.7 → 65.8 kJ): a symmetric trade-off",
    "• The spindle speed falls with the energy weight (7256 → 6490 RPM): the energy is bought with throughput, not with time",
    "• Dominated points (w₂ = 0.0, 0.2, 0.6) remove less for more",
    "• Each policy has its weight pinned at training and evaluated at that same weight; 3 seeds, 50 scenarios each",
]
add_bullet_list(slide, Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.7), insights, font_size=14, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE 10: One Policy Against 21 Policies
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "One Policy Against 21 Policies",
    font_size=30, bold=True, color=WHITE)

# Comparison table
table_shape = slide.shapes.add_table(8, 7, Inches(0.75), Inches(1.4), Inches(11.8), Inches(2.7))
table = table_shape.table
for j, w in enumerate([1.1, 1.9, 1.9, 1.5, 1.9, 1.9, 1.6]):
    table.columns[j].width = Inches(w)

cond_data = [
    ("w₂", "MRR, one policy", "MRR, per weight", "MRR gap", "Energy, one policy", "Energy, per weight", "Energy gap"),
    ("0.0", "1213.6", "946.3", "+28.2%", "69.6", "82.9", "−16.0%"),
    ("0.2", "1215.3", "1007.8", "+20.6%", "69.0", "77.1", "−10.5%"),
    ("0.4", "1205.1", "1038.0", "+16.1%", "68.3", "72.7", "−6.0%"),
    ("0.5", "1191.0", "1035.8", "+15.0%", "68.0", "70.3", "−3.3%"),
    ("0.6", "1169.3", "1023.9", "+14.2%", "67.6", "70.8", "−4.5%"),
    ("0.8", "1130.8", "971.1", "+16.4%", "67.1", "66.8", "+0.5%"),
    ("1.0", "1096.0", "951.3", "+15.2%", "67.1", "65.8", "+2.1%"),
]

for i, row in enumerate(cond_data):
    for j, val in enumerate(row):
        table.cell(i, j).text = val
style_table(table, font_size=13)

# Key message
add_shape(slide, Inches(0.5), Inches(4.35), Inches(12.3), Inches(2.6), fill_color=RGBColor(230, 245, 230), line_color=GREEN, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(4.5), Inches(11.7), Inches(0.5), "🎯 Result", font_size=22, bold=True, color=GREEN)
msg = [
    "• For w₂ from 0.0 to 0.6 the single policy removes 14% to 28% more material for 3% to 16% less energy",
    "• At w₂ = 0.8 and w₂ = 1.0 it still removes 16% and 15% more for 0.5% and 2.1% more energy",
    "• The weight stays a runtime query, not a retrain: the observation carries w₂ and the material context (C_wear, k₁)",
    "• The same policy swept at runtime: MRR 1213.6 (w₂ = 0.0) → 1096.0 (w₂ = 1.0), energy 69.6 → 67.1 kJ, spindle 7513 → 6475 RPM",
]
add_bullet_list(slide, Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.7), msg, font_size=15, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE 11: Baseline Comparison
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Baseline Comparison (50 Episodes, Nominal Condition, w₂ = 0.4)",
    font_size=30, bold=True, color=WHITE)

# Baseline table
table_shape = slide.shapes.add_table(7, 5, Inches(1.2), Inches(1.4), Inches(10.8), Inches(2.4))
table = table_shape.table
for j, w in enumerate([3.0, 1.8, 2.2, 2.0, 1.8]):
    table.columns[j].width = Inches(w)

base_data = [
    ("Policy", "Reward", "MRR (mm³/min)", "Energy (kJ)", "Cycle (s)"),
    ("PPO (ours)", "204.09", "894.3", "18.40", "5.56"),
    ("Receding-Horizon", "203.45", "981.7", "18.02", "4.80"),
    ("NSGA-II Static", "−46.39", "732.9", "18.8", "6.4"),
    ("Rule-Based", "−46.48", "719.8", "18.7", "6.6"),
    ("Constant-High", "−46.10", "1099.6", "17.8", "4.3  ⚠ violates"),
    ("Constant-Safe", "−139.25", "1.6", "1506.8", "3000.0"),
]

for i, row in enumerate(base_data):
    for j, val in enumerate(row):
        table.cell(i, j).text = val
style_table(table, font_size=14)
for j in range(5):
    table.cell(6, j).fill.solid()
    table.cell(6, j).fill.fore_color.rgb = RGBColor(255, 230, 230)
    table.cell(5, j).fill.solid()
    table.cell(5, j).fill.fore_color.rgb = RGBColor(255, 248, 220)

# Analysis
add_shape(slide, Inches(0.5), Inches(4.15), Inches(12.3), Inches(2.8), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
add_text_box(slide, Inches(0.8), Inches(4.3), Inches(11.7), Inches(0.4), "Analysis", font_size=20, bold=True, color=IIT_BLUE)
analysis = [
    "• Same 50 episodes of the nominal condition for all six policies, at w₂ = 0.4",
    "• Level on the reward: PPO 204.1 against the controller 203.5",
    "• The controller is ahead on the removal rate (981.7 vs 894.3) and the cycle time (4.80 s vs 5.56 s)",
    "• No reinforcement learning advantage is demonstrated yet",
    "• The three fast static policies finish outside the roughness safety limit, so the reward takes the −50 finishing penalty; Constant-Safe stays inside but needs 3000 s",
]
add_bullet_list(slide, Inches(0.8), Inches(4.85), Inches(11.7), Inches(2.0), analysis, font_size=15, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE 12: Controller Initialization, Method and Result
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Controller Initialization: Method and Result",
    font_size=30, bold=True, color=WHITE)

add_text_box(slide, Inches(0.5), Inches(1.25), Inches(12.3), Inches(0.5),
    "Question: does a policy that starts from the controller end up better than the expert it started from?",
    font_size=16, bold=True, color=RED)

# Method box
add_shape(slide, Inches(0.5), Inches(1.9), Inches(5.5), Inches(4.9), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
add_text_box(slide, Inches(0.8), Inches(2.05), Inches(5.0), Inches(0.5), "Method", font_size=22, bold=True, color=IIT_BLUE)
method_items = [
    "1. Record the controller on 2000 episodes",
    "",
    "2. Fit a supervised actor (2x256, Tanh) by",
    "   regression on those trajectories",
    "",
    "3. Copy the fitted weights into a fresh PPO policy",
    "",
    "4. Train with the production budget (1M steps),",
    "   evaluate on 100 scenarios at w₂ = 0.4",
    "",
    "MillingDT/agent/controller_init.py",
    "experiments/controller_experiment.py",
]
add_bullet_list(slide, Inches(0.8), Inches(2.65), Inches(5.0), Inches(4.0), method_items, font_size=14, color=DARK_GRAY, spacing=Pt(3))

# Three arms table
ci_data = [
    ("Metric", "Receding-horizon\ncontroller", "PPO\nfrom scratch", "PPO initialized\nfrom controller"),
    ("Mean episode reward", "187.96", "198.57", "199.35"),
    ("Reward std", "65.92", "30.03", "3.35"),
    ("MRR (mm³/min)", "1560.6", "1205.1", "1126.1"),
    ("Energy per part (kJ)", "68.01", "68.35", "77.17"),
    ("Cycle time (s)", "13.54", "20.25", "32.35"),
    ("Ra viol., finished", "0.0%", "0.0%", "0.0%"),
    ("Tool breakage", "5%", "1%", "0%"),
    ("Parts completed", "95%", "99%", "100%"),
]

table_shape = slide.shapes.add_table(len(ci_data), 4, Inches(6.3), Inches(1.9), Inches(6.5), Inches(3.6))
table = table_shape.table
for j, w in enumerate([2.1, 1.5, 1.4, 1.5]):
    table.columns[j].width = Inches(w)

for i, row in enumerate(ci_data):
    for j, val in enumerate(row):
        table.cell(i, j).text = val
style_table(table, font_size=12)

# Reading box under the table
add_shape(slide, Inches(6.3), Inches(5.65), Inches(6.5), Inches(1.15), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(6.55), Inches(5.75), Inches(6.0), Inches(0.4), "Same 100 scenarios, same energy weight w₂ = 0.4", font_size=15, bold=True, color=IIT_GOLD)
add_bullet_list(slide, Inches(6.55), Inches(6.15), Inches(6.0), Inches(0.6), [
    "All three arms keep the finished surface in spec (0% violations)",
], font_size=13, color=DARK_GRAY, spacing=Pt(2))

# ============================================================
# SLIDE 13: Controller Initialization, What It Shows
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Controller Initialization: What It Shows",
    font_size=30, bold=True, color=WHITE)

# Left: no advantage
add_shape(slide, Inches(0.5), Inches(1.35), Inches(6.05), Inches(3.6), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(0.5), "No advantage over the controller", font_size=20, bold=True, color=IIT_GOLD)
add_bullet_list(slide, Inches(0.8), Inches(2.1), Inches(5.5), Inches(2.7), [
    "• The controller still removes 38% more material per minute (1560.6 against 1126.1 mm³/min)",
    "• and finishes in 13.5 s against 32.3 s",
    "• The reward cannot separate the two learned policies: 199.35 against 198.57 is a gap of 0.8, while the standard error of the mean over 100 scenarios is 30.0/√100 = 3.0",
], font_size=15, color=DARK_GRAY, spacing=Pt(6))

# Right: consistency
add_shape(slide, Inches(6.8), Inches(1.35), Inches(6.05), Inches(3.6), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
add_text_box(slide, Inches(7.1), Inches(1.5), Inches(5.5), Inches(0.8), "Consistency, paid for with throughput and energy", font_size=20, bold=True, color=IIT_BLUE)
add_bullet_list(slide, Inches(7.1), Inches(2.1), Inches(5.5), Inches(2.7), [
    "• Reward std across the 100 scenarios falls from 30.0 to 3.4, tool breakage falls to zero, every part is completed",
    "• Against the from-scratch policy: 6.6% less removal rate and 12.9% more energy per part",
    "• The controller is the least reliable arm of the three: std 65.9, 5% breakage, 95% of the parts completed",
], font_size=15, color=DARK_GRAY, spacing=Pt(6))

# Claim box
add_shape(slide, Inches(0.5), Inches(5.2), Inches(12.35), Inches(1.6), fill_color=RGBColor(230, 240, 250), line_color=IIT_BLUE, line_width=Pt(1.5))
add_text_box(slide, Inches(0.8), Inches(5.35), Inches(11.7), Inches(0.5), "What the work claims", font_size=20, bold=True, color=IIT_BLUE)
add_bullet_list(slide, Inches(0.8), Inches(5.9), Inches(11.7), Inches(0.8), [
    "The reinforcement learning agent navigates the energy/throughput trade-off inside a fixed constraint set. It does not beat a classical controller that re-optimizes every pass with the exact physics.",
], font_size=16, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE 14: Transfer Experiment
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Transfer Experiment: Cross-Material Adaptation",
    font_size=30, bold=True, color=WHITE)

add_text_box(slide, Inches(0.5), Inches(1.3), Inches(12.3), Inches(0.8),
    "🎯 Target material (harder alloy, same machine): k₁ × 1.25, k₂ × 1.30, C_wear × 2.0. "
    "The source policy was trained for 300,000 steps on the nominal material; the adaptation budget on the target is 50,000 steps.",
    font_size=15, bold=True, color=RED)

# Four arms
arm_data = [
    ("Arm", "Mean reward", "R_a violations\n(worst pass)", "R_a violations\n(finished surface)", "Parts\ncompleted"),
    ("Source policy, zero shot", "182.5", "100.0%", "3.3%", "93.3%"),
    ("Source policy, fine tuned (50k)", "196.6", "0.0%", "0.0%", "100.0%"),
    ("Trained on the target from scratch (50k)", "196.6", "0.0%", "0.0%", "100.0%"),
    ("Production policy, zero shot", "200.3", "60.0%", "0.0%", "100.0%"),
]

table_shape = slide.shapes.add_table(5, 5, Inches(0.7), Inches(2.2), Inches(11.9), Inches(1.9))
table = table_shape.table
for j, w in enumerate([4.4, 1.7, 2.0, 2.2, 1.6]):
    table.columns[j].width = Inches(w)

for i, row in enumerate(arm_data):
    for j, val in enumerate(row):
        table.cell(i, j).text = val
style_table(table, font_size=13, highlight_rows=(4,))

# Conclusion
add_shape(slide, Inches(0.5), Inches(4.35), Inches(12.3), Inches(2.6), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(4.5), Inches(11.7), Inches(0.5), "⚠ How to read this", font_size=22, bold=True, color=IIT_GOLD)
conc_items = [
    "• The production policy was trained across a distribution of materials with the material context in its observation, so the harder material is inside its training distribution and needs no adaptation at all",
    "• It scores the highest reward (200.3); the finished surface, which is the part quality, is in spec in every scenario",
    "• Its 60% worst-pass violations fall on the roughing passes, which the reward does not constrain",
    "• The nominal-trained source policy cannot: 182.5, and it violates the roughness limit in every scenario",
    "• Fine tuning it for 50,000 steps, or training on the target from scratch, both fix it, and both land at 196.6",
]
add_bullet_list(slide, Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.8), conc_items, font_size=14, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE 15: Interactive Workstation
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
# SLIDE 16: Software Engineering & Reproducibility
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Software Engineering: Reproducibility & Quality",
    font_size=30, bold=True, color=WHITE)

left_items = [
    "Architecture:",
    "  MillingDT/  →  config, env, agent (controller init),",
    "                  baselines, sim, utils",
    "  experiments/ →  train, eval, ablation, baselines,",
    "                  transfer, controller experiment, pipeline",
    "  viz/         →  shared plotting utilities",
    "  results/     →  one directory per run (models, log,",
    "                  metrics, plots)",
    "",
    "Reproducibility:",
    "  • Global seeding (Python, NumPy, PyTorch, CUDA)",
    "  • Shared scenario tables (train = eval)",
    "  • experiments/summarize_results.py regenerates RESULTS.md",
    "  • All randomness deterministic given seed",
    "  • Centralized constants and paths (MillingDT/config/)",
]
right_items = [
    "Testing (34 tests):",
    "  • 22 env tests: clipping, truncation, soft norm, 11-D",
    "    observation, constants, overrides, accumulators, energy",
    "    norm, Gym checker, randomization, pinned values, scenario",
    "    scales, context (weight, material, random weight), speed",
    "    dependence, energy minimum, shared physics",
    "  • 2 reward tests: golden values (w₂ = 0.0, 0.4, 0.75),",
    "    finishing penalty",
    "  • 5 integration: train→eval roundtrip, one registration,",
    "    context in the observation, one policy at a pinned weight,",
    "    baseline artifact",
    "  • 3 callback: heartbeat delivery, failure report, interval",
    "  • 2 training: convergence, reproducibility",
    "",
    "Code Quality:",
    "  • No unused imports (AST verified)",
    "  • Conventional commits, clean git history",
    "  • Constants and paths in MillingDT/config/",
]

add_two_column(slide, Inches(0.5), Inches(1.3), Inches(12.3), Inches(5.9),
    left_items, right_items, font_size=14, color=DARK_GRAY, spacing=Pt(4),
    title_left="Architecture & Reproducibility", title_right="Test Coverage & Quality")

# ============================================================
# SLIDE 17: Discussion
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
        "One policy beats the 21 per-weight policies",
        "Training across materials makes the adaptation unnecessary",
        "Receding-horizon = strong, fair baseline (same physics)",
        "Controller initialization buys consistency: reward std 30.0 → 3.4, no tool breakage",
        "Full reproducibility: one script regenerates all tables",
    ],
    [
        "PPO is level with the receding-horizon, not ahead of it",
        "Initializing from the controller does not close the gap: more uniform, but 6.6% less removal rate and 12.9% more energy",
        "Ablation uses 150k steps/weight (vs 1M production)",
        "The reward charges a pass, the metric is a part",
        "Micro-topography is simplified (no chatter/vibration)",
        "Sim-only, no real machine calibration yet",
    ],
    [
        "Compare against a restricted controller (is the gap an information gap?)",
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
# SLIDE 18: Conclusion
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, IIT_BLUE)
add_shape(slide, Inches(0.5), Inches(0.8), Inches(12.3), Pt(4), fill_color=IIT_GOLD)

add_text_box(slide, Inches(1), Inches(1.2), Inches(11.333), Inches(1),
    "Conclusion",
    font_size=40, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

conclusions = [
    "🏭 Physics-grounded Digital Twin: speed-dependent cutting energy, tool wear, roughness, breakage",
    "🤖 One environment (MillingDTEnv-v1): domain randomization, 11-D observation with the context",
    "📊 One production policy (1M steps): the energy weight is a runtime query, not a retrain",
    "🔬 Monte Carlo (100 scenarios, 95% CIs), ablation frontier, sweep vs 21 per-weight policies",
    "⚔️ Six baselines on the same 50 episodes, level on the reward, no RL advantage yet",
    "🧬 Controller init: more uniform (std 3.4 vs 30.0), no tool breakage, no gain on throughput",
    "🔄 Transfer: the production policy needs no adaptation; the nominal-trained one does",
    "🖥️ Interactive workstation: 2D stock depletion, micro-topography, G-code, controllers",
    "🔁 Full reproducibility: every result table from committed artifacts by one script",
]

y = Inches(2.15)
for c in conclusions:
    add_text_box(slide, Inches(1), y, Inches(11.333), Inches(0.45), c, font_size=16, color=WHITE, alignment=PP_ALIGN.LEFT)
    y += Inches(0.45)

add_shape(slide, Inches(1), Inches(6.2), Inches(11.333), Pt(4), fill_color=IIT_GOLD)
add_text_box(slide, Inches(1), Inches(6.32), Inches(11.333), Inches(0.9),
    "Key Scientific Finding: The speed-dependent specific cutting energy creates a genuine energy/throughput trade-off. Across the measured frontier the removal rate falls 8.4% while the energy of the part falls 9.5%. One conditioned policy reaches that whole frontier at runtime, beats the 21 policies trained per weight, and transfers to a harder material with no adaptation.",
    font_size=15, bold=True, color=IIT_GOLD, alignment=PP_ALIGN.CENTER)

# ============================================================
# SLIDE 19: Reproducibility Commands
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Reproducibility: All Commands Run from Project Root",
    font_size=30, bold=True, color=WHITE)

cmds = [
    ("Tests", "pytest tests/ -v"),
    ("Production policy", "python -m experiments.train_rl --steps 1000000 --seed 42 --randomize --randomize-weight"),
    ("One pinned weight", "python -m experiments.train_rl --steps 150000 --seed 42 --randomize --w2 0.6"),
    ("Ablation grid", "python -m experiments.train_rl_ablation --steps 150000 --scenarios 50 --slots 7"),
    ("Monte Carlo eval", "python -m experiments.evaluate --scenarios 100"),
    ("Weight sweep", "python -m experiments.evaluate --sweep-weight --scenarios 100"),
    ("Compare with per-weight policies", "python -m experiments.evaluate --compare-only"),
    ("Classical baselines", "python -m experiments.run_baselines"),
    ("Transfer experiment", "python -m experiments.transfer_experiment 300000 50000"),
    ("Controller initialization", "python -m experiments.controller_experiment 1000000 2000"),
    ("Regenerate RESULTS.md", "python -m experiments.summarize_results"),
    ("Launch workstation", "streamlit run app.py"),
    ("Full pipeline", "python -m experiments.run_pipeline --steps 1000000"),
]

y = Inches(1.35)
for idx, (name, cmd) in enumerate(cmds):
    add_shape(slide, Inches(0.5), y, Inches(12.3), Inches(0.42), fill_color=LIGHT_GRAY if idx % 2 == 0 else WHITE)
    add_text_box(slide, Inches(0.7), y + Inches(0.02), Inches(3.0), Inches(0.4), name, font_size=15, bold=True, color=IIT_BLUE)
    add_text_box(slide, Inches(3.8), y + Inches(0.02), Inches(8.9), Inches(0.4), cmd, font_size=13, color=DARK_GRAY, font_name='Consolas')
    y += Inches(0.45)

# ============================================================
# SLIDE 20: Thank You / Questions
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
    "ME6324, AI in Manufacturing  •  IIT Madras",
    font_size=18, color=IIT_GOLD, alignment=PP_ALIGN.CENTER)

# Save
output_path = r"G:\My Drive\IITM\AI in Mfg\report\presentation.pptx"
prs.save(output_path)
print(f"Presentation saved to {output_path}")
print(f"Slides: {len(prs.slides)}")
