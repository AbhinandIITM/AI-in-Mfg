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
from PIL import Image
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

# The figures live in ../results/plots, relative to the report directory.
# The path is resolved against this file so the script runs from any directory.
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.normpath(os.path.join(REPORT_DIR, "..", "results", "plots"))

def figure_path(name):
    """Path of one figure, relative to the report directory."""
    return os.path.join(PLOTS_DIR, name)

def add_picture_fit(slide, image_path, left, top, max_width, max_height):
    """Place a picture scaled to fit the box, keeping its aspect ratio."""
    with Image.open(image_path) as im:
        iw, ih = im.size
    scale = min(max_width / iw, max_height / ih)
    width = int(iw * scale)
    height = int(ih * scale)
    left = left + int((max_width - width) / 2)
    top = top + int((max_height - height) / 2)
    return slide.shapes.add_picture(image_path, left, top, width, height)

def add_figure_slide(title, image_name, note_title, note):
    """A figure slide in the deck's layout: blue header, full-width image, reading box."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide, WHITE)
    add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
    add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
        title, font_size=30, bold=True, color=WHITE)

    add_picture_fit(slide, figure_path(image_name),
                    Inches(0.3), Inches(1.25), Inches(12.73), Inches(4.75))

    add_shape(slide, Inches(0.5), Inches(6.08), Inches(12.3), Inches(1.12),
        fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
    add_text_box(slide, Inches(0.8), Inches(6.16), Inches(11.7), Inches(0.35),
        note_title, font_size=16, bold=True, color=IIT_GOLD)
    add_text_box(slide, Inches(0.8), Inches(6.52), Inches(11.7), Inches(0.6),
        note, font_size=14, color=DARK_GRAY)
    return slide

def style_header_row(table, row_idx, ncols, font_size=13):
    """Blue header band for one row, skipping the cells covered by a merge."""
    for j in range(ncols):
        cell = table.cell(row_idx, j)
        if cell.is_spanned:
            continue
        cell.fill.solid()
        cell.fill.fore_color.rgb = IIT_BLUE
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(font_size)
            p.font.name = 'Calibri'
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER

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
        "Extended Taylor tool wear, anchored on a published tool life",
        "Empirical surface roughness regression",
        "Stochastic tool breakage",
        "Domain randomization per episode",
    ],
    [
        "One Gymnasium environment for a part, one for a batch",
        "Energy-aware reward: 100·(w₁·MRR − w₂·Energy) + progress − penalties",
        "Conditioned policy: one model covers the full frontier",
        "PPO with VecNormalize, 4 parallel envs, CPU training",
    ],
    [
        "100-scenario Monte Carlo with 95% CIs",
        "Ablation: 7 energy weights × 3 seeds, weight pinned",
        "5 classical baselines (incl. receding-horizon)",
        "Robustness: 2 regimes against the controller",
        "Regime map: two sweeps, the noise and the material model error",
        "Batch: one tool cuts ten parts",
        "Reward sensitivity: the objective form and scale",
        "Transfer experiment",
        "Full reproducibility",
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
    ("Robustness vs a controller", "✗", "✗", "✓"),
    ("Crossover vs that controller", "✗", "✗", "✓"),
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
    "energy/throughput trade-off, and one conditioned policy covers the whole frontier in a single run. None of the three "
    "carries a Taylor wear model, so the wear coefficient here is anchored on a published tool life.",
    font_size=15, bold=True, color=IIT_GOLD)

# ============================================================
# SLIDE 5: Physics - Key Equations
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Digital Twin Physics (MillingDT/config/physics.py)",
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
    "With m=0.6:",
    "  Cutting energy RISES with speed",
    "  Idle energy FALLS with speed",
    "  Part energy has INTERIOR MINIMUM",
    "  Energy weight w₂ selects operating point",
    "",
    "Without it (m=0): both objectives push to",
    "  MAX speed → no trade-off at all",
    "",
    "Wear: C_wear = 1.0e-2, anchored on a",
    "  published tool life (about 9 parts per tool)",
    "",
    "Validated by a unit test on the part energy",
]
add_bullet_list(slide, Inches(7.3), Inches(2.2), Inches(5.2), Inches(4.2), insight_items, font_size=14, color=DARK_GRAY, spacing=Pt(4))

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
    "Two registrations:",
    "  • MillingDTEnv-v1 (env.ENV_ID), 200-step limit, 11-D",
    "  • MillingDTBatchEnv-v1 (env.BATCH_ENV_ID), 13-D",
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
    "  r = 100·(w₁·MRR/MRR_max − w₂·E_pass/E_ref)",
    "       + 20·Δ(stock removed)",
    "       − 50·1[V_B > V_B_safe] − 100·1[broken]",
    "       − 50·1[finishing pass ∧ R_a > R_a_safe] − 0.05",
    "  +200 for a part finished in spec, +≤5 Pareto margin",
    "  w₁ = 1 − w₂; safety limits 0.255 mm and 1.36 μm",
    "  breakage p = 0.3·wear factor·(load factor)² above 0.15 mm",
    "  THE SCALE IS A DESIGN DECISION: the objective term is",
    "  worth about 50 points per part, a quarter of a part",
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
    "2️⃣ Ablation grid: 7 weights × 3 seeds = 21 policies",
    "    w₂ ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}",
    "    seeds {42, 43, 44}, 150k steps each",
    "    weight PINNED at training and at evaluation",
    "3️⃣ Batch-aware policy: 1M steps on MillingDTBatchEnv-v1",
    "4️⃣ Noisy-measurement policy: 1M steps, --obs-noise 0.05",
    "Heartbeat callback: queue signal every 1000 steps",
    "→ stall detection in run_experiments.py",
]
add_bullet_list(slide, Inches(7.3), Inches(2.2), Inches(5.2), Inches(4.5), runs, font_size=14, color=DARK_GRAY, spacing=Pt(4))

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
    ("MRR (mm³/min)", "1608.5", "550.7", "[1499.2, 1717.8]"),
    ("Energy/part (kJ)", "66.4", "48.0", "[56.9, 75.9]"),
    ("Cycle time (s)", "11.4", "6.9", "[10.1, 12.8]"),
    ("Max R_a (μm)", "1.610", "0.178", "[1.575, 1.645]"),
    ("Max V_B (mm)", "0.1407", "0.0672", "[0.1274, 0.1540]"),
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
    ("R_a viol., worst pass", "55.0%", "[45.2, 64.8]%"),
    ("R_a viol., finished surface", "4.0%", "[0.2, 7.8]%"),
    ("R_a above safety limit, worst", "87.0%", "[80.4, 93.6]%"),
    ("R_a above safety limit, finished", "4.0%", "[0.2, 7.8]%"),
    ("Flank wear violations (V_B > 0.3)", "0.0%", "[0, 0]%"),
    ("Tool breakage", "6.0%", "[1.3, 10.7]%"),
    ("Parts completed in spec", "94.0%", ""),
]

for i, row in enumerate(vio_data):
    for j, val in enumerate(row):
        vio_table.cell(i, j).text = val
style_table(vio_table, font_size=13)

# Key takeaway box
add_shape(slide, Inches(0.5), Inches(4.35), Inches(12.3), Inches(2.6), fill_color=RGBColor(230, 245, 230), line_color=GREEN, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(4.5), Inches(11.7), Inches(0.5), "✅ Key Findings", font_size=22, bold=True, color=GREEN)
takeaways = [
    "• A much more aggressive operator than under the old reward: 1608.5 mm³/min against 1205.2, and 11.4 s per part against 20.3 s",
    "• The price is paid in the tool and in the part quality: 6.0% tool breakage and 4.0% of the finished surfaces outside the roughness limit",
    "• The roughing passes are rough: 87% of the worst passes are above the safety margin, 55% above the hard limit",
    "• Wear is not the binding constraint on a single part (0% violations, mean 0.1407 mm); it binds on the aggregate of a batch",
    "• Mean operating point: 7907 RPM, 0.318 mm/r, 2.07 mm, 4.0 passes",
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
    ("0.0", "1397.1±14.8", "70.4±0.1", "14.8", "7998", "0.299", "✗ dominated"),
    ("0.2", "1456.9±14.6", "70.0±0.1", "13.8", "7992", "0.299", "✗ dominated"),
    ("0.4", "1524.9±40.2", "69.3±0.2", "13.0", "7896", "0.296", "✗ dominated"),
    ("0.5", "1556.6±9.7", "69.1±0.2", "12.7", "7859", "0.296", "✓"),
    ("0.6", "1519.8±36.4", "68.3±0.4", "13.1", "7638", "0.296", "✓"),
    ("0.8", "1224.4±11.3", "63.0±0.2", "15.4", "6573", "0.292", "✓"),
    ("1.0", "918.5±14.9", "57.2±0.2", "20.1", "5382", "0.292", "✓"),
]

for i, row in enumerate(abl_data):
    for j, val in enumerate(row):
        table.cell(i, j).text = val
style_table(table, font_size=13)

# Insight box
add_shape(slide, Inches(0.5), Inches(4.35), Inches(12.3), Inches(2.6), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(4.5), Inches(11.7), Inches(0.5), "📊 Frontier Analysis", font_size=22, bold=True, color=IIT_GOLD)
insights = [
    "• From w₂ = 0.5 to w₂ = 1.0 the removal rate falls 41% (1557 → 919 mm³/min) and the energy falls 17% (69.1 → 57.2 kJ): the trade-off is asymmetric",
    "• The energy is bought with a lot of removal rate, and with time (12.7 → 20.1 s); the spindle speed falls with the weight (7859 → 5382 RPM)",
    "• Dominated points (w₂ = 0.0, 0.2, 0.4) remove less for more",
    "• Each policy has its weight pinned at training and evaluated at that same weight; 3 seeds, 50 scenarios each",
]
add_bullet_list(slide, Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.7), insights, font_size=14, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE 10: Ablation Frontier (figure)
# ============================================================
add_figure_slide(
    "Ablation Frontier: Removal Rate against Energy",
    "ablation_pareto_frontier.png",
    "How to read this",
    "Seven pinned-weight policies, three seeds each. The four non-dominated points are joined into the frontier: "
    "w₂ = 0.5, 0.6, 0.8 and 1.0, and the three dominated points sit off it. Left: the energy of a part. "
    "Right: the energy of a pass, which the reward charges.")

# ============================================================
# SLIDE 11: One Policy Against 21 Policies
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
    ("0.0", "1475.9", "1397.1", "+5.6%", "68.4", "70.4", "−2.9%"),
    ("0.2", "1518.0", "1456.9", "+4.2%", "65.8", "70.0", "−6.1%"),
    ("0.4", "1608.5", "1524.9", "+5.5%", "66.4", "69.3", "−4.2%"),
    ("0.5", "1630.3", "1556.6", "+4.7%", "66.0", "69.1", "−4.4%"),
    ("0.6", "1540.8", "1519.8", "+1.4%", "65.5", "68.3", "−4.0%"),
    ("0.8", "1114.5", "1224.4", "−9.0%", "57.9", "63.0", "−8.0%"),
    ("1.0", "720.4", "918.5", "−21.6%", "51.7", "57.2", "−9.7%"),
]

for i, row in enumerate(cond_data):
    for j, val in enumerate(row):
        table.cell(i, j).text = val
style_table(table, font_size=13)

# Key message
add_shape(slide, Inches(0.5), Inches(4.35), Inches(12.3), Inches(2.6), fill_color=RGBColor(230, 245, 230), line_color=GREEN, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(4.5), Inches(11.7), Inches(0.5), "🎯 Result (the previous claim has reversed)", font_size=22, bold=True, color=GREEN)
msg = [
    "• Over the weights from 0.0 to 0.6 the single policy removes 1.4% to 5.6% more material for 2.9% to 6.1% less energy",
    "• At w₂ = 0.8 and w₂ = 1.0 it is behind by 9.0% and 21.6% of the removal rate",
    "• The weight in the observation is a weaker signal than a dedicated policy where the trade-off is steepest",
    "• The same policy swept at runtime: MRR 1475.9 (w₂ = 0.0) → 720.4 (w₂ = 1.0), energy 68.4 → 51.7 kJ, spindle 8000 → 4111 RPM",
]
add_bullet_list(slide, Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.7), msg, font_size=15, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE 12: Baseline Comparison
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Baseline Comparison (50 Episodes, Nominal Condition, w₂ = 0.4)",
    font_size=30, bold=True, color=WHITE)

# Baseline table
table_shape = slide.shapes.add_table(6, 5, Inches(1.2), Inches(1.4), Inches(10.8), Inches(2.4))
table = table_shape.table
for j, w in enumerate([3.0, 1.8, 2.2, 2.0, 1.8]):
    table.columns[j].width = Inches(w)

base_data = [
    ("Policy", "Reward", "MRR (mm³/min)", "Energy (kJ)", "Cycle (s)"),
    ("Constant high", "10.0", "1099.6", "17.8", "4.3"),
    ("NSGA-II static", "5.1", "732.9", "18.8", "6.4"),
    ("Rule based", "3.3", "719.8", "18.8", "6.6"),
    ("Receding-horizon", "253.5", "725.5", "19.0", "6.7"),
    ("Constant safe", "−1882.9", "1.6", "385.7", "768.0"),
]

for i, row in enumerate(base_data):
    for j, val in enumerate(row):
        table.cell(i, j).text = val
style_table(table, font_size=14)
for j in range(5):
    table.cell(5, j).fill.solid()
    table.cell(5, j).fill.fore_color.rgb = RGBColor(255, 230, 230)
    table.cell(4, j).fill.solid()
    table.cell(4, j).fill.fore_color.rgb = RGBColor(255, 248, 220)

# Analysis
add_shape(slide, Inches(0.5), Inches(4.15), Inches(12.3), Inches(2.8), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
add_text_box(slide, Inches(0.8), Inches(4.3), Inches(11.7), Inches(0.4), "Analysis", font_size=20, bold=True, color=IIT_BLUE)
analysis = [
    "• Same 50 episodes of the nominal condition for all six policies, at w₂ = 0.4",
    "• Three of the four static policies finish outside the roughness safety limit (1.56 μm against 1.36 μm): no completion bonus, the finishing penalty, so they land near zero",
    "• Constant-Safe stays inside the limit (0.26 μm) but needs 768 s and 385.7 kJ per part: −1882.9",
    "• Receding-horizon is the strongest baseline, +253.5, and the only classical policy that finishes every part in spec",
    "• It pays for that with throughput: 725.5 mm³/min and 6.7 s against 1099.6 mm³/min and 4.3 s",
]
add_bullet_list(slide, Inches(0.8), Inches(4.85), Inches(11.7), Inches(2.0), analysis, font_size=15, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE 13: Policy Comparison (figure)
# ============================================================
add_figure_slide(
    "Policy Comparison: Six Policies, Same 50 Episodes",
    "baseline_vs_ppo_comparison.png",
    "How to read this",
    "The three fast static policies land near zero: they finish outside the roughness safety limit, so they are paid "
    "no completion bonus and are charged the finishing penalty. Constant-Safe stays inside but needs 768 s and "
    "385.7 kJ per part. The comparison that matters is the receding-horizon controller against the learned policy, "
    "the only two with a positive reward.")

# ============================================================
# SLIDE 14: Robustness Against the Classical Controller
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Robustness Against the Classical Controller", font_size=30, bold=True, color=WHITE)

rob_data = [
    ("Regime", "Arm", "Reward", "Std", "MRR", "kJ", "s", "Broken", "Done"),
    ("Controller has every advantage", "Receding-horizon", "248.9", "79.5", "1318.2", "68.9", "17.1", "3%", "96%"),
    ("", "PPO", "233.6", "79.7", "1608.5", "66.4", "11.4", "6%", "94%"),
    ("Wear measurement is noisy", "Receding-horizon", "181.3", "185.9", "1300.4", "74.2", "31.1", "4%", "91%"),
    ("", "PPO", "226.8", "87.2", "1602.1", "66.3", "11.5", "7%", "93%"),
    ("", "PPO trained with noise", "213.6", "87.3", "1554.2", "66.9", "12.0", "6%", "94%"),
]

rob_table = slide.shapes.add_table(6, 9, Inches(0.4), Inches(1.35), Inches(12.5), Inches(2.6)).table
for j, w in enumerate([2.6, 2.2, 1.1, 1.0, 1.2, 1.0, 1.0, 1.1, 1.0]):
    rob_table.columns[j].width = Inches(w)
for i, row in enumerate(rob_data):
    for j, val in enumerate(row):
        rob_table.cell(i, j).text = val
style_table(rob_table, font_size=12)
for j in range(9):
    rob_table.cell(2, j).fill.solid()
    rob_table.cell(2, j).fill.fore_color.rgb = RGBColor(255, 248, 220)
    rob_table.cell(4, j).fill.solid()
    rob_table.cell(4, j).fill.fore_color.rgb = RGBColor(255, 248, 220)

add_text_box(slide, Inches(0.5), Inches(4.35), Inches(12.3), Inches(0.5),
    "Same 100 randomized scenarios at w₂ = 0.4 in every row. The regime changes what the CONTROLLER is given, "
    "not what the policy is given.", font_size=15, bold=True, color=IIT_BLUE)

add_shape(slide, Inches(0.5), Inches(4.9), Inches(12.3), Inches(2.55), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(4.98), Inches(11.7), Inches(0.4), "🎯 The central result of the work", font_size=20, bold=True, color=IIT_GOLD)
rob_notes = [
    "• Inside a perfect digital twin the two methods are comparable: 248.9 against 233.6, and the gap of 15 is inside the noise (the standard error of the mean is about 8 for both arms)",
    "• The learned policy is the better throughput operator: 22% more material per minute, a third less time, 3.7% less energy, at twice the tool breakage",
    "• The advantage appears where the controller's information advantage disappears: a noisy wear measurement (181.3 against 226.8). The regime map on the next slides puts a number on it",
    "• About nine of those fifteen points are the reward's artifact: the controller cuts the part in 5.85 passes against 4.00 and is paid for every pass",
    "• Fairness: both arms are the general ones, neither is trained for the noise, and the noise is tilted against the policy, which reads it on both measured quantities while the controller reads it on the wear alone",
]
add_bullet_list(slide, Inches(0.8), Inches(5.45), Inches(11.7), Inches(1.9), rob_notes, font_size=11, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE: Robustness, six ways the setup's model can be wrong (figure)
# ============================================================
add_figure_slide(
    "Robustness: Six Ways the Setup's Model Can Be Wrong",
    "robustness_vs_controller.png",
    "How to read this",
    "Six panels: the reward, its spread, the removal rate, the energy, the cycle time and the "
    "reliability. The six cases are the perfect twin, the wear measurement noisy by 2% and by 10%, "
    "and the material model off by 5%, 20% and 30%. The values are in the tables.")

# ============================================================
# SLIDE: The Regime Map, the figure and the statement
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "The Regime Map: Two Sweeps, One Headline",
    font_size=30, bold=True, color=WHITE)

add_picture_fit(slide, figure_path("regime_map.png"),
                Inches(0.5), Inches(1.25), Inches(12.33), Inches(4.2))

add_text_box(slide, Inches(0.5), Inches(5.55), Inches(12.3), Inches(0.5),
    "Mean episode reward over the same 100 randomized scenarios at w₂ = 0.4. Left: the measurement "
    "noise on the flank wear and the remaining stock, as a percent of their range. Right: the error in "
    "the setup's belief about the material, as a percentage below the truth.",
    font_size=13, color=MED_GRAY)

add_shape(slide, Inches(0.5), Inches(6.1), Inches(12.3), Inches(1.15), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(6.18), Inches(11.7), Inches(0.4),
    "One headline, two sweeps",
    font_size=20, bold=True, color=IIT_GOLD)
add_bullet_list(slide, Inches(0.8), Inches(6.62), Inches(11.7), Inches(0.55), [
    "• The learned policy is the better operator once the wear measurement is noisier than about 2% of its range, and the classical controller is better below that. The noise sweep is the headline: it applies to both arms in the same way.",
], font_size=13, color=DARK_GRAY, spacing=Pt(3))

# ============================================================
# SLIDE: The Regime Map, the sweep
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Regime Map: How Noisy the Wear Measurement Has to Be",
    font_size=30, bold=True, color=WHITE)

regime_data = [
    ("Measurement\nnoise", "Receding-\nhorizon", "PPO", "PPO trained\nfor the noise", "PPO\nahead by"),
    ("0.0%", "248.9", "233.6", "219.6", "−15.3"),
    ("1.0%", "242.3", "233.4", "216.9", "−8.9"),
    ("2.0%", "231.5", "233.1", "214.3", "+1.7"),
    ("3.0%", "211.5", "230.0", "214.1", "+18.5"),
    ("5.0%", "181.3", "226.8", "213.6", "+45.5"),
    ("7.5%", "179.1", "223.2", "207.9", "+44.1"),
    ("10.0%", "171.5", "222.0", "206.5", "+50.6"),
]

regime_table = slide.shapes.add_table(8, 5, Inches(0.4), Inches(1.35), Inches(6.0), Inches(2.7)).table
for j, w in enumerate([1.5, 1.15, 0.85, 1.45, 1.05]):
    regime_table.columns[j].width = Inches(w)
for i, row in enumerate(regime_data):
    for j, val in enumerate(row):
        regime_table.cell(i, j).text = val
style_table(regime_table, font_size=12)
# The crossover row: the controller still leads, the policy is already ahead.
for j in range(5):
    regime_table.cell(3, j).fill.solid()
    regime_table.cell(3, j).fill.fore_color.rgb = RGBColor(255, 248, 220)

add_text_box(slide, Inches(0.4), Inches(4.2), Inches(6.0), Inches(1.1),
    "Mean episode reward over the same 100 randomized scenarios at w₂ = 0.4. The noise is a fraction of the "
    "range of the flank wear and of the remaining stock. The last column is the general policy's reward "
    "minus the controller's. This sweep is the headline: the noise is the fair test, a property of the "
    "instrumentation that applies to both arms alike.",
    font_size=12, color=MED_GRAY)

add_shape(slide, Inches(6.7), Inches(1.35), Inches(6.15), Inches(3.95), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
add_text_box(slide, Inches(6.95), Inches(1.45), Inches(5.7), Inches(0.4), "Three readings", font_size=18, bold=True, color=IIT_BLUE)
add_bullet_list(slide, Inches(6.95), Inches(1.95), Inches(5.7), Inches(3.2), [
    "• The crossover is at about 2% of the range of the wear measurement. Below it the classical controller is the better operator, above it the learned policy is.",
    "• The two degrade differently: the controller falls from 248.9 to 171.5, a 31% loss, and the policy from 233.6 to 222.0, a 5% loss. A model-based controller has a cliff, a learned policy has a slope.",
    "• Training for the noise does not pay: the noise-trained policy is below the general policy at every noise level, including the 5% it was trained for (213.6 against 226.8).",
], font_size=13, color=DARK_GRAY, spacing=Pt(6))

add_shape(slide, Inches(0.5), Inches(5.55), Inches(12.3), Inches(1.65), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(5.63), Inches(11.7), Inches(0.4),
    "The crossover is at about 2% of the range of the wear measurement",
    font_size=20, bold=True, color=IIT_GOLD)
add_bullet_list(slide, Inches(0.8), Inches(6.1), Inches(11.7), Inches(1.0), [
    "• Below it the classical controller is the better operator, above it the learned policy is: a plant that can measure the flank wear to better than 2% of its range has the better controller, one that cannot has the better policy",
    "• The noise regime is the fair one, because the noise is a property of the instrumentation and not of the method: it changes only what the instrument reports, and it applies to both arms in the same way",
], font_size=14, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE: The Regime Map, the material model sweep
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Regime Map: How Wrong the Material Model Can Be",
    font_size=30, bold=True, color=WHITE)

model_data = [
    ("Material model\noff by", "Receding-\nhorizon", "PPO", "PPO trained\nfor the noise", "PPO\nahead by"),
    ("0%", "248.9", "233.6", "219.6", "−15.3"),
    ("5%", "201.5", "232.8", "219.6", "+31.2"),
    ("10%", "136.0", "237.8", "219.6", "+101.8"),
    ("20%", "35.7", "201.5", "219.6", "+165.8"),
    ("30%", "32.5", "116.7", "219.6", "+84.3"),
    ("40%", "39.3", "117.0", "219.6", "+77.7"),
]

model_table = slide.shapes.add_table(7, 5, Inches(0.4), Inches(1.35), Inches(6.0), Inches(2.7)).table
for j, w in enumerate([1.5, 1.15, 0.85, 1.45, 1.05]):
    model_table.columns[j].width = Inches(w)
for i, row in enumerate(model_data):
    for j, val in enumerate(row):
        model_table.cell(i, j).text = val
style_table(model_table, font_size=12)
# The row where the policy pulls clearly ahead of the controller.
for j in range(5):
    model_table.cell(4, j).fill.solid()
    model_table.cell(4, j).fill.fore_color.rgb = RGBColor(255, 248, 220)

add_text_box(slide, Inches(0.4), Inches(4.2), Inches(6.0), Inches(1.1),
    "Mean episode reward over the same 100 randomized scenarios at w₂ = 0.4. The belief about the "
    "material is a percentage below the truth, so the setup under-rates the wear and the roughness of "
    "the material. The same belief goes to both arms; the measurement noise is zero.",
    font_size=12, color=MED_GRAY)

add_shape(slide, Inches(6.7), Inches(1.35), Inches(6.15), Inches(3.95), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
add_text_box(slide, Inches(6.95), Inches(1.45), Inches(5.7), Inches(0.4), "Three readings", font_size=18, bold=True, color=IIT_BLUE)
add_bullet_list(slide, Inches(6.95), Inches(1.95), Inches(5.7), Inches(3.2), [
    "• The controller's tolerance of a wrong material model is about 2% to 3%: a 5% error costs it 47 points (248.9 to 201.5) and a 20% error costs it 213 (to 35.7).",
    "• The learned policy holds to a 20% error (201.5, 13% below its own no-error score) and collapses at 30% (116.7): the belief has left the distribution it was trained on and its context feature is clipped at the edge of the band.",
    "• A conservative policy is insensitive to a wrong model: the policy trained for the noise is flat at 219.6 across the whole range, and it is below the general policy everywhere.",
], font_size=13, color=DARK_GRAY, spacing=Pt(6))

add_shape(slide, Inches(0.5), Inches(5.55), Inches(12.3), Inches(1.65), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(5.63), Inches(11.7), Inches(0.4),
    "Supporting evidence, not the headline",
    font_size=20, bold=True, color=IIT_GOLD)
add_bullet_list(slide, Inches(0.8), Inches(6.1), Inches(11.7), Inches(1.0), [
    "• Fairness: the same belief goes to both arms, the error is in the damaging direction, both arms are the general ones, and the noise model is tilted against the policy wherever noise is present",
    "• The noise sweep stays the headline, because the noise is the fair test: it is a property of the instrumentation and it applies to both arms in the same way",
], font_size=14, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE: The Regime Map, the fairness decisions
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "The Regime Map: The Fairness Decisions",
    font_size=30, bold=True, color=WHITE)

add_shape(slide, Inches(0.5), Inches(1.35), Inches(12.3), Inches(3.9), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
add_text_box(slide, Inches(0.8), Inches(1.45), Inches(11.7), Inches(0.4),
    "The fairness decisions", font_size=20, bold=True, color=IIT_BLUE)
add_bullet_list(slide, Inches(0.8), Inches(1.95), Inches(11.7), Inches(3.1), [
    "• Noise sweep: both arms of the headline comparison are the general ones, neither is trained for the noise.",
    "• The noise is a property of the instrumentation and not of the method: it changes only what the instrument reports, and it applies to both arms in the same way.",
    "• The noise is tilted against the policy: the controller gets a noisy reading of the flank wear alone, the policy gets the noise on the flank wear and on the remaining stock.",
    "• The noise-trained policy is a third, clearly labelled arm in the table and in the sweep, and it is not the headline: a controller can also be prepared for a noisy instrument by widening its safety margin.",
    "• Model error sweep: the same belief about the material goes to both arms, the controller solves with it and the policy reads it in its context.",
    "• The error is in the damaging direction, because a setup that under-rates the wear and the roughness of the material it is cutting will push the tool harder than the material can take.",
    "• Both arms are the general ones, neither is prepared for the error, and the noise model is tilted against the policy wherever noise is present.",
], font_size=13, color=DARK_GRAY, spacing=Pt(4))

add_shape(slide, Inches(0.5), Inches(5.45), Inches(12.3), Inches(1.75), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(5.53), Inches(11.7), Inches(0.4),
    "What the comparison claims", font_size=20, bold=True, color=IIT_GOLD)
add_bullet_list(slide, Inches(0.8), Inches(6.0), Inches(11.7), Inches(1.1), [
    "• The learned policy is the better operator once the wear measurement is noisier than about 2% of its range, and the classical controller is the better operator below that, inside a perfect digital twin",
    "• The model error sweep supports that boundary and does not carry it: the controller tolerates 2% to 3% of a wrong material model, the learned policy 20%",
], font_size=14, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE 15: Reward Sensitivity
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Reward Sensitivity: How the Reward Pays for a Rate", font_size=30, bold=True, color=WHITE)

add_text_box(slide, Inches(0.5), Inches(1.2), Inches(12.3), Inches(0.85),
    "The reward pays the removal rate of every pass, so the total credit grows with the number of passes a plan takes: a plan that "
    "cuts the same part in more, shallower passes is paid more for it. On the same 100 scenarios at w₂ = 0.4, measured by "
    "experiments/reward_sensitivity.py, the controller takes 5.85 passes against the policy's 4.00 and collects 27% more objective "
    "credit (+0.413 against +0.326) for the same material.",
    font_size=14, color=DARK_GRAY)

rs_rows = [
    ("5", "209.7", "202.6", "+7.1", "208.2", "201.6", "+6.5"),
    ("20", "215.9", "207.5", "+8.3", "209.7", "203.4", "+6.3"),
    ("50", "228.3", "217.3", "+10.9", "212.9", "207.1", "+5.8"),
    ("100", "248.9", "233.6", "+15.3", "218.2", "213.2", "+5.0"),
    ("200", "290.1", "266.2", "+23.9", "228.7", "225.3", "+3.4"),
    ("400", "372.7", "331.5", "+41.2", "249.9", "249.6", "+0.2"),
    ("1000", "620.2", "527.2", "+93.0", "313.2", "322.6", "-9.4"),
    ("4000", "1857.9", "1506.0", "+352.0", "630.1", "687.4", "-57.4"),
]

rs_table = slide.shapes.add_table(10, 7, Inches(0.6), Inches(2.05), Inches(12.1), Inches(2.8)).table
for j, w in enumerate([2.0, 1.7, 1.5, 1.3, 1.7, 1.5, 1.3]):
    rs_table.columns[j].width = Inches(w)

rs_table.cell(0, 0).text = "Objective scale"
rs_table.cell(0, 1).merge(rs_table.cell(0, 3))
rs_table.cell(0, 1).text = "Shipped form (sum of the per-pass rates)"
rs_table.cell(0, 4).merge(rs_table.cell(0, 6))
rs_table.cell(0, 4).text = "Alternative form (material against energy)"
for j, val in enumerate(["Controller", "Policy", "Gap", "Controller", "Policy", "Gap"]):
    rs_table.cell(1, j + 1).text = val
for i, row in enumerate(rs_rows):
    for j, val in enumerate(row):
        rs_table.cell(i + 2, j).text = val
style_table(rs_table, header_rows=0, font_size=11, highlight_rows=(5,))
style_header_row(rs_table, 0, 7, font_size=11)
style_header_row(rs_table, 1, 7, font_size=11)

add_shape(slide, Inches(0.5), Inches(5.05), Inches(12.3), Inches(2.15), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(5.12), Inches(11.7), Inches(0.4), "Three checks and the consequence", font_size=19, bold=True, color=IIT_GOLD)
rs_notes = [
    "• The objective scale is not the lever: the controller leads at every scale from 5 to 4000 and the gap grows from 7.1 to 352.0 points, so the sign never flips",
    "• The part level alternative was implemented and rejected: it removes the pass count from the accounting (5.0 points at the scale of 100 against 15.3, and the policy ahead by 9.4 at 1000 and 57.4 at 4000), but the material a part gives up is fixed by its stock, so the throughput weight would be inert",
    "• A correct form is not obvious: a rate cannot be accumulated over a part without becoming a volume, so a reward has to price the cycle time directly or compare the part's average rate to a reference at the end of the part",
    "• Consequence: the ideal case reads as a tie, not a win. The policy is ahead on the physical metrics and behind on the breakage, and the reward as shipped carries the artifact, documented rather than hidden",
]
add_bullet_list(slide, Inches(0.8), Inches(5.58), Inches(11.7), Inches(1.55), rs_notes, font_size=12, color=DARK_GRAY, spacing=Pt(3))

# ============================================================
# SLIDE: Reward Sensitivity, the figure
# ============================================================
add_figure_slide(
    "Reward Sensitivity: How Much of the Comparison the Reward Decides",
    "reward_sensitivity.png",
    "How to read this",
    "Three panels: the reward as the environment pays it, the part level alternative, and the gap "
    "between the arms. The scale is not the lever: the controller leads at every scale and the gap "
    "grows with it, 7.1 points at 5 to 352.0 at 4000. The accounting of a rate is: the arms close to "
    "5.0 points at the scale of 100, and the policy overtakes by 9.4 at 1000 and 57.4 at 4000.")

# ============================================================
# SLIDE 15: Batch Against the Part
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Batch Against the Part: One Tool, Ten Parts", font_size=30, bold=True, color=WHITE)

batch_data = [
    ("Arm", "Batch reward", "Std", "Reward/part", "Tool changes", "Batch time (s)", "Batch energy (kJ)"),
    ("PPO, batch aware", "1817.0", "477.8", "166.8", "0.63", "113.1", "610.4"),
    ("PPO, per part", "1709.9", "366.1", "169.4", "0.83", "114.7", "674.3"),
    ("Receding-horizon, per part", "1859.9", "328.8", "173.7", "0.83", "207.5", "692.0"),
]

batch_table = slide.shapes.add_table(4, 7, Inches(0.5), Inches(1.4), Inches(12.3), Inches(1.7)).table
for j, w in enumerate([2.9, 1.6, 1.1, 1.6, 1.6, 1.6, 1.9]):
    batch_table.columns[j].width = Inches(w)
for i, row in enumerate(batch_data):
    for j, val in enumerate(row):
        batch_table.cell(i, j).text = val
style_table(batch_table, font_size=13)

add_shape(slide, Inches(0.5), Inches(3.4), Inches(12.3), Inches(3.5), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
add_text_box(slide, Inches(0.8), Inches(3.55), Inches(11.7), Inches(0.4), "How to read this", font_size=20, bold=True, color=IIT_BLUE)
batch_notes = [
    "• A batch is ten parts cut with one tool. The tool is replaced when it breaks or reaches its wear limit, and the replacement costs 500 reward units against the 200 a finished part is worth",
    "• The observation is the eleven features of the part plus the tool life left and the fraction of the batch still to cut (MillingDTBatchEnv-v1, 13-D)",
    "• The batch context is worth 24% fewer tool changes and 9.5% less energy at the same completion: 0.63 tool changes against 0.83, and 610.4 kJ against 674.3 kJ. It also collects more batch reward, 1817.0 against 1709.9",
    "• The per-part controller gets the highest batch reward, 1859.9, and buys it with the machine time: 207.5 s for the batch against the batch-aware policy's 113.1 s, about 1.8 times the machine time for a reward that is 2.4% higher",
    "• It solves one part at a time, so it cannot price a tool change against the parts that follow it",
]
add_bullet_list(slide, Inches(0.8), Inches(4.1), Inches(11.7), Inches(2.6), batch_notes, font_size=15, color=DARK_GRAY, spacing=Pt(6))

# ============================================================
# SLIDE: Batch Against the Part, the figure
# ============================================================
add_figure_slide(
    "Batch Against the Part: One Tool, Ten Parts",
    "batch_vs_ppo.png",
    "How to read this",
    "Six panels: the batch reward, the reward per part, the tool changes, the parts cut, the batch "
    "time and the batch energy. The values are in the table. The batch-aware policy replaces the tool "
    "0.63 times per batch against 0.83, spends 610.4 kJ against 674.3 and 692.0, and cuts the batch in "
    "113.1 s against 207.5 s for the per-part controller.")

# ============================================================
# SLIDE 16: Controller Initialization, Method and Result
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
    ("Mean episode reward", "248.9", "233.6", "119.8"),
    ("Reward std", "79.5", "79.7", "178.7"),
    ("MRR (mm³/min)", "1318.2", "1608.5", "1274.9"),
    ("Energy per part (kJ)", "68.9", "66.4", "81.5"),
    ("Cycle time (s)", "17.1", "11.4", "42.6"),
    ("Ra viol., finished", "2%", "4%", "4%"),
    ("Flank wear violations", "1%", "0%", "9%"),
    ("Tool breakage", "3%", "6%", "4%"),
    ("Parts completed", "96%", "94%", "87%"),
]

table_shape = slide.shapes.add_table(len(ci_data), 4, Inches(6.3), Inches(1.9), Inches(6.5), Inches(3.6))
table = table_shape.table
for j, w in enumerate([2.1, 1.5, 1.4, 1.5]):
    table.columns[j].width = Inches(w)

for i, row in enumerate(ci_data):
    for j, val in enumerate(row):
        table.cell(i, j).text = val
style_table(table, font_size=11)

# Reading box under the table
add_shape(slide, Inches(6.3), Inches(5.65), Inches(6.5), Inches(1.15), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(6.55), Inches(5.75), Inches(6.0), Inches(0.4), "Same 100 scenarios, same energy weight w₂ = 0.4", font_size=15, bold=True, color=IIT_GOLD)
add_bullet_list(slide, Inches(6.55), Inches(6.15), Inches(6.0), Inches(0.6), [
    "The initialized arm is the only one that exceeds the flank wear limit (9%)",
], font_size=13, color=DARK_GRAY, spacing=Pt(2))

# ============================================================
# SLIDE 17: Controller Initialization, What It Shows
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Controller Initialization: What It Shows",
    font_size=30, bold=True, color=WHITE)

# Left: no advantage
add_shape(slide, Inches(0.5), Inches(1.35), Inches(6.05), Inches(3.6), fill_color=RGBColor(255, 248, 220), line_color=IIT_GOLD, line_width=Pt(2))
add_text_box(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(0.5), "The initialization is now harmful", font_size=20, bold=True, color=IIT_GOLD)
add_bullet_list(slide, Inches(0.8), Inches(2.1), Inches(5.5), Inches(2.7), [
    "• The initialized policy scores 119.8 against the from-scratch policy's 233.6, a gap of 113.8, more than fourteen times the standard error of the mean",
    "• The clone starts the policy at the controller's conservative operating point, and with the wear constraint binding that conservatism is expensive",
    "• It cuts a part in 42.6 s against 17.1 s and 11.4 s, and spends 81.5 kJ against 66.4 kJ",
    "• It is the only arm that exceeds the flank wear limit (9%)",
], font_size=15, color=DARK_GRAY, spacing=Pt(6))

# Right: consistency
add_shape(slide, Inches(6.8), Inches(1.35), Inches(6.05), Inches(3.6), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
add_text_box(slide, Inches(7.1), Inches(1.5), Inches(5.5), Inches(0.8), "The result reversed", font_size=20, bold=True, color=IIT_BLUE)
add_bullet_list(slide, Inches(7.1), Inches(2.1), Inches(5.5), Inches(2.7), [
    "• The earlier revision found the initialization harmless: it made the policy more consistent and did not change the reward",
    "• That was the reward not counting the cost of the slow operating point",
    "• With the throughput counted, the initialization is what the measurement says it is",
    "• What stays is the record and clone machinery, not the initialization itself",
], font_size=15, color=DARK_GRAY, spacing=Pt(6))

# Claim box
add_shape(slide, Inches(0.5), Inches(5.2), Inches(12.35), Inches(2.0), fill_color=RGBColor(230, 240, 250), line_color=IIT_BLUE, line_width=Pt(1.5))
add_text_box(slide, Inches(0.8), Inches(5.35), Inches(11.7), Inches(0.5), "What the work claims", font_size=20, bold=True, color=IIT_BLUE)
add_bullet_list(slide, Inches(0.8), Inches(5.9), Inches(11.7), Inches(1.2), [
    "Inside a perfect digital twin the learned policy and the model-based controller are comparable: the policy is ahead on the physical metrics and behind on the breakage, and the gap the reward does show is partly the reward's artifact. Its advantage appears where the controller's information advantage disappears: the learned policy is the better operator once the wear measurement is noisier than about 2% of its range.",
], font_size=16, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE 18: Controller Initialization, The Three Arms (figure)
# ============================================================
add_figure_slide(
    "Controller Initialization: The Three Arms",
    "controller_vs_ppo.png",
    "How to read this",
    "Six panels at slide size, so the values are in the table. The initialized arm sits 114 points below the from-scratch "
    "arm on the reward, and its spread of 178.7 is more than twice the spread of the other two: the initialization does "
    "not buy consistency any more, it buys variance.")

# ============================================================
# SLIDE 19: Transfer Experiment
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
    ("Source policy, zero shot", "168.5", "100.0%", "6.7%", "86.7%"),
    ("Source policy, fine tuned (50k)", "101.9", "26.7%", "0.0%", "86.7%"),
    ("Trained on the target from scratch (50k)", "101.9", "26.7%", "0.0%", "86.7%"),
    ("Production policy, zero shot", "190.1", "100.0%", "10.0%", "90.0%"),
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
    "• The production policy was trained across a distribution of materials with the material context in its observation, so the harder material is inside its training distribution and it needs no adaptation at all",
    "• It scores the highest reward (190.1) and completes 90.0% of the parts; its worst pass violates the roughness limit in every scenario, and its finished surface in 10.0% of them",
    "• Adaptation repairs the roughness and costs more than it gains: fine tuning the nominal-trained policy for 50,000 steps, or training on the target from scratch, both cut the worst-pass violations to a quarter and land at 101.9 against the source policy's 168.5 zero shot",
    "• The target alloy wears the tool twice as fast, and a policy that slows down for that trades away more throughput than it gains in specification",
]
add_bullet_list(slide, Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.8), conc_items, font_size=14, color=DARK_GRAY, spacing=Pt(4))

# ============================================================
# SLIDE 20: Interactive Workstation
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
# SLIDE 21: Software Engineering & Reproducibility
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.1), fill_color=IIT_BLUE)
add_text_box(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.7),
    "Software Engineering: Reproducibility & Quality",
    font_size=30, bold=True, color=WHITE)

left_items = [
    "Architecture:",
    "  MillingDT/  →  config, env (part + batch), agent",
    "                  (controller init), baselines, sim, utils",
    "  experiments/ →  train, eval, ablation, baselines,",
    "                  transfer, controller, robustness, regime",
    "                  map, batch, reward sensitivity, pipeline",
    "  viz/         →  shared plotting utilities",
    "  results/     →  one directory per run",
    "",
    "Reproducibility:",
    "  • Global seeding (Python, NumPy, PyTorch, CUDA)",
    "  • Shared scenario tables (train = eval)",
    "  • experiments/summarize_results.py regenerates RESULTS.md",
    "  • All randomness deterministic given seed",
    "  • Centralized constants and paths (MillingDT/config/)",
]
right_items = [
    "Testing (45 tests):",
    "  • 24 env tests: clipping, truncation, soft norm, 11-D",
    "    observation, constants, overrides, energy norm, Gym",
    "    checker, randomization, scenario scales, context",
    "    (weight, material), observation noise, speed",
    "    dependence, energy minimum, reward golden values",
    "  • 11 batch env tests: the 13-D observation, the tool",
    "    life and the batch left, the wear carried over, the",
    "    tool change, the batch reward, the seeded reproducibility",
    "  • 5 integration, 3 callback, 2 training",
    "",
    "Code Quality:",
    "  • No unused imports (AST verified)",
    "  • Conventional commits, clean git history",
]

add_two_column(slide, Inches(0.5), Inches(1.3), Inches(12.3), Inches(5.9),
    left_items, right_items, font_size=14, color=DARK_GRAY, spacing=Pt(4),
    title_left="Architecture & Reproducibility", title_right="Test Coverage & Quality")

# ============================================================
# SLIDE 22: Discussion
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
        "The reward scale decides the operating point: 50 points per part",
        "Domain randomization → 94% completion",
        "One policy covers the range, ahead over w₂ = 0.0 to 0.6",
        "Training across materials makes the adaptation unnecessary",
        "Receding-horizon = strong, fair baseline (184.3 → 248.9)",
        "Robustness: comparable inside a perfect twin, and the advantage appears where the controller's information runs out",
        "The regime map: the learned policy is the better operator once the wear measurement is noisier than about 2% of its range",
        "A second sweep, a wrong material model, supports it: the controller tolerates 2% to 3% of that error, the learned policy 20%",
        "Measuring the reward found the artifact: the credit grows with the number of passes",
        "Batch context: 24% fewer tool changes, 9.5% less energy",
    ],
    [
        "Inside a perfect twin the two are comparable: 248.9 against 233.6, the gap inside the noise, and nine of those points are the reward's artifact",
        "The learned policy is the better throughput operator, at twice the tool breakage",
        "The crossover at 2% is a property of this controller, this reward and this noise model",
        "Tolerance to a wrong material model is set by the training band: 20% holds, 30% collapses",
        "The comparison is one noise model deep: the regime map sweeps that noise and, separately, the material model error",
        "Initializing from the controller is now harmful: 119.8 against 233.6",
        "The reward pays a rate per pass, so the credit grows with the pass count: the bias is in the reward, not the scale",
        "The wear tool life varies by only a factor of a few across the envelope",
        "Ablation uses 150k steps/weight (vs 1M production); sim-only, no real calibration yet",
    ],
    [
        "A reward that pays a rate correctly: price the cycle time directly, or compare the part's average rate to a reference at the end of the part",
        "Publish the sensitivity of any comparison to the reward form, as it is here",
        "Compare against a restricted controller (is the gap an information gap?)",
        "Measure the crossover under a controller prepared for a noisy instrument, and on the noise of a real instrument",
        "A wider randomization band: retrain, repeat the model error sweep, and measure the error in the other direction",
        "Sim-to-real: calibrate on machine data, add noise/delay",
        "True multi-objective RL (Pareto-conditioned, hypervolume)",
        "Online adaptation: runtime w₂ optimizer for energy budget",
        "Batch-level planning: optimize the whole batch, not a part",
    ],
]

x = Inches(0.4)
for title, items in zip(titles, contents):
    add_shape(slide, x, Inches(1.3), col_w, Inches(5.8), fill_color=LIGHT_GRAY, line_color=IIT_BLUE, line_width=Pt(1.5))
    add_text_box(slide, x + Inches(0.2), Inches(1.4), col_w - Inches(0.4), Inches(0.5), title, font_size=20, bold=True, color=IIT_BLUE)
    add_bullet_list(slide, x + Inches(0.2), Inches(2.0), col_w - Inches(0.4), Inches(4.9), items, font_size=11, color=DARK_GRAY, spacing=Pt(4))
    x += col_w + Inches(0.15)

# ============================================================
# SLIDE 23: Conclusion
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, IIT_BLUE)
add_shape(slide, Inches(0.5), Inches(0.8), Inches(12.3), Pt(4), fill_color=IIT_GOLD)

add_text_box(slide, Inches(1), Inches(1.2), Inches(11.333), Inches(1),
    "Conclusion",
    font_size=40, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

conclusions = [
    "🏭 Digital Twin: speed-dependent cutting energy, tool wear, roughness, breakage; the wear coefficient is anchored on a published tool life",
    "🤖 Two environments: the part (11-D) and the batch of ten parts cut with one tool (13-D), both with domain randomization",
    "📊 One production policy (1M steps): the energy weight is a runtime query, not a retrain",
    "🔬 Monte Carlo (100 scenarios, 95% CIs), ablation frontier, sweep against the 21 per-weight policies",
    "⚔️ Five classical baselines on the same 50 episodes; the receding-horizon controller is the strongest at +253.5",
    "🛡️ Robustness: comparable inside a perfect twin (248.9 against 233.6, the gap inside the noise and partly the reward's artifact); with a noisy wear measurement the controller falls to 181.3, the policy holds at 226.8",
    "🗺️ Regime map: two sweeps. The noise sweep carries the claim: the learned policy is the better operator once the wear measurement is noisier than about 2% of its range, and the classical controller is better below that. The model error sweep supports it: the controller tolerates 2% to 3% of a wrong material model, the learned policy 20%",
    "⚖️ Reward sensitivity: the reward pays a rate per pass, so the credit grows with the pass count; the controller collects 27% more credit for the same material",
    "📦 Batch: the batch context is worth 24% fewer tool changes and 9.5% less energy; the per-part controller needs 207.5 s against 113.1 s",
    "🧬 Controller init: the result reversed, the initialization is now harmful (119.8 against 233.6)",
    "🔄 Transfer: the production policy needs no adaptation; the nominal-trained one does",
    "🖥️ Interactive workstation and full reproducibility: every result table from committed artifacts by one script",
]

y = Inches(1.85)
for c in conclusions:
    add_text_box(slide, Inches(1), y, Inches(11.333), Inches(0.4), c, font_size=13, color=WHITE, alignment=PP_ALIGN.LEFT)
    # One line of room per line of text, so the items never overlap.
    y += Inches(0.20 * (1 + len(c) // 140) + 0.07)

add_shape(slide, Inches(1), Inches(6.2), Inches(11.333), Pt(4), fill_color=IIT_GOLD)
add_text_box(slide, Inches(1), Inches(6.3), Inches(11.333), Inches(1.0),
    "Key Scientific Finding: The speed-dependent specific cutting energy creates a genuine energy/throughput trade-off. "
    "Across the frontier the removal rate falls 41% while the energy of the part falls 17%, and one conditioned policy "
    "reaches that whole frontier at runtime. Against a model-based controller the claim is a boundary, not an advantage: "
    "inside a perfect digital twin the two methods are comparable, and the learned policy is the better operator once the "
    "wear measurement is noisier than about 2% of its range.",
    font_size=13, bold=True, color=IIT_GOLD, alignment=PP_ALIGN.CENTER)

# ============================================================
# SLIDE 24: Reproducibility Commands
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
    ("Noisy measurement", "python -m experiments.train_rl --obs-noise 0.05 --run-name PPO_noisystate_seed42"),
    ("Batch policy", "python -m experiments.train_rl --env-id MillingDTBatchEnv-v1 --run-name PPO_batch_seed42"),
    ("Ablation grid", "python -m experiments.train_rl_ablation --steps 150000 --scenarios 50 --slots 7"),
    ("Monte Carlo eval", "python -m experiments.evaluate --scenarios 100"),
    ("Weight sweep", "python -m experiments.evaluate --sweep-weight --scenarios 100"),
    ("Compare with per-weight policies", "python -m experiments.evaluate --compare-only"),
    ("Classical baselines", "python -m experiments.run_baselines"),
    ("Transfer experiment", "python -m experiments.transfer_experiment 300000 50000"),
    ("Controller initialization", "python -m experiments.controller_experiment 1000000 2000"),
    ("Robustness experiment", "python -m experiments.robustness_experiment"),
    ("Regime map", "python -m experiments.regime_map"),
    ("Batch experiment", "python -m experiments.batch_experiment"),
    ("Reward sensitivity", "python -m experiments.reward_sensitivity"),
    ("Regenerate RESULTS.md", "python -m experiments.summarize_results"),
    ("Launch workstation", "streamlit run app.py"),
    ("Full pipeline", "python -m experiments.run_pipeline --steps 1000000 --ablation_steps 150000"),
]

y = Inches(1.3)
for idx, (name, cmd) in enumerate(cmds):
    add_shape(slide, Inches(0.5), y, Inches(12.3), Inches(0.31), fill_color=LIGHT_GRAY if idx % 2 == 0 else WHITE)
    add_text_box(slide, Inches(0.7), y + Inches(0.01), Inches(3.0), Inches(0.3), name, font_size=13, bold=True, color=IIT_BLUE)
    add_text_box(slide, Inches(3.8), y + Inches(0.01), Inches(8.9), Inches(0.3), cmd, font_size=11, color=DARK_GRAY, font_name='Consolas')
    y += Inches(0.315)

# ============================================================
# SLIDE 25: Thank You / Questions
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
