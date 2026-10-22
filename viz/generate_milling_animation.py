"""
viz/generate_milling_animation.py

Generates an industrial-grade, fully labeled animation video of the CNC milling
digital twin driven by the trained PPO policy (MillingDTEnv-v1).

Visual features:
- Vertical Machining Center 2D cross-sectional milling view.
- Multi-pass stock removal: material dynamically carved away pass-by-pass.
- Tool assembly: HSK spindle toolholder, 50mm face milling cutter body,
  and 4 rotating carbide flutes/inserts spinning at real-time spindle RPM.
- Dynamic chip shear zone: animated glowing metallic chips ejected during cut.
- Rapid traverse retract and repositioning (G00).
- Industrial Telemetry HUD: instantaneous PPO action recommendations (n, fr, ap),
  process physical metrics (MRR, Power, Energy), and safety constraint gauges
  with margin thresholds for Flank Wear (VB) and Surface Roughness (Ra).
- Multi-pass progression strip charts at bottom.
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import cv2
except ImportError:
    cv2 = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from MillingDT.sim.machining_sim import load_ppo_policy, run_single_simulation
from MillingDT.config import PRIMARY_MODEL_DIR

def render_frame(
    fig,
    ax_mill,
    ax_hud,
    ax_strip,
    df_passes: pd.DataFrame,
    current_pass_idx: int,
    tool_x: float,
    tool_z_bottom: float,
    is_rapid: bool,
    rotation_deg: float,
    frame_idx: int,
    total_frames: int,
    L_cut: float = 100.0,
    kappa_total: float = 5.0,
    D_tool: float = 50.0
):
    """Draws a single high-fidelity, industrial labeled frame onto the figure."""
    ax_mill.clear()
    ax_hud.clear()
    ax_strip.clear()

    # Current pass data
    row = df_passes.iloc[current_pass_idx]
    pass_num = int(row['pass'])
    total_passes = len(df_passes)
    n_rpm = float(row['n_rpm'])
    f_r = float(row['f_r_mm_rev'])
    ap_current = float(row['a_p_mm'])
    v_c = float(row['v_c_m_min'])
    mrr = float(row['MRR_mm3_min']) / 1000.0  # cm3/min
    power = float(row['cutting_power_W'])
    energy = float(row['cum_energy_kJ'])
    vb = float(row['VB_mm'])
    ra = float(row['Ra_um'])
    is_finishing = bool(row['is_finishing'])

    # All pass depths and cumulative depths
    all_ap = df_passes['a_p_mm'].tolist()
    z_baseline = 0.0

    # -------------------------------------------------------------------------
    # 1. Main Milling Viewport (Left Area)
    # -------------------------------------------------------------------------
    ax_mill.set_facecolor('#0f172a')
    ax_mill.set_xlim(-25, L_cut + 35)
    ax_mill.set_ylim(-4.5, kappa_total + 18.0)
    ax_mill.grid(True, linestyle=':', color='#334155', alpha=0.35)

    # 1A. Finished Workpiece Substrate (Z <= 0)
    rect_sub = patches.Rectangle(
        (0, -3.5), L_cut, 3.5,
        facecolor='#1e293b', edgecolor='#475569', linewidth=2.0, hatch='///', zorder=2
    )
    ax_mill.add_patch(rect_sub)
    ax_mill.text(
        L_cut / 2, -1.75, "Finished Workpiece Substrate (Reference Z = 0.0 mm)",
        color='#94a3b8', fontsize=10, ha='center', va='center', fontweight='bold', zorder=3
    )

    # 1B. Workpiece Stock Layers
    # Calculate layer boundaries
    # Layer 0 is at top: [kappa_total - ap_0, kappa_total]
    cum_depths = [0.0] + list(np.cumsum(all_ap))

    # Palette for passes
    pass_colors = ['#0284c7', '#0d9488', '#10b981', '#f59e0b', '#8b5cf6']

    # Draw remaining stock layers that are below current pass
    for p_i in range(current_pass_idx + 1, total_passes):
        z_bot = max(0.0, kappa_total - cum_depths[p_i + 1])
        z_top = kappa_total - cum_depths[p_i]
        rect_fut = patches.Rectangle(
            (0, z_bot), L_cut, z_top - z_bot,
            facecolor='#334155', edgecolor='#475569', linewidth=1.2, alpha=0.6, zorder=2
        )
        ax_mill.add_patch(rect_fut)
        if (z_top - z_bot) >= 1.0:
            ax_mill.text(
                L_cut * 0.85, (z_top + z_bot) / 2, f"Pass {p_i+1} Stock",
                color='#94a3b8', fontsize=8, ha='center', va='center', zorder=3
            )

    # Current pass layer stock
    cur_z_bot = max(0.0, kappa_total - cum_depths[current_pass_idx + 1])
    cur_z_top = kappa_total - cum_depths[current_pass_idx]

    # Material ahead of tool (uncut)
    x_cut = max(0.0, min(tool_x, L_cut))
    if x_cut < L_cut:
        rect_ahead = patches.Rectangle(
            (x_cut, cur_z_bot), L_cut - x_cut, cur_z_top - cur_z_bot,
            facecolor=pass_colors[current_pass_idx % len(pass_colors)],
            edgecolor='#e2e8f0', linewidth=1.5, alpha=0.9, zorder=2
        )
        ax_mill.add_patch(rect_ahead)
        if (L_cut - x_cut) > 28:
            pass_label = "ROUGHING" if not is_finishing else "FINISHING (ISO 1302)"
            ax_mill.text(
                (x_cut + L_cut) / 2, (cur_z_top + cur_z_bot) / 2,
                f"Pass {pass_num} {pass_label} (ap = {ap_current:.2f} mm)",
                color='#ffffff', fontsize=9, ha='center', va='center', fontweight='bold', zorder=3
            )

    # Freshly Machined Surface behind tool
    if x_cut > 0:
        surf_color = '#38bdf8' if not is_finishing else '#34d399'
        ax_mill.plot([0, x_cut], [cur_z_bot, cur_z_bot], color=surf_color, linewidth=2.8, zorder=4)
        if is_finishing and x_cut > 20:
            ax_mill.text(
                x_cut / 2, cur_z_bot + 0.6, f"Machined Surface (Ra = {ra:.2f} um)",
                color=surf_color, fontsize=9, ha='center', va='bottom', fontweight='bold', zorder=5
            )

    # 1C. Tool Assembly (Spindle, Toolholder, Cutter Body, Flutes)
    cutter_r = 10.0   # Display radius in plot units (representing D=50mm)
    cutter_h = 9.0    # Height of cutter body
    z_cut_bot = tool_z_bottom

    # Toolholder / Spindle geometry
    holder_h = 7.0
    holder_w_top = cutter_r * 2.8
    holder_w_bot = cutter_r * 2.2
    holder_pts = [
        [tool_x - holder_w_bot/2, z_cut_bot + cutter_h],
        [tool_x - holder_w_top/2, z_cut_bot + cutter_h + holder_h],
        [tool_x + holder_w_top/2, z_cut_bot + cutter_h + holder_h],
        [tool_x + holder_w_bot/2, z_cut_bot + cutter_h]
    ]
    poly_holder = patches.Polygon(holder_pts, facecolor='#475569', edgecolor='#94a3b8', linewidth=1.5, zorder=6)
    ax_mill.add_patch(poly_holder)
    ax_mill.text(
        tool_x, z_cut_bot + cutter_h + holder_h * 0.5, "HSK-A63 Spindle",
        color='#e2e8f0', fontsize=8, ha='center', va='center', fontweight='bold', zorder=7
    )

    # Cutter Cylinder Body
    rect_cutter = patches.Rectangle(
        (tool_x - cutter_r, z_cut_bot), cutter_r * 2, cutter_h,
        facecolor='#334155', edgecolor='#94a3b8', linewidth=1.6, zorder=6
    )
    ax_mill.add_patch(rect_cutter)

    # Rotating Flutes / Inserts (4 Flutes)
    n_flutes = 4
    for fl_i in range(n_flutes):
        angle = np.radians(rotation_deg + fl_i * (360 / n_flutes))
        flute_x_offset = cutter_r * 0.85 * np.cos(angle)
        flute_w = cutter_r * 0.28 * abs(np.sin(angle)) + 0.6
        fl_col = '#f59e0b' if np.sin(angle) >= 0 else '#b45309'
        rect_flute = patches.Rectangle(
            (tool_x + flute_x_offset - flute_w/2, z_cut_bot), flute_w, cutter_h,
            facecolor=fl_col, edgecolor='#fcd34d', linewidth=0.8, alpha=0.9, zorder=7
        )
        ax_mill.add_patch(rect_flute)

    # Carbide Insert Cutting Tips
    tip_l = patches.Rectangle((tool_x - cutter_r, z_cut_bot), 2.2, 1.2, facecolor='#f59e0b', edgecolor='#fcd34d', zorder=8)
    tip_r = patches.Rectangle((tool_x + cutter_r - 2.2, z_cut_bot), 2.2, 1.2, facecolor='#f59e0b', edgecolor='#fcd34d', zorder=8)
    ax_mill.add_patch(tip_l)
    ax_mill.add_patch(tip_r)

    # 1D. Dynamic Chip Ejection (Active during feed cut inside stock)
    if not is_rapid and 0 <= tool_x <= L_cut:
        np.random.seed(int(tool_x * 17 + frame_idx) % 10000)
        n_chips = 16
        chip_dx = np.random.uniform(2.0, 18.0, n_chips)
        chip_dz = np.random.uniform(0.5, 9.0, n_chips)
        chip_sizes = np.random.uniform(12, 45, n_chips)
        chip_cols = ['#f59e0b', '#fbbf24', '#ef4444', '#f97316']
        ax_mill.scatter(
            tool_x + chip_dx, z_cut_bot + chip_dz,
            s=chip_sizes, c=[chip_cols[i % len(chip_cols)] for i in range(n_chips)],
            marker='o', alpha=0.85, zorder=12
        )
        ax_mill.text(
            tool_x + 12, z_cut_bot + 7.5, "Shear Zone",
            color='#fbbf24', fontsize=8, style='italic', fontweight='bold', zorder=12
        )

    # 1E. Labels & Annotations with Arrows
    # Tool diameter and Spindle Speed
    ax_mill.annotate(
        f"Face Mill Cutter D = {D_tool:.0f} mm\nSpindle n = {n_rpm:.0f} RPM",
        xy=(tool_x - cutter_r, z_cut_bot + cutter_h * 0.7),
        xytext=(tool_x - 30, z_cut_bot + cutter_h + 3.0),
        arrowprops=dict(arrowstyle="->", color='#38bdf8', lw=1.2),
        color='#38bdf8', fontsize=8, fontweight='bold', zorder=10
    )

    # Feed Direction Arrow (G01) or Rapid Traverse (G00)
    vf = n_rpm * f_r
    if not is_rapid:
        ax_mill.annotate(
            f"Feed vf = {vf:.0f} mm/min",
            xy=(tool_x + cutter_r + 2, z_cut_bot + 1.2),
            xytext=(tool_x + cutter_r + 14, z_cut_bot + 1.2),
            arrowprops=dict(arrowstyle="<-", color='#10b981', lw=1.8),
            color='#10b981', fontsize=8, fontweight='bold', zorder=10
        )
    else:
        ax_mill.annotate(
            "RAPID G00 RETRACT",
            xy=(tool_x - cutter_r - 2, z_cut_bot + 1.2),
            xytext=(tool_x - cutter_r - 18, z_cut_bot + 1.2),
            arrowprops=dict(arrowstyle="<-", color='#f59e0b', lw=1.8),
            color='#f59e0b', fontsize=8, fontweight='bold', zorder=10
        )

    # Workpiece Allowance Dimension (left bracket)
    ax_mill.annotate(
        '', xy=(-5, 0), xytext=(-5, kappa_total),
        arrowprops=dict(arrowstyle='<->', color='#94a3b8', lw=1.4)
    )
    ax_mill.text(-7, kappa_total / 2, f"Stock Allowance\nkappa = {kappa_total:.1f} mm",
                 color='#94a3b8', fontsize=8, ha='right', va='center')

    # Workpiece Length Dimension (bottom bracket)
    ax_mill.annotate(
        '', xy=(0, -3.8), xytext=(L_cut, -3.8),
        arrowprops=dict(arrowstyle='<->', color='#94a3b8', lw=1.4)
    )
    ax_mill.text(L_cut / 2, -4.8, f"Workpiece Cut Length L = {L_cut:.0f} mm",
                 color='#94a3b8', fontsize=8, ha='center', va='top')

    ax_mill.set_title("CNC Vertical Machining Center • 2D Tool-Workpiece Cut Cross-Section",
                      color='#f8fafc', fontsize=12, fontweight='bold', pad=10)
    ax_mill.set_xlabel("Workpiece Longitudinal Feed Axis X (mm)", color='#94a3b8', fontsize=9)
    ax_mill.set_ylabel("Axial Depth Axis Z (mm)", color='#94a3b8', fontsize=9)
    ax_mill.tick_params(colors='#94a3b8', labelsize=8)

    # -------------------------------------------------------------------------
    # 2. Industrial Telemetry HUD (Right Panel)
    # -------------------------------------------------------------------------
    ax_hud.set_facecolor('#0f172a')
    ax_hud.set_xlim(0, 100)
    ax_hud.set_ylim(0, 100)
    ax_hud.axis('off')

    # HUD Card Background
    card_bg = patches.FancyBboxPatch(
        (2, 2), 96, 96, boxstyle="round,pad=2",
        facecolor='#1e293b', edgecolor='#334155', linewidth=1.5
    )
    ax_hud.add_patch(card_bg)

    # Card Title
    ax_hud.text(50, 94, "TELEMETRY & CONSTRAINTS", color='#38bdf8',
                fontsize=11, fontweight='bold', ha='center', va='top')
    ax_hud.plot([10, 90], [89, 89], color='#334155', lw=1.2)

    # Pass Status Badge
    badge_color = '#10b981' if not is_rapid else '#f59e0b'
    status_label = "ROUGHING PASS" if not is_finishing else "FINISHING PASS"
    if is_rapid:
        status_label = "RAPID RETRACT (G00)"
    badge_text = f"PASS {pass_num}/{total_passes} • {status_label}"

    badge_rect = patches.FancyBboxPatch(
        (10, 80), 80, 6.5, boxstyle="round,pad=1",
        facecolor=badge_color, edgecolor=badge_color, alpha=0.25
    )
    ax_hud.add_patch(badge_rect)
    ax_hud.text(50, 83.2, badge_text, color=badge_color, fontsize=8.5, fontweight='bold', ha='center', va='center')

    # DRL Recommended Action Parameters
    y_pos = 73
    ax_hud.text(12, y_pos, "DRL PPO RECOMMENDED ACTIONS:", color='#94a3b8', fontsize=8, fontweight='bold')
    y_pos -= 5.5
    ax_hud.text(15, y_pos, f"Spindle Speed n: {n_rpm:.0f} RPM", color='#f8fafc', fontsize=8.5)
    y_pos -= 4.5
    ax_hud.text(15, y_pos, f"Feed Rate fr: {f_r:.3f} mm/rev", color='#f8fafc', fontsize=8.5)
    y_pos -= 4.5
    ax_hud.text(15, y_pos, f"Axial Depth ap: {ap_current:.2f} mm", color='#f8fafc', fontsize=8.5)
    y_pos -= 4.5
    ax_hud.text(15, y_pos, f"Cutting Speed vc: {v_c:.1f} m/min", color='#94a3b8', fontsize=8)

    # Physical Process Telemetry
    y_pos -= 6.5
    ax_hud.text(12, y_pos, "PHYSICAL PROCESS TELEMETRY:", color='#94a3b8', fontsize=8, fontweight='bold')
    y_pos -= 5.5
    ax_hud.text(15, y_pos, f"MRR: {mrr:.1f} cm3/min", color='#f8fafc', fontsize=8.5)
    y_pos -= 4.5
    ax_hud.text(15, y_pos, f"Cutting Power: {power:.0f} W", color='#f8fafc', fontsize=8.5)
    y_pos -= 4.5
    ax_hud.text(15, y_pos, f"Energy (Cum.): {energy:.1f} kJ", color='#f8fafc', fontsize=8.5)

    # Constraints Gauges
    # 1. Tool Flank Wear VB (limit = 0.30 mm, safe = 0.255 mm)
    y_pos -= 6.5
    ax_hud.text(12, y_pos, "TOOL FLANK WEAR VB:", color='#94a3b8', fontsize=8, fontweight='bold')
    y_pos -= 3.5
    bar_w = 76
    ax_hud.add_patch(patches.Rectangle((12, y_pos - 2.5), bar_w, 3.5, facecolor='#0f172a', edgecolor='#475569', lw=1))
    vb_frac = min(vb / 0.30, 1.0)
    vb_color = '#10b981' if vb <= 0.255 else '#ef4444'
    ax_hud.add_patch(patches.Rectangle((12, y_pos - 2.5), bar_w * vb_frac, 3.5, facecolor=vb_color))
    # Safety limit marker (0.255 / 0.30 = 0.85)
    ax_hud.plot([12 + bar_w * 0.85, 12 + bar_w * 0.85], [y_pos - 3.5, y_pos + 2.0], color='#f59e0b', lw=1.5)
    y_pos -= 5.5
    ax_hud.text(15, y_pos, f"VB = {vb:.3f} mm  (Safe < 0.255 mm | Limit 0.30 mm)", color=vb_color, fontsize=8, fontweight='bold')

    # 2. Surface Roughness Ra (limit = 1.60 um, safe = 1.36 um)
    y_pos -= 6.0
    ax_hud.text(12, y_pos, "SURFACE ROUGHNESS Ra:", color='#94a3b8', fontsize=8, fontweight='bold')
    y_pos -= 3.5
    ax_hud.add_patch(patches.Rectangle((12, y_pos - 2.5), bar_w, 3.5, facecolor='#0f172a', edgecolor='#475569', lw=1))
    ra_frac = min(ra / 1.60, 1.0)
    ra_color = '#10b981' if ra <= 1.36 else ('#f59e0b' if ra <= 1.60 else '#ef4444')
    ax_hud.add_patch(patches.Rectangle((12, y_pos - 2.5), bar_w * ra_frac, 3.5, facecolor=ra_color))
    ax_hud.plot([12 + bar_w * 0.85, 12 + bar_w * 0.85], [y_pos - 3.5, y_pos + 2.0], color='#f59e0b', lw=1.5)
    y_pos -= 5.5
    ax_hud.text(15, y_pos, f"Ra = {ra:.2f} um  (Safe < 1.36 um | Spec <= 1.60 um)", color=ra_color, fontsize=8, fontweight='bold')

    # -------------------------------------------------------------------------
    # 3. Bottom Strip Charts: Telemetry Across Passes
    # -------------------------------------------------------------------------
    ax_strip.set_facecolor('#0f172a')
    ax_strip.tick_params(colors='#94a3b8', labelsize=8)
    ax_strip.grid(True, linestyle=':', color='#334155', alpha=0.4)

    all_passes = df_passes['pass'].tolist()
    all_vbs = df_passes['VB_mm'].tolist()
    all_ras = df_passes['Ra_um'].tolist()

    ax_strip_right = ax_strip.twinx()
    ax_strip_right.tick_params(colors='#38bdf8', labelsize=8)

    ax_strip.plot(all_passes, all_vbs, 'o-', color='#10b981', lw=2, label="Flank Wear VB (mm)")
    ax_strip_right.plot(all_passes, all_ras, 's--', color='#38bdf8', lw=2, label="Surface Roughness Ra (um)")

    # Highlight current pass marker
    ax_strip.plot([pass_num], [vb], marker='o', markersize=10, color='#fcd34d', zorder=10)
    ax_strip_right.plot([pass_num], [ra], marker='s', markersize=10, color='#fcd34d', zorder=10)

    ax_strip.axhline(0.255, color='#f59e0b', linestyle=':', label="VB Safe (0.255 mm)")
    ax_strip_right.axhline(1.60, color='#ef4444', linestyle=':', label="Ra Limit (1.60 um)")

    ax_strip.set_xlim(0.7, len(all_passes) + 0.3)
    ax_strip.set_xticks(all_passes)
    ax_strip.set_xticklabels([f"Pass {p}" for p in all_passes])
    ax_strip.set_ylabel("Tool Wear VB (mm)", color='#10b981', fontsize=8)
    ax_strip_right.set_ylabel("Roughness Ra (um)", color='#38bdf8', fontsize=8)
    ax_strip.set_title("Multi-Pass Cumulative State Progression (PPO Policy Trajectory)",
                       color='#e2e8f0', fontsize=9, fontweight='bold')

    # Top Title Banner
    fig.suptitle(
        f"AUTONOMOUS CNC FACE MILLING DIGITAL TWIN • DRL (PPO) REAL-TIME CONTROL  "
        f"[Frame {frame_idx+1}/{total_frames}]",
        fontsize=13, fontweight='bold', color='#f8fafc', y=0.98
    )


def render_pass_background(
    fig,
    ax_mill,
    ax_hud,
    ax_strip,
    ax_strip_right,
    df_passes: pd.DataFrame,
    current_pass_idx: int,
    is_rapid: bool,
    L_cut: float = 100.0,
    kappa_total: float = 5.0,
    D_tool: float = 50.0
):
    """Renders all static elements for a given pass & phase and caches the canvas."""
    ax_mill.clear()
    ax_hud.clear()
    ax_strip.clear()
    ax_strip_right.clear()

    row = df_passes.iloc[current_pass_idx]
    pass_num = int(row['pass'])
    total_passes = len(df_passes)
    n_rpm = float(row['n_rpm'])
    f_r = float(row['f_r_mm_rev'])
    ap_current = float(row['a_p_mm'])
    v_c = float(row['v_c_m_min'])
    mrr = float(row['MRR_mm3_min']) / 1000.0  # cm3/min
    power = float(row['cutting_power_W'])
    energy = float(row['cum_energy_kJ'])
    vb = float(row['VB_mm'])
    ra = float(row['Ra_um'])
    is_finishing = bool(row['is_finishing'])

    all_ap = df_passes['a_p_mm'].tolist()
    cum_depths = [0.0] + list(np.cumsum(all_ap))
    cur_z_bot = max(0.0, kappa_total - cum_depths[current_pass_idx + 1])

    # 1. Main Milling Viewport
    ax_mill.set_facecolor('#0f172a')
    ax_mill.set_xlim(-25, L_cut + 35)
    ax_mill.set_ylim(-4.5, kappa_total + 18.0)
    ax_mill.grid(True, linestyle=':', color='#334155', alpha=0.35)

    # 1A. Finished Workpiece Substrate (Z <= 0)
    rect_sub = patches.Rectangle(
        (0, -3.5), L_cut, 3.5,
        facecolor='#1e293b', edgecolor='#475569', linewidth=2.0, hatch='///', zorder=2
    )
    ax_mill.add_patch(rect_sub)
    ax_mill.text(
        L_cut / 2, -1.75, "Finished Workpiece Substrate (Reference Z = 0.0 mm)",
        color='#94a3b8', fontsize=10, ha='center', va='center', fontweight='bold', zorder=3
    )

    # 1B. Workpiece Stock Layers (Future layers below current)
    for p_i in range(current_pass_idx + 1, total_passes):
        z_bot = max(0.0, kappa_total - cum_depths[p_i + 1])
        z_top = kappa_total - cum_depths[p_i]
        rect_fut = patches.Rectangle(
            (0, z_bot), L_cut, z_top - z_bot,
            facecolor='#334155', edgecolor='#475569', linewidth=1.2, alpha=0.6, zorder=2
        )
        ax_mill.add_patch(rect_fut)
        if (z_top - z_bot) >= 1.0:
            ax_mill.text(
                L_cut * 0.85, (z_top + z_bot) / 2, f"Pass {p_i+1} Stock",
                color='#94a3b8', fontsize=8, ha='center', va='center', zorder=3
            )

    # If in rapid traverse or completed pass, current layer is already cut:
    if is_rapid:
        surf_color = '#38bdf8' if not is_finishing else '#34d399'
        ax_mill.plot([0, L_cut], [cur_z_bot, cur_z_bot], color=surf_color, linewidth=2.8, zorder=4)
        if is_finishing:
            ax_mill.text(
                L_cut / 2, cur_z_bot + 0.6, f"Machined Surface (Ra = {ra:.2f} um)",
                color=surf_color, fontsize=9, ha='center', va='bottom', fontweight='bold', zorder=5
            )

    # Brackets & Dimension Annotations
    ax_mill.annotate(
        '', xy=(-5, 0), xytext=(-5, kappa_total),
        arrowprops=dict(arrowstyle='<->', color='#94a3b8', lw=1.4)
    )
    ax_mill.text(-7, kappa_total / 2, f"Stock Allowance\nkappa = {kappa_total:.1f} mm",
                 color='#94a3b8', fontsize=8, ha='right', va='center')

    ax_mill.annotate(
        '', xy=(0, -3.8), xytext=(L_cut, -3.8),
        arrowprops=dict(arrowstyle='<->', color='#94a3b8', lw=1.4)
    )
    ax_mill.text(L_cut / 2, -4.8, f"Workpiece Cut Length L = {L_cut:.0f} mm",
                 color='#94a3b8', fontsize=8, ha='center', va='top')

    ax_mill.set_title("CNC Vertical Machining Center • 2D Tool-Workpiece Cut Cross-Section",
                      color='#f8fafc', fontsize=12, fontweight='bold', pad=10)
    ax_mill.set_xlabel("Workpiece Longitudinal Feed Axis X (mm)", color='#94a3b8', fontsize=9)
    ax_mill.set_ylabel("Axial Depth Axis Z (mm)", color='#94a3b8', fontsize=9)
    ax_mill.tick_params(colors='#94a3b8', labelsize=8)

    # 2. Industrial Telemetry HUD
    ax_hud.set_facecolor('#0f172a')
    ax_hud.set_xlim(0, 100)
    ax_hud.set_ylim(0, 100)
    ax_hud.axis('off')

    card_bg = patches.FancyBboxPatch(
        (2, 2), 96, 96, boxstyle="round,pad=2",
        facecolor='#1e293b', edgecolor='#334155', linewidth=1.5
    )
    ax_hud.add_patch(card_bg)

    ax_hud.text(50, 94, "TELEMETRY & CONSTRAINTS", color='#38bdf8',
                fontsize=11, fontweight='bold', ha='center', va='top')
    ax_hud.plot([10, 90], [89, 89], color='#334155', lw=1.2)

    badge_color = '#10b981' if not is_rapid else '#f59e0b'
    status_label = "ROUGHING PASS" if not is_finishing else "FINISHING PASS"
    if is_rapid:
        status_label = "RAPID RETRACT (G00)"
    badge_text = f"PASS {pass_num}/{total_passes} • {status_label}"

    badge_rect = patches.FancyBboxPatch(
        (10, 80), 80, 6.5, boxstyle="round,pad=1",
        facecolor=badge_color, edgecolor=badge_color, alpha=0.25
    )
    ax_hud.add_patch(badge_rect)
    ax_hud.text(50, 83.2, badge_text, color=badge_color, fontsize=8.5, fontweight='bold', ha='center', va='center')

    y_pos = 73
    ax_hud.text(12, y_pos, "DRL PPO RECOMMENDED ACTIONS:", color='#94a3b8', fontsize=8, fontweight='bold')
    y_pos -= 5.5
    ax_hud.text(15, y_pos, f"Spindle Speed n: {n_rpm:.0f} RPM", color='#f8fafc', fontsize=8.5)
    y_pos -= 4.5
    ax_hud.text(15, y_pos, f"Feed Rate fr: {f_r:.3f} mm/rev", color='#f8fafc', fontsize=8.5)
    y_pos -= 4.5
    ax_hud.text(15, y_pos, f"Axial Depth ap: {ap_current:.2f} mm", color='#f8fafc', fontsize=8.5)
    y_pos -= 4.5
    ax_hud.text(15, y_pos, f"Cutting Speed vc: {v_c:.1f} m/min", color='#94a3b8', fontsize=8)

    y_pos -= 6.5
    ax_hud.text(12, y_pos, "PHYSICAL PROCESS TELEMETRY:", color='#94a3b8', fontsize=8, fontweight='bold')
    y_pos -= 5.5
    ax_hud.text(15, y_pos, f"MRR: {mrr:.1f} cm3/min", color='#f8fafc', fontsize=8.5)
    y_pos -= 4.5
    ax_hud.text(15, y_pos, f"Cutting Power: {power:.0f} W", color='#f8fafc', fontsize=8.5)
    y_pos -= 4.5
    ax_hud.text(15, y_pos, f"Energy (Cum.): {energy:.1f} kJ", color='#f8fafc', fontsize=8.5)

    # Constraints Gauges
    y_pos -= 6.5
    ax_hud.text(12, y_pos, "TOOL FLANK WEAR VB:", color='#94a3b8', fontsize=8, fontweight='bold')
    y_pos -= 3.5
    bar_w = 76
    ax_hud.add_patch(patches.Rectangle((12, y_pos - 2.5), bar_w, 3.5, facecolor='#0f172a', edgecolor='#475569', lw=1))
    vb_frac = min(vb / 0.30, 1.0)
    vb_color = '#10b981' if vb <= 0.255 else '#ef4444'
    ax_hud.add_patch(patches.Rectangle((12, y_pos - 2.5), bar_w * vb_frac, 3.5, facecolor=vb_color))
    ax_hud.plot([12 + bar_w * 0.85, 12 + bar_w * 0.85], [y_pos - 3.5, y_pos + 2.0], color='#f59e0b', lw=1.5)
    y_pos -= 5.5
    ax_hud.text(15, y_pos, f"VB = {vb:.3f} mm  (Safe < 0.255 mm | Limit 0.30 mm)", color=vb_color, fontsize=8, fontweight='bold')

    y_pos -= 6.0
    ax_hud.text(12, y_pos, "SURFACE ROUGHNESS Ra:", color='#94a3b8', fontsize=8, fontweight='bold')
    y_pos -= 3.5
    ax_hud.add_patch(patches.Rectangle((12, y_pos - 2.5), bar_w, 3.5, facecolor='#0f172a', edgecolor='#475569', lw=1))
    ra_frac = min(ra / 1.60, 1.0)
    ra_color = '#10b981' if ra <= 1.36 else ('#f59e0b' if ra <= 1.60 else '#ef4444')
    ax_hud.add_patch(patches.Rectangle((12, y_pos - 2.5), bar_w * ra_frac, 3.5, facecolor=ra_color))
    ax_hud.plot([12 + bar_w * 0.85, 12 + bar_w * 0.85], [y_pos - 3.5, y_pos + 2.0], color='#f59e0b', lw=1.5)
    y_pos -= 5.5
    ax_hud.text(15, y_pos, f"Ra = {ra:.2f} um  (Safe < 1.36 um | Limit 1.60 um)", color=ra_color, fontsize=8, fontweight='bold')

    # 3. Bottom Strip Charts
    ax_strip.set_facecolor('#0f172a')
    ax_strip.tick_params(colors='#94a3b8', labelsize=8)
    ax_strip.grid(True, linestyle=':', color='#334155', alpha=0.4)

    all_passes = df_passes['pass'].tolist()
    all_vbs = df_passes['VB_mm'].tolist()
    all_ras = df_passes['Ra_um'].tolist()

    ax_strip_right.tick_params(colors='#38bdf8', labelsize=8)

    ax_strip.plot(all_passes, all_vbs, 'o-', color='#10b981', lw=2, label="Flank Wear VB (mm)")
    ax_strip_right.plot(all_passes, all_ras, 's--', color='#38bdf8', lw=2, label="Surface Roughness Ra (um)")

    ax_strip.plot([pass_num], [vb], marker='o', markersize=10, color='#fcd34d', zorder=10)
    ax_strip_right.plot([pass_num], [ra], marker='s', markersize=10, color='#fcd34d', zorder=10)

    ax_strip.axhline(0.255, color='#f59e0b', linestyle=':', label="VB Safe (0.255 mm)")
    ax_strip_right.axhline(1.60, color='#ef4444', linestyle=':', label="Ra Limit (1.60 um)")

    ax_strip.set_xlim(0.7, len(all_passes) + 0.3)
    ax_strip.set_xticks(all_passes)
    ax_strip.set_xticklabels([f"Pass {p}" for p in all_passes])
    ax_strip.set_ylabel("Tool Wear VB (mm)", color='#10b981', fontsize=8)
    ax_strip_right.set_ylabel("Roughness Ra (um)", color='#38bdf8', fontsize=8)
    ax_strip.set_title("Multi-Pass Cumulative State Progression (PPO Policy Trajectory)",
                       color='#e2e8f0', fontsize=9, fontweight='bold')

    # Rasterize static background
    fig.canvas.draw()
    return fig.canvas.copy_from_bbox(fig.bbox)


def render_dynamic_frame(
    fig,
    ax_mill,
    bg,
    df_passes: pd.DataFrame,
    current_pass_idx: int,
    tool_x: float,
    tool_z_bottom: float,
    is_rapid: bool,
    rotation_deg: float,
    frame_idx: int,
    total_frames: int,
    L_cut: float = 100.0,
    kappa_total: float = 5.0,
    D_tool: float = 50.0
):
    """Draws dynamic components on top of the cached background using blitting."""
    fig.canvas.restore_region(bg)
    dynamic_artists = []

    row = df_passes.iloc[current_pass_idx]
    pass_num = int(row['pass'])
    n_rpm = float(row['n_rpm'])
    f_r = float(row['f_r_mm_rev'])
    ap_current = float(row['a_p_mm'])
    ra = float(row['Ra_um'])
    is_finishing = bool(row['is_finishing'])

    all_ap = df_passes['a_p_mm'].tolist()
    cum_depths = [0.0] + list(np.cumsum(all_ap))
    cur_z_bot = max(0.0, kappa_total - cum_depths[current_pass_idx + 1])
    cur_z_top = kappa_total - cum_depths[current_pass_idx]

    pass_colors = ['#0284c7', '#0d9488', '#d97706', '#7c3aed', '#dc2626']

    # 1. Dynamic Stock Ahead & Machined Line (Cut Phase only)
    if not is_rapid:
        x_cut = max(0.0, min(tool_x, L_cut))
        if x_cut < L_cut:
            rect_ahead = patches.Rectangle(
                (x_cut, cur_z_bot), L_cut - x_cut, cur_z_top - cur_z_bot,
                facecolor=pass_colors[current_pass_idx % len(pass_colors)],
                edgecolor='#e2e8f0', linewidth=1.5, alpha=0.9, zorder=2
            )
            ax_mill.add_patch(rect_ahead)
            ax_mill.draw_artist(rect_ahead)
            dynamic_artists.append(rect_ahead)

            if (L_cut - x_cut) > 28:
                pass_label = "ROUGHING" if not is_finishing else "FINISHING (ISO 1302)"
                t_ahead = ax_mill.text(
                    (x_cut + L_cut) / 2, (cur_z_top + cur_z_bot) / 2,
                    f"Pass {pass_num} {pass_label} (ap = {ap_current:.2f} mm)",
                    color='#ffffff', fontsize=9, ha='center', va='center', fontweight='bold', zorder=3
                )
                ax_mill.draw_artist(t_ahead)
                dynamic_artists.append(t_ahead)

        if x_cut > 0:
            surf_color = '#38bdf8' if not is_finishing else '#34d399'
            l_surf, = ax_mill.plot([0, x_cut], [cur_z_bot, cur_z_bot], color=surf_color, linewidth=2.8, zorder=4)
            ax_mill.draw_artist(l_surf)
            dynamic_artists.append(l_surf)
            if is_finishing and x_cut > 20:
                t_surf = ax_mill.text(
                    x_cut / 2, cur_z_bot + 0.6, f"Machined Surface (Ra = {ra:.2f} um)",
                    color=surf_color, fontsize=9, ha='center', va='bottom', fontweight='bold', zorder=5
                )
                ax_mill.draw_artist(t_surf)
                dynamic_artists.append(t_surf)

    # 2. Tool Assembly
    cutter_r = 10.0
    cutter_h = 9.0
    z_cut_bot = tool_z_bottom

    holder_h = 7.0
    holder_w_top = cutter_r * 2.8
    holder_w_bot = cutter_r * 2.2
    holder_pts = [
        [tool_x - holder_w_bot/2, z_cut_bot + cutter_h],
        [tool_x - holder_w_top/2, z_cut_bot + cutter_h + holder_h],
        [tool_x + holder_w_top/2, z_cut_bot + cutter_h + holder_h],
        [tool_x + holder_w_bot/2, z_cut_bot + cutter_h]
    ]
    poly_holder = patches.Polygon(holder_pts, facecolor='#475569', edgecolor='#94a3b8', linewidth=1.5, zorder=6)
    ax_mill.add_patch(poly_holder)
    ax_mill.draw_artist(poly_holder)
    dynamic_artists.append(poly_holder)

    t_holder = ax_mill.text(
        tool_x, z_cut_bot + cutter_h + holder_h * 0.5, "HSK-A63 Spindle",
        color='#e2e8f0', fontsize=8, ha='center', va='center', fontweight='bold', zorder=7
    )
    ax_mill.draw_artist(t_holder)
    dynamic_artists.append(t_holder)

    rect_cutter = patches.Rectangle(
        (tool_x - cutter_r, z_cut_bot), cutter_r * 2, cutter_h,
        facecolor='#334155', edgecolor='#94a3b8', linewidth=1.6, zorder=6
    )
    ax_mill.add_patch(rect_cutter)
    ax_mill.draw_artist(rect_cutter)
    dynamic_artists.append(rect_cutter)

    n_flutes = 4
    for fl_i in range(n_flutes):
        angle = np.radians(rotation_deg + fl_i * (360 / n_flutes))
        flute_x_offset = cutter_r * 0.85 * np.cos(angle)
        flute_w = cutter_r * 0.28 * abs(np.sin(angle)) + 0.6
        fl_col = '#f59e0b' if np.sin(angle) >= 0 else '#b45309'
        rect_flute = patches.Rectangle(
            (tool_x + flute_x_offset - flute_w/2, z_cut_bot), flute_w, cutter_h,
            facecolor=fl_col, edgecolor='#fcd34d', linewidth=0.8, alpha=0.9, zorder=7
        )
        ax_mill.add_patch(rect_flute)
        ax_mill.draw_artist(rect_flute)
        dynamic_artists.append(rect_flute)

    tip_l = patches.Rectangle((tool_x - cutter_r, z_cut_bot), 2.2, 1.2, facecolor='#f59e0b', edgecolor='#fcd34d', zorder=8)
    tip_r = patches.Rectangle((tool_x + cutter_r - 2.2, z_cut_bot), 2.2, 1.2, facecolor='#f59e0b', edgecolor='#fcd34d', zorder=8)
    ax_mill.add_patch(tip_l)
    ax_mill.add_patch(tip_r)
    ax_mill.draw_artist(tip_l)
    ax_mill.draw_artist(tip_r)
    dynamic_artists.extend([tip_l, tip_r])

    # 3. Dynamic Chips (Shear Zone)
    if not is_rapid and (-5.0 <= tool_x <= L_cut):
        np.random.seed((frame_idx * 17) % 10000)
        n_chips = 14
        chip_x = tool_x + np.random.uniform(-4.0, 12.0, n_chips)
        chip_z = z_cut_bot + np.random.uniform(0.5, 8.0, n_chips)
        chip_sizes = np.random.uniform(15, 65, n_chips)
        chip_alphas = np.random.uniform(0.5, 0.95, n_chips)
        sc_chips = ax_mill.scatter(
            chip_x, chip_z, s=chip_sizes, c='#fef08a',
            edgecolors='#f59e0b', linewidth=0.8, alpha=0.8, zorder=9
        )
        ax_mill.draw_artist(sc_chips)
        dynamic_artists.append(sc_chips)

        t_shear = ax_mill.text(
            tool_x + 5.0, z_cut_bot + 4.5, "Dynamic Chip\nShear Zone",
            color='#fef08a', fontsize=7.5, fontweight='bold', zorder=10
        )
        ax_mill.draw_artist(t_shear)
        dynamic_artists.append(t_shear)

    # 4. Kinematics Annotations
    feed_arrow_len = 16.0
    ann_feed = ax_mill.annotate(
        '', xy=(tool_x + feed_arrow_len, z_cut_bot + cutter_h/2),
        xytext=(tool_x, z_cut_bot + cutter_h/2),
        arrowprops=dict(arrowstyle="-|>", color='#38bdf8', lw=2.2)
    )
    ax_mill.draw_artist(ann_feed)
    dynamic_artists.append(ann_feed)

    t_feed = ax_mill.text(
        tool_x + feed_arrow_len/2, z_cut_bot + cutter_h/2 + 1.2,
        f"vf = {n_rpm * f_r:.0f} mm/min", color='#38bdf8', fontsize=8,
        ha='center', va='bottom', fontweight='bold'
    )
    ax_mill.draw_artist(t_feed)
    dynamic_artists.append(t_feed)

    ann_spin = ax_mill.annotate(
        '', xy=(tool_x + cutter_r * 0.8, z_cut_bot + cutter_h + 1.5),
        xytext=(tool_x - cutter_r * 0.8, z_cut_bot + cutter_h + 1.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.4", color='#fcd34d', lw=2.0)
    )
    ax_mill.draw_artist(ann_spin)
    dynamic_artists.append(ann_spin)

    t_spin = ax_mill.text(
        tool_x, z_cut_bot + cutter_h + 3.2,
        f"n = {n_rpm:.0f} RPM", color='#fcd34d', fontsize=8,
        ha='center', va='bottom', fontweight='bold'
    )
    ax_mill.draw_artist(t_spin)
    dynamic_artists.append(t_spin)

    # 5. Top Title Banner
    t_title = fig.text(
        0.5, 0.98,
        f"AUTONOMOUS CNC FACE MILLING DIGITAL TWIN • DRL (PPO) REAL-TIME CONTROL  "
        f"[Frame {frame_idx+1}/{total_frames}]",
        fontsize=13, fontweight='bold', color='#f8fafc', ha='center', va='top'
    )
    fig.draw_artist(t_title)
    dynamic_artists.append(t_title)

    # Blit canvas
    fig.canvas.blit(fig.bbox)
    rgba = np.asarray(fig.canvas.buffer_rgba())

    # Clean up dynamic artists
    for a in dynamic_artists:
        a.remove()

    return rgba


def generate_animation(
    kappa_total: float = 5.0,
    VB_0: float = 0.02,
    D: float = 50.0,
    L_cut: float = 100.0,
    w2: float = 0.4,
    output_mp4: Path = None,
    output_gif: Path = None,
    fps: int = 30
):
    """Executes PPO simulation and renders full animated video."""
    print("=" * 70)
    print("Generating CNC Milling PPO Animation Video...")
    print("=" * 70)

    # 1. Load Model and Run Simulation
    print("Loading PPO policy and running physical simulation...")
    model_tuple = load_ppo_policy(PRIMARY_MODEL_DIR)
    df_passes = run_single_simulation(
        policy_type='PPO',
        kappa_total=kappa_total,
        VB_0=VB_0,
        D=D,
        L_cut=L_cut,
        w2=w2,
        ppo_model_tuple=model_tuple
    )

    print(f"Simulation completed: {len(df_passes)} passes executed.")
    for idx, row in df_passes.iterrows():
        print(f"  Pass {int(row['pass'])}: ap={row['a_p_mm']:.2f}mm, n={row['n_rpm']:.0f}RPM, "
              f"fr={row['f_r_mm_rev']:.3f}mm/rev, Ra={row['Ra_um']:.2f}um, VB={row['VB_mm']:.3f}mm")

    # 2. Plan trajectory frames
    # Cut frames per pass: ~75 frames (~2.5 seconds)
    # Rapid retract frames per pass: ~15 frames (~0.5 seconds)
    # Final hold: ~30 frames (1.0 second)
    frames_cut = 75
    frames_rapid = 15
    total_passes = len(df_passes)

    total_frames = total_passes * (frames_cut + frames_rapid) + 30
    print(f"Total animation frames to render: {total_frames} (~{total_frames/fps:.1f} s @ {fps} fps)")

    # 3. Setup Figure and Axes
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor('#0c131f')

    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[2.2, 1.0],
        height_ratios=[1.0, 0.22],
        left=0.04, right=0.96, top=0.92, bottom=0.06,
        wspace=0.12, hspace=0.25
    )

    ax_mill = fig.add_subplot(gs[0, 0])
    ax_hud = fig.add_subplot(gs[0, 1])
    ax_strip = fig.add_subplot(gs[1, :])
    ax_strip_right = ax_strip.twinx()

    # 4. Prepare Video Pipeline (Direct FFmpeg pipe with fallback to VideoWriter)
    if output_mp4 is None:
        output_mp4 = Path("assets/ppo_milling_animation.mp4")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    use_ffmpeg_pipe = shutil.which("ffmpeg") is not None
    ffmpeg_proc = None
    video_writer = None
    temp_avi = None

    if use_ffmpeg_pipe:
        print("Using direct in-memory FFmpeg stdin pipe (zero disk intermediate)...")
        cmd_mp4 = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", "1600x900",
            "-pix_fmt", "rgba",
            "-r", str(fps),
            "-i", "-",
            "-vcodec", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_mp4)
        ]
        ffmpeg_proc = subprocess.Popen(cmd_mp4, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    else:
        import tempfile
        temp_avi = Path(tempfile.gettempdir()) / f"temp_milling_raw_{os.getpid()}.avi"
        print(f"FFmpeg CLI not detected in PATH; falling back to OpenCV VideoWriter: {temp_avi}")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(str(temp_avi), fourcc, float(fps), (1600, 900))

    def write_frame(rgba_frame):
        if ffmpeg_proc is not None:
            ffmpeg_proc.stdin.write(rgba_frame.tobytes())
        elif video_writer is not None:
            bgr = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGR)
            video_writer.write(bgr)

    frame_counter = 0
    rotation_angle = 0.0
    cum_ap = list(np.cumsum(df_passes['a_p_mm'].tolist()))

    for p_i in range(total_passes):
        cur_z_bot = max(0.0, kappa_total - cum_ap[p_i])
        n_rpm = float(df_passes.iloc[p_i]['n_rpm'])
        deg_per_frame = (n_rpm / 60.0) * (360.0 / fps) * 0.2  # visual rotation rate

        # Cut phase: X advances from -5 to L_cut + 5
        bg_cut = render_pass_background(
            fig, ax_mill, ax_hud, ax_strip, ax_strip_right,
            df_passes=df_passes,
            current_pass_idx=p_i,
            is_rapid=False,
            L_cut=L_cut,
            kappa_total=kappa_total,
            D_tool=D
        )

        x_start = -5.0
        x_end = L_cut + 5.0
        for f in range(frames_cut):
            prog = f / float(frames_cut - 1)
            tx = x_start + prog * (x_end - x_start)
            tz = cur_z_bot
            rotation_angle = (rotation_angle + deg_per_frame) % 360.0

            rgba = render_dynamic_frame(
                fig, ax_mill, bg_cut,
                df_passes=df_passes,
                current_pass_idx=p_i,
                tool_x=tx,
                tool_z_bottom=tz,
                is_rapid=False,
                rotation_deg=rotation_angle,
                frame_idx=frame_counter,
                total_frames=total_frames,
                L_cut=L_cut,
                kappa_total=kappa_total,
                D_tool=D
            )
            write_frame(rgba)
            frame_counter += 1

        # Rapid traverse phase: retract tool up by 4 mm, rapid back to X = -5
        if p_i < total_passes - 1:
            bg_rapid = render_pass_background(
                fig, ax_mill, ax_hud, ax_strip, ax_strip_right,
                df_passes=df_passes,
                current_pass_idx=p_i,
                is_rapid=True,
                L_cut=L_cut,
                kappa_total=kappa_total,
                D_tool=D
            )

            z_retract = cur_z_bot + 4.0
            for f in range(frames_rapid):
                prog = f / float(frames_rapid - 1)
                tx = x_end - prog * (x_end - x_start)
                tz = z_retract
                rotation_angle = (rotation_angle + deg_per_frame) % 360.0

                rgba = render_dynamic_frame(
                    fig, ax_mill, bg_rapid,
                    df_passes=df_passes,
                    current_pass_idx=p_i,
                    tool_x=tx,
                    tool_z_bottom=tz,
                    is_rapid=True,
                    rotation_deg=rotation_angle,
                    frame_idx=frame_counter,
                    total_frames=total_frames,
                    L_cut=L_cut,
                    kappa_total=kappa_total,
                    D_tool=D
                )
                write_frame(rgba)
                frame_counter += 1

    # Final hold phase: tool retracted at end, finished state held for 30 frames
    bg_final = render_pass_background(
        fig, ax_mill, ax_hud, ax_strip, ax_strip_right,
        df_passes=df_passes,
        current_pass_idx=total_passes - 1,
        is_rapid=True,
        L_cut=L_cut,
        kappa_total=kappa_total,
        D_tool=D
    )
    for f in range(30):
        rgba = render_dynamic_frame(
            fig, ax_mill, bg_final,
            df_passes=df_passes,
            current_pass_idx=total_passes - 1,
            tool_x=L_cut + 15.0,
            tool_z_bottom=cur_z_bot + 4.0,
            is_rapid=True,
            rotation_deg=rotation_angle,
            frame_idx=frame_counter,
            total_frames=total_frames,
            L_cut=L_cut,
            kappa_total=kappa_total,
            D_tool=D
        )
        write_frame(rgba)
        frame_counter += 1

    if ffmpeg_proc is not None:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    elif video_writer is not None:
        video_writer.release()
        cmd_mp4 = [
            "ffmpeg", "-y", "-i", str(temp_avi),
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", str(output_mp4)
        ]
        subprocess.run(cmd_mp4, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if temp_avi.exists():
            temp_avi.unlink()

    plt.close(fig)
    print(f"Successfully generated MP4: {output_mp4} ({output_mp4.stat().st_size / 1e6:.2f} MB)")

    # Copy to report/assets for presentation/report
    report_mp4 = Path("report/assets/ppo_milling_animation.mp4")
    report_mp4.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_mp4, report_mp4)
    print(f"Copied to: {report_mp4}")

    # 6. Optional GIF export
    if output_gif is None:
        output_gif = Path("assets/ppo_milling_animation.gif")
    print(f"Generating optimized GIF preview: {output_gif}")
    cmd_gif = [
        "ffmpeg", "-y", "-i", str(output_mp4),
        "-vf", "fps=15,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        str(output_gif)
    ]
    subprocess.run(cmd_gif, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Successfully generated GIF: {output_gif} ({output_gif.stat().st_size / 1e6:.2f} MB)")

    print("=" * 70)
    print("Animation Generation Complete!")
    print("=" * 70)

if __name__ == '__main__':
    generate_animation()
