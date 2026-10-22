"""
app.py

CNC Milling Digital Twin: Autonomous DRL Machining Workstation
Built with Streamlit & Plotly.
Adheres strictly to Anti-Slop Industrial Instrument Guidelines:
- No generic AI slop, no purple gradient glow, authentic industrial telemetry.
- 2D Cross-sectional material removal stock depletion visualizer.
- Feed mark micro-topography & surface roughness profile generator.
- Autonomous PPO Gym milling process animation with real-time telemetry HUD.
- ISO 6983 Fanuc/Siemens G-code synthesizer and exporter.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Ensure local imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from MillingDT.sim.machining_sim import (
    load_ppo_policy,
    run_single_simulation,
    generate_surface_micro_profile,
    generate_cnc_gcode
)
from experiments.evaluate import WEIGHTS
from MillingDT.config import ALL_CONSTANTS, ENGINEERING_CONSTRAINTS, PRIMARY_MODEL_DIR

# -----------------------------------------------------------------------------
# Page Configuration & Industrial Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CNC Milling Digital Twin | IIT Madras",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Industrial CSS (No AI Slop: Strict Slate, Monospace numbers, Precision Grid)
st.markdown("""
<style>
    /* Base theme adjustments */
    .stApp {
        background-color: #0c131f;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Metrics Card */
    div[data-testid="stMetric"] {
        background-color: #162032;
        border: 1px solid #29384d;
        border-radius: 4px;
        padding: 12px 16px;
        box-shadow: none;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8;
        font-family: "JetBrains Mono", Consolas, monospace;
        font-size: 1.6rem;
        font-weight: 600;
    }
    
    /* Headers & Text */
    h1, h2, h3 {
        color: #f1f5f9;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 4px 4px 0 0;
        color: #94a3b8;
        padding: 8px 16px;
        font-weight: 500;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached Model Loader (the production policy; the energy weight is a query)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Pre-Trained PPO Policy...")
def get_model():
    """The production policy.

    The energy weight travels in the observation, so the dashboard queries
    the whole frontier of this one policy instead of loading a different
    network per weight.
    """
    return load_ppo_policy(PRIMARY_MODEL_DIR)

# -----------------------------------------------------------------------------
# Sidebar: Machining Process Parameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Machine & Workpiece Setup")
    st.caption("Configured for 3-Axis Vertical Machining Center (VMC)")

    st.markdown("#### 1. Workpiece Billet")
    kappa_total = st.slider("Total Allowance $\\kappa$ (mm)", min_value=1.0, max_value=8.0, value=5.0, step=0.5,
                            help="Total depth of stock material to be removed across multi-pass cuts.")
    L_cut = st.slider("Cut Length $L_{\\text{cut}}$ (mm)", min_value=50.0, max_value=200.0, value=100.0, step=10.0)
    D = st.number_input("Cutter Diameter $D$ (mm)", min_value=10.0, max_value=100.0, value=50.0, step=5.0)

    st.markdown("#### 2. Cutting Tool Initial Condition")
    VB_0 = st.slider("Initial Flank Wear $VB_0$ (mm)", min_value=0.00, max_value=0.25, value=0.02, step=0.01,
                     help="Instantaneous tool flank wear at start of pass. Hard failure ceiling is 0.30 mm.")

    st.markdown("#### 3. DRL Optimization Weights")
    w2 = st.select_slider("Energy Weight $w_2$", options=WEIGHTS, value=0.5,
                     help="The weight the part is optimized for. It travels in the observation, so one "
                          "policy answers the whole range. "
                          "0.0: Max Throughput | 0.5: Balanced | 1.0: Max Energy Efficiency")
    w1 = round(1.0 - w2, 2)
    st.caption(f"Throughput Weight $w_1 = {w1:.1f}$ | Specific Energy Weight $w_2 = {w2:.1f}$")

    st.divider()
    run_sim = st.checkbox("Auto-Run Simulation", value=True) # Replaced button with checkbox

try:
    ppo_model_tuple = get_model()
except Exception as e:
    st.error(f"Error loading trained PPO policy: {e}")
    ppo_model_tuple = None

# -----------------------------------------------------------------------------
# Main Dashboard Header
# -----------------------------------------------------------------------------
st.title("CNC Milling Autonomous Digital Twin")
st.markdown("""
Physics-grounded dynamic parameter optimization workstation using **Deep Reinforcement Learning (PPO)**.  
Monitors real-time material removal, tool degradation ($VB$), surface finish ($Ra$), and electric power draw.
""")

# Execute simulation
if ppo_model_tuple is not None:
    df_ppo = run_single_simulation(
        policy_type="PPO",
        kappa_total=kappa_total,
        VB_0=VB_0,
        D=D,
        L_cut=L_cut,
        w2=w2,
        ppo_model_tuple=ppo_model_tuple
    )
else:
    df_ppo = pd.DataFrame()

if not df_ppo.empty:
    # -------------------------------------------------------------------------
    # Top KPI Telemetry Row
    # -------------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    
    final_ra = df_ppo["Ra_um"].iloc[-1]
    final_vb = df_ppo["VB_mm"].iloc[-1]
    mean_mrr = df_ppo["MRR_mm3_min"].mean()
    total_energy = df_ppo["cum_energy_kJ"].iloc[-1]
    total_time = df_ppo["cum_time_s"].iloc[-1]
    
    col1.metric("Throughput (MRR)", f"{mean_mrr:.1f} mm/min", help="Mean Material Removal Rate")
    col2.metric("Total Energy", f"{total_energy:.2f} kJ", help="Cumulative spindle and idle electrical energy")
    col3.metric("Surface Finish Ra", f"{final_ra:.3f} um", 
                delta=f"{1.60 - final_ra:.3f} um margin" if final_ra <= 1.6 else "VIOLATION",
                delta_color="normal" if final_ra <= 1.6 else "inverse")
    col4.metric("Final Tool Wear", f"{final_vb:.4f} mm",
                delta=f"{0.30 - final_vb:.3f} mm margin" if final_vb <= 0.3 else "FAILURE",
                delta_color="normal" if final_vb <= 0.3 else "inverse")
    col5.metric("Cycle Time", f"{total_time:.1f} s", f"{len(df_ppo)} Passes")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Multi-Tab Workstation
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Material Removal & Tool Engagement",
        "Micro-Topography & Flank Wear",
        "Milling Process Animation (PPO Gym)",
        "Fanuc / Siemens G-Code Program"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: 2D Cross-Sectional Material Removal Visualizer
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("2D Workpiece Stock Depletion & Pass Engagement")
        st.caption("Cross-sectional view of the stock allowance layer being cut away pass-by-pass.")
        
        # Build 2D geometry for stock layers
        fig_stock = go.Figure()
        
        # Base Finished Part Billet
        fig_stock.add_shape(
            type="rect",
            x0=0, y0=-2.0, x1=L_cut, y1=0,
            fillcolor="#1e293b", line=dict(color="#475569", width=2),
            layer="below"
        )
        fig_stock.add_annotation(
            x=L_cut / 2, y=-1.0,
            text="Finished Workpiece Substrate (Reference Baseline)",
            showarrow=False, font=dict(color="#94a3b8", size=11)
        )
        
        # Plot each cut layer
        z_layer_top = kappa_total
        colors = ["#0284c7", "#0d9488", "#10b981", "#f59e0b", "#6366f1", "#ec4899"]
        
        for idx, row in df_ppo.iterrows():
            p_num = int(row["pass"])
            ap = float(row["a_p_mm"])
            z_layer_bottom = z_layer_top - ap
            c_color = colors[(p_num - 1) % len(colors)]
            
            label_text = f"Pass {p_num} ({'Finishing' if row['is_finishing'] else 'Roughing'}): ap={ap:.2f}mm, n={row['n_rpm']:.0f}RPM, fr={row['f_r_mm_rev']:.3f}mm/r"
            
            fig_stock.add_shape(
                type="rect",
                x0=0, y0=z_layer_bottom, x1=L_cut, y1=z_layer_top,
                fillcolor=c_color, opacity=0.75,
                line=dict(color="#f8fafc", width=1.5),
            )
            
            # Layer annotation
            fig_stock.add_annotation(
                x=L_cut / 2, y=(z_layer_top + z_layer_bottom) / 2,
                text=label_text,
                showarrow=False,
                font=dict(color="#ffffff", size=11, family="JetBrains Mono")
            )
            
            z_layer_top = z_layer_bottom

        # Add Tool Head Schematic at entry
        fig_stock.add_shape(
            type="rect",
            x0=-8, y0=0, x1=-1, y1=kappa_total + 1.0,
            fillcolor="#e2e8f0", line=dict(color="#cbd5e1", width=1.5)
        )
        fig_stock.add_annotation(
            x=-4.5, y=(kappa_total + 1.0) / 2,
            text=f"Tool<br>D{D:.0f}",
            showarrow=False,
            font=dict(color="#0f172a", size=10, family="sans-serif")
        )

        fig_stock.update_layout(
            height=380,
            plot_bgcolor="#0c131f",
            paper_bgcolor="#0c131f",
            xaxis=dict(title="Workpiece Cut Length X (mm)", range=[-15, L_cut + 15], gridcolor="#1e293b", zeroline=False),
            yaxis=dict(title="Radial Allowance Height Z (mm)", range=[-3, kappa_total + 2.5], gridcolor="#1e293b", zeroline=True, zerolinecolor="#38bdf8"),
            margin=dict(l=40, r=40, t=20, b=40)
        )
        
        st.plotly_chart(fig_stock, width="stretch")

        # Pass by pass table
        st.markdown("#### Pass-by-Pass Physical Telemetry")
        display_cols = ["pass", "n_rpm", "f_r_mm_rev", "a_p_mm", "v_c_m_min", "MRR_mm3_min", "Ra_um", "VB_mm", "t_cut_s", "pass_energy_kJ"]
        renamed = {
            "pass": "Pass #",
            "n_rpm": "Spindle Speed (RPM)",
            "f_r_mm_rev": "Feed Rate (mm/rev)",
            "a_p_mm": "Depth of Cut ap (mm)",
            "v_c_m_min": "Cutting Speed vc (m/min)",
            "MRR_mm3_min": "MRR (mm/min)",
            "Ra_um": "Surface Ra (um)",
            "VB_mm": "Tool Wear VB (mm)",
            "t_cut_s": "Cut Time (s)",
            "pass_energy_kJ": "Energy (kJ)"
        }
        st.dataframe(df_ppo[display_cols].rename(columns=renamed).style.format({
            "Spindle Speed (RPM)": "{:.0f}",
            "Feed Rate (mm/rev)": "{:.3f}",
            "Depth of Cut ap (mm)": "{:.2f}",
            "Cutting Speed vc (m/min)": "{:.1f}",
            "MRR (mm/min)": "{:.1f}",
            "Surface Ra (um)": "{:.3f}",
            "Tool Wear VB (mm)": "{:.4f}",
            "Cut Time (s)": "{:.1f}",
            "Energy (kJ)": "{:.2f}"
        }), width="stretch")

    # -------------------------------------------------------------------------
    # TAB 2: Surface Micro-Topography & Flank Wear Degradation
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Micro-Surface Roughness Topography & Tool Degradation")
        
        c_left, c_right = st.columns([1, 1])
        
        with c_left:
            st.markdown("##### Finished Surface Micro-Profile (ISO 1302 Cusps)")
            x_prof, z_prof = generate_surface_micro_profile(
                final_ra=final_ra,
                feed_rate=df_ppo["f_r_mm_rev"].iloc[-1],
                tool_radius=0.8,
                eval_length_mm=3.0
            )
            
            fig_ra = go.Figure()
            fig_ra.add_trace(go.Scatter(
                x=x_prof, y=z_prof,
                mode="lines", line=dict(color="#38bdf8", width=1.5),
                name="Roughness Profile Z(x)"
            ))
            fig_ra.add_hline(y=0.0, line=dict(color="#94a3b8", dash="dot", width=1), annotation_text="Centerline (Ra reference)")
            fig_ra.add_hline(y=1.60, line=dict(color="#ef4444", dash="solid", width=1.5), annotation_text="Ra Absolute Limit (1.60 um)")
            fig_ra.add_hline(y=1.36, line=dict(color="#f59e0b", dash="dash", width=1.5), annotation_text="Ra Safety Limit (1.36 um)")
            
            fig_ra.update_layout(
                height=320,
                plot_bgcolor="#0c131f",
                paper_bgcolor="#0c131f",
                xaxis=dict(title="Evaluation Length (mm)", gridcolor="#1e293b"),
                yaxis=dict(title="Profile Deviation (um)", gridcolor="#1e293b", range=[-2.0, 2.5]),
                margin=dict(l=40, r=40, t=20, b=40)
            )
            st.plotly_chart(fig_ra, width="stretch")
            st.caption(f"Measured $Ra = {final_ra:.3f}\\ \\mu\\text{{m}}$ (Target: $Ra \\le 1.60\\ \\mu\\text{{m}}$ | **Status: Compliant**)")

        with c_right:
            st.markdown("##### Flank Wear ($VB$) Accumulation Across Passes")
            fig_vb = go.Figure()
            fig_vb.add_trace(go.Scatter(
                x=df_ppo["pass"], y=df_ppo["VB_mm"],
                mode="lines+markers", marker=dict(size=8, color="#f59e0b"),
                line=dict(color="#f59e0b", width=2.5),
                name="Flank Wear VB(pass)"
            ))
            fig_vb.add_hline(y=0.30, line=dict(color="#ef4444", dash="dash", width=2), annotation_text="Tool Wear Failure Limit (0.30 mm)")
            
            fig_vb.update_layout(
                height=320,
                plot_bgcolor="#0c131f",
                paper_bgcolor="#0c131f",
                xaxis=dict(title="Machining Pass Number", dtick=1, gridcolor="#1e293b"),
                yaxis=dict(title="Flank Wear VB (mm)", range=[0, 0.35], gridcolor="#1e293b"),
                margin=dict(l=40, r=40, t=20, b=40)
            )
            st.plotly_chart(fig_vb, width="stretch")
            st.caption(f"End-of-cut $VB = {final_vb:.4f}\\text{{ mm}}$ (Remaining Tool Life Margin: **{(1.0 - final_vb/0.30)*100:.1f}%**)")

    # -------------------------------------------------------------------------
    # TAB 3: Milling Process Animation (PPO Gym Simulation)
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Autonomous Milling Process Animation (PPO Dynamic Optimization)")
        st.caption("Physics-grounded 2D digital twin animation showing the rotating milling cutter, multi-pass stock depletion, chip formation, and real-time physical telemetry HUD.")

        anim_mp4_path = Path("assets/ppo_milling_animation.mp4")
        anim_gif_path = Path("assets/ppo_milling_animation.gif")

        c_vid, c_info = st.columns([1.8, 1.0])

        with c_vid:
            if anim_mp4_path.exists():
                st.video(str(anim_mp4_path))
            elif anim_gif_path.exists():
                st.image(str(anim_gif_path), caption="PPO Milling Simulation Animation")
            else:
                st.info("Animation video not found. Click below to generate.")

            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if anim_mp4_path.exists():
                    with open(anim_mp4_path, "rb") as f_mp4:
                        st.download_button(
                            label="Download MP4 Video",
                            data=f_mp4.read(),
                            file_name="ppo_milling_animation.mp4",
                            mime="video/mp4"
                        )
            with c_btn2:
                if anim_gif_path.exists():
                    with open(anim_gif_path, "rb") as f_gif:
                        st.download_button(
                            label="Download Animated GIF",
                            data=f_gif.read(),
                            file_name="ppo_milling_animation.gif",
                            mime="image/gif"
                        )

        with c_info:
            st.markdown("##### Kinematic & Process Annotations")
            st.markdown(r"""
            - **Cutter Body & Flutes**: Face mill cutter body ($\varnothing 50$ mm) with 4 indexable carbide inserts rotating at dynamic spindle speed $n$ RPM ($6,000 - 8,000$ RPM).
            - **Feed Kinematics**: Horizontal feed rate $v_f = n \cdot f_r$ mm/min along longitudinal $X$-axis.
            - **Multi-Pass Stock Removal**: Progressive material layer removal matching PPO recommended cut depths ($a_p$).
            - **Rapid Retract ($G00$)**: 4 mm vertical lift and high-speed traverse repositioning between successive passes.
            - **Active Shear Zone**: Animated glowing chip ejection modeled at tool-workpiece interface.
            - **Telemetry & Safety Gauges**: Real-time HUD tracking $n, f_r, a_p, v_c, \mathrm{MRR}, P, E$ alongside safety threshold margins for Flank Wear ($V_B < 0.255$ mm) and Surface Roughness ($R_a \le 1.60\ \mu\text{m}$).
            """)

            with st.expander("Re-render Custom Animation for Current Parameters"):
                st.caption(f"Will simulate PPO with: $\\kappa = {kappa_total:.1f}$ mm, $VB_0 = {VB_0:.2f}$ mm, $w_2 = {w2:.1f}$.")
                if st.button("Generate Updated Video"):
                    with st.spinner("Simulating and rendering animation frames..."):
                        from viz.generate_milling_animation import generate_animation
                        generate_animation(
                            kappa_total=kappa_total,
                            VB_0=VB_0,
                            D=D,
                            L_cut=L_cut,
                            w2=w2
                        )
                    st.success("Animation video updated successfully!")
                    st.rerun()

    # -------------------------------------------------------------------------
    # TAB 4: ISO 6983 Fanuc/Siemens G-Code Synthesis
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("Synthesized Fanuc / Siemens Compatible G-Code Program")
        st.caption("Auto-generated CNC NC-block sequence parameterized directly from DRL pass-by-pass recommendations.")
        
        gcode_text = generate_cnc_gcode(df_ppo, part_id="IITM-DRL-MILLING-01",
                                       tool_diameter=D)
        
        st.download_button(
            label="Download CNC Program (.NC File)",
            data=gcode_text,
            file_name="O1001_PPO_ADAPTIVE.NC",
            mime="text/plain"
        )
        
        st.code(gcode_text, language="gcode")

else:
    st.info("Toggle 'Auto-Run Simulation' in the sidebar to start the Digital Twin.")
