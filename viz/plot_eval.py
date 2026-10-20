import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from viz.plotting_utils import set_style, plot_kde_distribution, plot_failure_landscape
from MillingDT.config import ENGINEERING_CONSTRAINTS, PRIMARY_RUN, run_metrics, run_plots

def plot_mc_results():
    plots_dir = run_plots(PRIMARY_RUN)
    plots_dir.mkdir(parents=True, exist_ok=True)
    mc_path = run_metrics(PRIMARY_RUN) / "monte_carlo.csv"

    # The episode table of the primary run. The summary table beside it
    # holds other rows and is not plotted here.
    if not mc_path.exists():
        print(f"No Monte Carlo table found at {mc_path}")
        return

    df_all = pd.read_csv(mc_path)
    df_all['algo'] = "PPO"

    # Normalize column names between v3 (Ra_um, VB_mm) and v4 (Ra_max, VB_max)
    if 'Ra_max' in df_all.columns and 'Ra_um' not in df_all.columns:
        df_all['Ra_um'] = df_all['Ra_max']
    if 'VB_max' in df_all.columns and 'VB_mm' not in df_all.columns:
        df_all['VB_mm'] = df_all['VB_max']

    set_style()


    # Group by config and algo to get the max constraints per episode
    max_constraints = df_all.groupby(['algo', 'config'])[['Ra_um', 'VB_mm']].max().reset_index()

    ra_safe = ENGINEERING_CONSTRAINTS['Ra_max'] * ENGINEERING_CONSTRAINTS.get('safety_factor', 0.85)
    vb_safe = ENGINEERING_CONSTRAINTS['VB_max'] * ENGINEERING_CONSTRAINTS.get('safety_factor', 0.85)
    ra_limit = ENGINEERING_CONSTRAINTS['Ra_max']
    vb_limit = ENGINEERING_CONSTRAINTS['VB_max']

    # 1. KDE Distributions using plotting_utils
    # For Ra_um
    plot_kde_distribution(
        data=max_constraints['Ra_um'],
        title='Robustness: Surface Roughness over Random Scenarios',
        xlabel='Max Surface Roughness (um)',
        ylabel='Density',
        color='blue',
        limit=ra_limit,
        limit_label=f"Roughness Limit (Ra  {ra_limit:.1f} um)",
        safety_limit=ra_safe,
        safety_limit_label=f"Ra Safety Limit ({ra_safe:.1f} um)",
        filename=str(plots_dir / 'monte_carlo_constraints_kde_ra.png')
    )

    # For VB_mm
    plot_kde_distribution(
        data=max_constraints['VB_mm'],
        title='Robustness: Tool Wear over Random Scenarios',
        xlabel='Max Tool Wear (mm)',
        ylabel='Density',
        color='red',
        limit=vb_limit,
        limit_label=f"Wear Breakage Limit (VB  {vb_limit:.2f} mm)",
        safety_limit=vb_safe,
        safety_limit_label=f"VB Safety Limit ({vb_safe:.2f} mm)",
        filename=str(plots_dir / 'monte_carlo_constraints_kde_vb.png')
    )

    # 2. Scatter plot / Failure landscape using plotting_utils
    # Make sure 'broken' or success logic is handled
    if 'broken' not in max_constraints.columns:
        max_constraints['broken'] = max_constraints['VB_mm'] > vb_limit
    if 'Ra_violation' not in max_constraints.columns:
        max_constraints['Ra_violation'] = max_constraints['Ra_um'] > ra_limit

    plot_failure_landscape(
        df=max_constraints,
        x_col='VB_mm',
        y_col='Ra_um',
        title='Failure Landscape & Operating Envelope across Machining Scenarios',
        x_label='Max Tool Wear VB (mm)',
        y_label='Max Surface Roughness Ra (Im)',
        x_limit=vb_limit,
        y_limit=ra_limit,
        x_limit_label=f"Wear Limit (VB %  {vb_limit:.2f} mm)",
        y_limit_label=f"Roughness Limit (Ra %  {ra_limit:.1f} Im)",
        x_safe=vb_safe,
        y_safe=ra_safe,
        x_safe_label=f"VB Safety Limit ({vb_safe:.2f} mm)",
        y_safe_label=f"Ra Safety Limit ({ra_safe:.1f} um)",
        filename=str(plots_dir / 'monte_carlo_failure_landscape.png')
    )

    print(f"Saved Monte Carlo statistical plots with constraint limits to {plots_dir}/")



if __name__ == "__main__":
    plot_mc_results()
