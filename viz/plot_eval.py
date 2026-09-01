import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def plot_mc_results():
    os.makedirs('plots', exist_ok=True)
    files = glob.glob('logs/monte_carlo/*.csv')

    if not files:
        print("No Monte Carlo CSVs found in logs/monte_carlo/")
        return

    data = []
    for f in files:
        df = pd.read_csv(f)
        algo = "PPO" if "PPO" in f else "SAC"
        df['algo'] = algo
        data.append(df)

    df_all = pd.concat(data)

    sns.set_theme(style="whitegrid")

    # Group by config and algo to get the max constraints per episode
    max_constraints = df_all.groupby(['algo', 'config'])[['Ra_um', 'VB_mm']].max().reset_index()

    # 1. KDE Distribution of Constraints over 100 Random Scenarios
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.kdeplot(data=max_constraints, x="Ra_um", hue="algo", fill=True, ax=ax1, palette="Set1", common_norm=False, alpha=0.5)
    ax1.axvline(x=1.6, color='red', linestyle='--', linewidth=2, label='Ra Limit (1.6 um)')
    ax1.set_xlabel('Max Surface Roughness (um) per Scenario')
    ax1.set_title('Robustness: Surface Roughness over 100 Random Scenarios')
    ax1.legend()
    
    sns.kdeplot(data=max_constraints, x="VB_mm", hue="algo", fill=True, ax=ax2, palette="Set1", common_norm=False, alpha=0.5)
    ax2.axvline(x=0.3, color='red', linestyle='--', linewidth=2, label='Wear Limit (0.3 mm)')
    ax2.set_xlabel('Max Tool Wear (mm) per Scenario')
    ax2.set_title('Robustness: Tool Wear over 100 Random Scenarios')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('plots/monte_carlo_constraints_kde.png', dpi=150)
    plt.close()
    
    # 2. Scatter plot of Ra vs VB to show Pareto front of failures
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=max_constraints, x="VB_mm", y="Ra_um", hue="algo", palette="Set1", alpha=0.7)
    plt.axvline(x=0.3, color='red', linestyle='--', label='Wear Limit (0.3 mm)')
    plt.axhline(y=1.6, color='red', linestyle='-', label='Ra Limit (1.6 um)')
    plt.title('Failure Landscape across 100 Random Machining Scenarios')
    plt.xlabel('Max Tool Wear (mm)')
    plt.ylabel('Max Surface Roughness (um)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/monte_carlo_failure_landscape.png', dpi=150)
    plt.close()

    print("Saved Monte Carlo statistical plots to plots/")

if __name__ == "__main__":
    plot_mc_results()
