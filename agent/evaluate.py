import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import env.milling_env

# Register v2 env
gym.register(
    id="MillingDTEnv-v1",
    entry_point="env.milling_env:MillingMultipassEnv",
    max_episode_steps=50,
)

def generate_random_configs(n=100):
    configs = {}
    np.random.seed(42) # For reproducibility of the sweep
    for i in range(n):
        configs[f"Random_{i:03d}"] = {
            "kappa_total": np.random.uniform(2.0, 15.0),
            "D": np.random.uniform(40.0, 150.0),
            "L_cut": np.random.uniform(50.0, 200.0),
            "C_wear": np.random.uniform(1.0e-2, 6.0e-2),
            "k1": np.random.uniform(10.0, 15.0),
            "k2": np.random.uniform(0.5, 1.2),
        }
    return configs

def evaluate(episodes=1):
    logs_dir = PROJECT_ROOT / "logs" / "monte_carlo"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  MASSIVE MONTE CARLO INFERENCE: RANDOM FEATURE SWEEP")
    print("=" * 70)
    
    algos = ["PPO", "SAC"]
    configs = generate_random_configs(100) # 100 random machining scenarios
    
    for algo in algos:
        model_dir = PROJECT_ROOT / "models" / f"{algo}"
        model_path = model_dir / "final_model.zip"
        vec_norm_path = model_dir / "vecnormalize.pkl"
        
        if not model_path.exists():
            continue
            
        ModelClass = PPO if algo == "PPO" else SAC
        model = ModelClass.load(str(model_path))
        
        all_steps = []
        
        print(f"Running sweep for {algo}...")
        for conf_name, env_kwargs in tqdm(configs.items()):
            eval_env = make_vec_env("MillingDTEnv-v1", env_kwargs=env_kwargs, n_envs=1)
            eval_env = VecNormalize.load(str(vec_norm_path), eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            
            for ep in range(episodes):
                obs = eval_env.reset()
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, done, info = eval_env.step(action)
                    info = info[0]
                    
                    all_steps.append({
                        'config': conf_name,
                        'episode': ep,
                        'pass': info['pass'],
                        'a_p_mm': info['a_p'],
                        'f_r_mm_r': info['f_r'],
                        'n_rpm': info['n'],
                        'Ra_um': info['Ra'],
                        'VB_mm': info['VB'],
                        'kappa_rem_mm': info['kappa_rem'],
                        'is_finishing': info['is_finishing'],
                        'broken': info['broken']
                    })
                    
        df_steps = pd.DataFrame(all_steps)
        csv_path = logs_dir / f"eval_{algo}_MC.csv"
        df_steps.to_csv(csv_path, index=False)

if __name__ == "__main__":
    evaluate()
