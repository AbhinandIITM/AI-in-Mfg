"""
MillingDT/agent/controller_init.py

Initialization of the policy from the receding-horizon controller.

The controller in MillingDT/baselines/receding_horizon.py is a competent
expert: it re-solves the next pass with the current state and the current
energy weight, and it stays inside the roughness and wear limits. This module
records what it does, fits a supervised actor with the same shape as the PPO
actor on those trajectories, and copies the fitted weights into a PPO policy,
so that the reinforcement learning starts from a behaviour that is already
feasible rather than from noise.

The environment is the one source of truth for the physics, so the controller
is queried with the coefficients of the scenario in play, not with the
nominal ones.
"""

import numpy as np
import torch
import torch.nn as nn

from MillingDT.baselines.receding_horizon import solve_next_pass
from MillingDT.config import ALL_CONSTANTS

MACHINE = ALL_CONSTANTS['machine']


def normalized_action(n, f_r, a_p):
    """Physical parameters to the normalized action of the environment."""
    return np.array([
        2.0 * (n - MACHINE['n_min']) / (MACHINE['n_max'] - MACHINE['n_min']) - 1.0,
        2.0 * (f_r - MACHINE['f_r_min']) / (MACHINE['f_r_max'] - MACHINE['f_r_min']) - 1.0,
        2.0 * (a_p - MACHINE['a_p_min']) / (MACHINE['a_p_max'] - MACHINE['a_p_min']) - 1.0,
    ], dtype=np.float32)


def current_scenario(unwrapped):
    """The coefficients of the scenario in play, for the controller to solve."""
    return {
        'C_wear': unwrapped.tool_wear_cfg['C_wear'],
        'k1': unwrapped.roughness_cfg['k1'],
        'k2': unwrapped.roughness_cfg['k2'],
        'k3': unwrapped.roughness_cfg['k3'],
    }


def controller_action(unwrapped):
    """The next pass the controller chooses for the current state."""
    n, f_r, a_p = solve_next_pass(
        VB=unwrapped.VB,
        kappa_rem=unwrapped.kappa,
        D=unwrapped.D,
        L_cut=unwrapped.L_cut,
        w1=unwrapped.reward_cfg['w1'],
        w2=unwrapped.reward_cfg['w2'],
        MRR_max=unwrapped.MRR_max,
        E_ref=unwrapped.E_ref,
        machine_cfg=unwrapped.machine_cfg,
        scenario=current_scenario(unwrapped),
    )
    return normalized_action(n, f_r, a_p)


class ClonedActor(nn.Module):
    """The actor shape of the PPO policy: two hidden layers of 256 with Tanh."""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)


def collect_controller_dataset(episodes=2000, seed=42, randomize=True, randomize_weight=True):
    """Record what the controller does, and what it sees when it does it."""
    import gymnasium as gym

    env = gym.make("MillingDTEnv-v1", randomize=randomize,
                   randomize_weight=randomize_weight)
    unwrapped = env.unwrapped

    observations, actions = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        while not done:
            action = controller_action(unwrapped)
            observations.append(np.asarray(obs, dtype=np.float32))
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    env.close()

    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def fit_cloned_actor(X, Y, epochs=60, batch_size=256, lr=1e-3, seed=42):
    """Fit the supervised actor on the controller's trajectories."""
    torch.manual_seed(seed)
    actor = ClonedActor(X.shape[1], Y.shape[1])
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.as_tensor(X)
    Yt = torch.as_tensor(Y)
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        permutation = torch.randperm(len(Xt), generator=generator)
        total = 0.0
        for start in range(0, len(Xt), batch_size):
            index = permutation[start:start + batch_size]
            loss = loss_fn(actor(Xt[index]), Yt[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(index)
        if epoch % 15 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch:>3}: mse {total/len(Xt):.5f}")

    return actor


def copy_into_policy(model, actor):
    """Copy the fitted weights into the actor of a PPO policy."""
    policy = model.policy
    target = policy.mlp_extractor.policy_net
    source = actor.net

    pairs = [(target[0], source[0]), (target[2], source[2]), (policy.action_net, source[4])]
    for dst, src in pairs:
        if dst.weight.shape != src.weight.shape:
            raise ValueError(f"Shape mismatch: {dst.weight.shape} against {src.weight.shape}")
        with torch.no_grad():
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
    return model
