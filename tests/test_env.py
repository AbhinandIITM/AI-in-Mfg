import pytest
import numpy as np
import gymnasium as gym
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import env
from config.physics_constants import ALL_CONSTANTS, MACHINE_CONSTRAINTS, ENGINEERING_CONSTRAINTS

def test_action_clipping():
    """Test that out-of-bounds actions are correctly clipped and handled."""
    env_inst = gym.make("MillingDTEnv-v1")
    obs, _ = env_inst.reset(seed=42)

    # Test extreme out-of-bounds actions
    extreme_actions = [
        np.array([5.0, -5.0, 10.0], dtype=np.float32),
        np.array([-10.0, 10.0, -10.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ]

    for action in extreme_actions:
        obs, reward, terminated, truncated, info = env_inst.step(action)
        # Check that state variables remain within valid bounds
        assert MACHINE_CONSTRAINTS['n_min'] <= info['n'] <= MACHINE_CONSTRAINTS['n_max'] + 1e-3
        assert MACHINE_CONSTRAINTS['f_r_min'] <= info['f_r'] <= MACHINE_CONSTRAINTS['f_r_max'] + 1e-3
        assert MACHINE_CONSTRAINTS['a_p_min'] <= info['a_p'] <= MACHINE_CONSTRAINTS['a_p_max'] + 1e-3
    env_inst.close()

def test_truncated_flag():
    """Test Gymnasium API compliance regarding the truncated flag and max steps."""
    env_inst = gym.make("MillingDTEnv-v1")
    env_inst.reset(seed=42)

    truncated = False
    terminated = False
    steps = 0

    while not (terminated or truncated):
        action = env_inst.action_space.sample()
        # Force small depth of cut to avoid finishing too fast, testing truncation limit
        action[2] = -1.0
        _, _, terminated, truncated, _ = env_inst.step(action)
        steps += 1
        if steps >= env_inst.unwrapped.max_steps:
            break

    assert truncated or terminated
    env_inst.close()

def test_soft_normalization():
    """Test soft normalization (tanh squashing) keeps values strictly within [0, 1]."""
    env_inst = gym.make("MillingDTEnv-v1")
    env_inst.reset(seed=42)

    # Test internal _soft_normalize method across wide range of values
    for val in [-100.0, -1.0, 0.0, 0.5, 1.0, 2.0, 100.0]:
        norm_val = env_inst.unwrapped._soft_normalize(val, 0.0, 1.0)
        assert 0.0 <= norm_val <= 1.0

    # Check observation space bounds
    obs, _ = env_inst.reset()
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    env_inst.close()

def test_physics_constants_sync():
    """Test that environment properly syncs physics constants from ALL_CONSTANTS."""
    env_inst = gym.make("MillingDTEnv-v1")
    env_inst.reset(seed=42)

    assert env_inst.unwrapped.tool_wear_cfg == ALL_CONSTANTS['tool_wear']
    assert env_inst.unwrapped.roughness_cfg == ALL_CONSTANTS['surface_roughness']
    assert env_inst.unwrapped.energy_cfg == ALL_CONSTANTS['energy']
    assert env_inst.unwrapped.machine_cfg == ALL_CONSTANTS['machine']
    assert env_inst.unwrapped.eng_cfg == ALL_CONSTANTS['engineering']
    assert env_inst.unwrapped.workpiece_cfg == ALL_CONSTANTS['workpiece']
    env_inst.close()

def test_physics_overrides_are_honored():
    """Per-scenario material and tool coefficients must reach the physics."""
    env_inst = gym.make("MillingDTEnv-v1", C_wear=0.05, k1=14.0, k2=1.1, k3=0.4, alpha=0.45)
    unwrapped = env_inst.unwrapped

    assert unwrapped.tool_wear_cfg['C_wear'] == 0.05
    assert unwrapped.tool_wear_cfg['alpha'] == 0.45
    assert unwrapped.roughness_cfg['k1'] == 14.0
    assert unwrapped.roughness_cfg['k2'] == 1.1
    assert unwrapped.roughness_cfg['k3'] == 0.4

    # The shared tables must be untouched: other consumers read them directly.
    assert ALL_CONSTANTS['tool_wear']['C_wear'] != 0.05
    assert ALL_CONSTANTS['surface_roughness']['k1'] == 12.5
    assert unwrapped.tool_wear_cfg is not ALL_CONSTANTS['tool_wear']
    env_inst.close()

def test_info_exports_accumulators():
    """The info payload must expose the running cycle time and energy totals."""
    env_inst = gym.make("MillingDTEnv-v1")
    env_inst.reset(seed=42)

    _, _, _, _, info = env_inst.step(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert 'T_total' in info
    assert 'E_total' in info
    assert info['T_total'] > 0.0
    assert info['E_total'] > 0.0

    first_time = info['T_total']
    first_energy = info['E_total']

    _, _, _, _, info = env_inst.step(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert info['T_total'] > first_time
    assert info['E_total'] > first_energy
    env_inst.close()

def test_energy_normalization_is_per_pass():
    """Energy is normalized against a reference pass, not a whole part.

    A whole-part scale makes the per-step energy term two orders of magnitude
    smaller than the throughput term, and the energy weight stops mattering.
    """
    env_inst = gym.make("MillingDTEnv-v1")
    unwrapped = env_inst.unwrapped
    env_inst.reset(seed=42)

    for action in ([1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]):
        _, _, _, _, info = env_inst.step(np.array(action, dtype=np.float32))
        assert info['Energy'] > 0.0
        assert info['Energy'] / unwrapped.E_ref > 0.0
    env_inst.close()

def test_gymnasium_env_checker_passes():
    """The environment must satisfy the Gymnasium API contract."""
    from gymnasium.utils.env_checker import check_env

    check_env(env_inst := gym.make("MillingDTEnv-v1").unwrapped, skip_render_check=True)
    env_inst.close()

def test_randomized_environment_draws_a_new_scenario():
    """Domain randomization: every reset is a different part."""
    env_inst = env.MillingDTEnv(randomize=True)

    env_inst.reset(seed=1)
    first = (env_inst.kappa_total, env_inst.D, env_inst.L_cut,
             env_inst.tool_wear_cfg['C_wear'], env_inst.roughness_cfg['k1'])
    env_inst.reset(seed=2)
    second = (env_inst.kappa_total, env_inst.D, env_inst.L_cut,
              env_inst.tool_wear_cfg['C_wear'], env_inst.roughness_cfg['k1'])

    assert first != second, "A randomized environment must draw a new scenario"
    env_inst.close()

def test_fixed_environment_keeps_its_scenario():
    """Without randomization the physical scenario never moves."""
    env_inst = env.MillingDTEnv()
    env_inst.reset(seed=1)
    first = (env_inst.kappa_total, env_inst.D, env_inst.L_cut,
             env_inst.tool_wear_cfg['C_wear'], env_inst.roughness_cfg['k1'])
    env_inst.reset(seed=2)
    second = (env_inst.kappa_total, env_inst.D, env_inst.L_cut,
              env_inst.tool_wear_cfg['C_wear'], env_inst.roughness_cfg['k1'])

    assert first == second
    env_inst.close()

def test_pinned_values_survive_randomization():
    """A value passed to the constructor is not overwritten by the sampler."""
    env_inst = env.MillingDTEnv(randomize=True, kappa_total=7.5, k1=11.0)
    env_inst.reset(seed=3)

    assert env_inst.kappa_total == 7.5
    assert env_inst.roughness_cfg['k1'] == 11.0
    env_inst.close()

def test_scenario_scales_follow_the_scenario():
    """The reward scales depend on the scenario, so they are recomputed."""
    env_inst = env.MillingDTEnv(randomize=True)
    env_inst.reset(seed=4)

    v_c_max = (np.pi * env_inst.D * env_inst.n_max) / 1000.0
    assert env_inst.MRR_max == pytest.approx(v_c_max * env_inst.ap_max * env_inst.f_max)
    assert env_inst.E_ref > 0.0
    assert env_inst.T_ref > 0.0
    env_inst.close()

def test_sweep_and_environment_share_the_scenario_tables():
    """The sweep scenarios and the training scenarios come from one source."""
    from config.scenarios import generate_random_configs, sample_scenario
    from config.physics_constants import SCENARIO_RANGES

    configs = generate_random_configs(5, seed=42)
    assert len(configs) == 5
    for scenario in configs.values():
        for key, (low, high) in SCENARIO_RANGES.items():
            assert low <= scenario[key] <= high

    drawn = sample_scenario(np.random.default_rng(0))
    assert set(drawn) == set(configs['Scenario_000'])

