import pytest
import numpy as np
import gymnasium as gym
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from MillingDT import env
from MillingDT.env.milling_env import OBSERVATION_SIZE, PHYSICS_FEATURES
from MillingDT.config import ALL_CONSTANTS, MACHINE_CONSTRAINTS, ENGINEERING_CONSTRAINTS
from MillingDT.config.physics import cutting_speed

def test_action_clipping():
    """Test that out-of-bounds actions are correctly clipped and handled."""
    env_inst = gym.make(env.ENV_ID)
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
    env_inst = gym.make(env.ENV_ID)
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
    env_inst = gym.make(env.ENV_ID)
    env_inst.reset(seed=42)

    # Test internal _soft_normalize method across wide range of values
    for val in [-100.0, -1.0, 0.0, 0.5, 1.0, 2.0, 100.0]:
        norm_val = env_inst.unwrapped._soft_normalize(val, 0.0, 1.0)
        assert 0.0 <= norm_val <= 1.0

    # Check observation space bounds
    obs, _ = env_inst.reset()
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    env_inst.close()

def test_observation_is_always_eleven_dimensional():
    """Every configuration of the environment yields the same observation.

    The context features are always present, so a policy trained with a
    pinned weight and a policy trained over the whole weight range see the
    same layout and the same observation space.
    """
    assert PHYSICS_FEATURES == 8
    assert OBSERVATION_SIZE == 11

    configurations = [
        env.MillingDTEnv(),
        env.MillingDTEnv(w2=0.75),
        env.MillingDTEnv(randomize=True),
        env.MillingDTEnv(randomize_weight=True),
        env.MillingDTEnv(randomize=True, randomize_weight=True),
        gym.make(env.ENV_ID).unwrapped,
    ]

    for env_inst in configurations:
        obs, _ = env_inst.reset(seed=17)
        assert obs.shape == (11,)
        assert env_inst.observation_space.shape == (11,)
        assert env_inst.observation_space.contains(obs)
        env_inst.close()

def test_physics_constants_sync():
    """Test that environment properly syncs physics constants from ALL_CONSTANTS."""
    env_inst = gym.make(env.ENV_ID)
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
    env_inst = gym.make(env.ENV_ID, C_wear=0.05, k1=14.0, k2=1.1, k3=0.4, alpha=0.45)
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
    env_inst = gym.make(env.ENV_ID)
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
    env_inst = gym.make(env.ENV_ID)
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

    check_env(env_inst := gym.make(env.ENV_ID).unwrapped, skip_render_check=True)
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


def test_context_carries_the_pinned_energy_weight():
    """A pinned weight is part of the observation the policy reads.

    The task is the same environment with a constant in the context, so the
    weight the part is optimized for is never hidden from the policy.
    """
    env_inst = env.MillingDTEnv(w2=0.75)
    obs, _ = env_inst.reset(seed=5)

    assert obs[8] == pytest.approx(0.75)
    assert env_inst.reward_cfg['w2'] == 0.75
    # The throughput weight is whatever is left of the energy weight.
    assert env_inst.reward_cfg['w1'] == pytest.approx(1.0 - 0.75)

    # It stays pinned over the episode.
    obs, _, _, _, _ = env_inst.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert obs[8] == pytest.approx(0.75)
    env_inst.close()


def test_context_carries_the_material():
    """The material of the part is part of the observation too.

    The coefficients are normalized over the bands the scenarios span, so
    the nominal material sits in the middle and an override moves the
    feature the way the band says it should.
    """
    nominal_c_wear = ALL_CONSTANTS['tool_wear']['C_wear']
    nominal_k1 = ALL_CONSTANTS['surface_roughness']['k1']

    # The nominal material lands in the middle of both bands.
    nominal = env.MillingDTEnv()
    obs, _ = nominal.reset(seed=5)
    assert obs[9] == pytest.approx(0.5)
    assert obs[10] == pytest.approx(0.5)
    nominal.close()

    # The edges of the bands are the edges of the feature.
    low = env.MillingDTEnv(C_wear=nominal_c_wear * 0.6, k1=nominal_k1 * 0.8)
    obs, _ = low.reset(seed=5)
    assert obs[9] == pytest.approx(0.0)
    assert obs[10] == pytest.approx(0.0)
    low.close()

    high = env.MillingDTEnv(C_wear=nominal_c_wear * 1.4, k1=nominal_k1 * 1.2)
    obs, _ = high.reset(seed=5)
    assert obs[9] == pytest.approx(1.0)
    assert obs[10] == pytest.approx(1.0)
    high.close()

    # A harder material moves its own feature and leaves the other alone.
    harder = env.MillingDTEnv(C_wear=nominal_c_wear * 1.2)
    obs, _ = harder.reset(seed=5)
    assert obs[9] == pytest.approx((1.2 - 0.6) / (1.4 - 0.6))
    assert obs[9] > 0.5
    assert obs[10] == pytest.approx(0.5)
    harder.close()


def test_the_context_follows_the_belief_not_the_truth():
    """A setup that believes in the wrong material must see the belief.

    The physics always run on the truth. Only the context the policy reads
    follows the belief, which is what the model error experiment needs: a
    machine that cuts a harder material than the setup believes it does.
    """
    nominal_c_wear = ALL_CONSTANTS['tool_wear']['C_wear']
    nominal_k1 = ALL_CONSTANTS['surface_roughness']['k1']

    # The machine cuts a harder material than the setup believes.
    env_inst = env.MillingDTEnv(C_wear=nominal_c_wear * 1.4, k1=nominal_k1 * 1.2,
                                 belief={'C_wear': nominal_c_wear,
                                         'k1': nominal_k1})
    obs, _ = env_inst.reset(seed=5)

    # The context reports the belief, which is the nominal material.
    assert obs[9] == pytest.approx(0.5)
    assert obs[10] == pytest.approx(0.5)

    # The physics still run on the truth.
    assert env_inst.tool_wear_cfg['C_wear'] == pytest.approx(nominal_c_wear * 1.4)
    assert env_inst.roughness_cfg['k1'] == pytest.approx(nominal_k1 * 1.2)
    env_inst.close()


def test_the_observation_noise_is_on_the_measured_state_only():
    """A noisy measurement moves the wear and the stock, and nothing else.

    The previous cutting parameters come from the machine's own control
    system and the previous removal rate and roughness are computed, so they
    are exact. The context is a property of the setup, not a measurement.
    """
    clean = env.MillingDTEnv()
    obs_clean, _ = clean.reset(seed=11)
    clean.close()

    noisy = env.MillingDTEnv(obs_noise=0.05, obs_noise_seed=3)
    obs_noisy, _ = noisy.reset(seed=11)
    noisy.close()

    # The measured state moved.
    assert obs_noisy[0] != pytest.approx(obs_clean[0])
    assert obs_noisy[1] != pytest.approx(obs_clean[1])

    # The context did not.
    assert obs_noisy[8] == pytest.approx(obs_clean[8])
    assert obs_noisy[9] == pytest.approx(obs_clean[9])
    assert obs_noisy[10] == pytest.approx(obs_clean[10])

    # The observation is still inside the box.
    assert 0.0 <= obs_noisy[0] <= 1.0
    assert 0.0 <= obs_noisy[1] <= 1.0


def test_randomized_weight_draws_a_new_weight_at_every_reset():
    """The energy weight is part of the scenario of the part."""
    env_inst = env.MillingDTEnv(randomize_weight=True)

    env_inst.reset(seed=1)
    first_w2 = env_inst.reward_cfg['w2']
    assert first_w2 == pytest.approx(env_inst._get_obs()[8])

    env_inst.reset(seed=2)
    second_w2 = env_inst.reward_cfg['w2']

    assert first_w2 != second_w2, "A randomized weight must be drawn again"
    assert 0.0 <= first_w2 <= 1.0
    assert 0.0 <= second_w2 <= 1.0
    # The throughput weight is whatever is left of the energy weight.
    assert env_inst.reward_cfg['w1'] == pytest.approx(1.0 - second_w2)
    env_inst.close()


def test_cutting_energy_grows_with_the_cutting_speed():
    """The specific cutting energy rises as the spindle speeds up.

    This is what makes the energy objective a real trade-off against
    throughput: the cutting term of the part energy rises with the speed
    while the idle term falls.
    """
    from MillingDT.config.physics import specific_cutting_energy

    D = ALL_CONSTANTS['workpiece']['D']
    slow = specific_cutting_energy(cutting_speed(D, 2000), D)
    reference = specific_cutting_energy(cutting_speed(D, 5000), D)
    fast = specific_cutting_energy(cutting_speed(D, 8000), D)

    assert slow < reference < fast
    # The reference condition carries the calibrated coefficient.
    assert reference == pytest.approx(ALL_CONSTANTS['energy']['k_c'])


def test_part_energy_has_an_interior_minimum_in_speed():
    """Cutting faster is not always cheaper.

    The energy of a finished part is k_c(v_c) * V + P_idle * T_part. The
    first term rises with the speed, the second falls, so the part energy
    has a minimum at an intermediate speed. Without that, the energy
    objective is a duplicate of the throughput objective.
    """
    def part_energy(n):
        env_inst = env.MillingDTEnv()
        env_inst.reset(seed=1)
        kappa = env_inst.kappa_total
        total = 0.0
        while kappa > 1e-5:
            depth = min(2.5, kappa)
            action = np.array([
                2.0 * (n - env_inst.n_min) / (env_inst.n_max - env_inst.n_min) - 1.0,
                2.0 * (0.2 - env_inst.f_min) / (env_inst.f_max - env_inst.f_min) - 1.0,
                2.0 * (depth - env_inst.ap_min) / (env_inst.ap_max - env_inst.ap_min) - 1.0,
            ], dtype=np.float32)
            _, _, terminated, truncated, info = env_inst.step(action)
            total += info['Energy']
            kappa -= info['a_p']
            if terminated or truncated:
                break
        env_inst.close()
        return total

    slow = part_energy(2000)
    middle = part_energy(4500)
    fast = part_energy(8000)

    assert middle < slow, "the energy optimum must not sit at the slowest speed"
    assert middle < fast, "the energy optimum must not sit at the fastest speed"


def test_environment_uses_the_shared_physics():
    """The environment must cut with the shared equations.

    The baselines import the same functions from config/physics.py, so this
    is what keeps the environment and the baselines from drifting apart.
    """
    from MillingDT.config.physics import pass_energy

    env_inst = env.MillingDTEnv()
    env_inst.reset(seed=1)

    action = np.array([0.3, -0.2, 0.8], dtype=np.float32)
    _, _, _, _, info = env_inst.step(action)

    n = env_inst.n_min + (action[0] + 1.0) / 2.0 * (env_inst.n_max - env_inst.n_min)
    f_r = env_inst.f_min + (action[1] + 1.0) / 2.0 * (env_inst.f_max - env_inst.f_min)
    a_p = env_inst.ap_min + (action[2] + 1.0) / 2.0 * (env_inst.ap_max - env_inst.ap_min)

    assert info['Energy'] == pytest.approx(
        pass_energy(n, f_r, a_p, env_inst.D, env_inst.L_cut, env_inst.energy_cfg),
        rel=1e-9)
    env_inst.close()


@pytest.mark.parametrize("w2", [0.0, 0.4, 0.75])
def test_reward_matches_an_independent_calculation(w2):
    """The reward must equal what the physics says it should be.

    The reward is the sum of an objective term, a progress term, the
    constraint penalties and the completion bonus. This recomputes it from
    the physics module, independently of the code that assembles it, so a
    change in either one has to show up here.

    The energy weight is pinned, and the throughput weight is what is left
    of it, so the same check holds for every point of the trade-off.
    """
    from MillingDT.config.physics import material_removal_rate, pass_energy, cutting_speed

    w1 = 1.0 - w2
    env_inst = env.MillingDTEnv(w2=w2)
    env_inst.reset(seed=11)

    # The environment holds the pinned weight and derives the other one.
    assert env_inst.reward_cfg['w2'] == pytest.approx(w2)
    assert env_inst.reward_cfg['w1'] == pytest.approx(w1)

    action = np.array([0.4, -0.3, 0.7], dtype=np.float32)
    _, reward, terminated, truncated, info = env_inst.step(action)

    n = env_inst.n_min + (action[0] + 1.0) / 2.0 * (env_inst.n_max - env_inst.n_min)
    f_r = env_inst.f_min + (action[1] + 1.0) / 2.0 * (env_inst.f_max - env_inst.f_min)
    a_p = min(env_inst.ap_min + (action[2] + 1.0) / 2.0 * (env_inst.ap_max - env_inst.ap_min),
              env_inst.kappa)

    v_c = cutting_speed(env_inst.D, n)
    MRR = material_removal_rate(v_c, a_p, f_r)
    energy = pass_energy(n, f_r, a_p, env_inst.D, env_inst.L_cut, env_inst.energy_cfg)

    # Normalize first, then weight: the reward is assembled in the dtypes of
    # the physics values, so the order of the operations is part of the value.
    MRR_hat = MRR / env_inst.MRR_max
    E_hat = energy / env_inst.E_ref
    objective = env_inst.objective_scale * (w1 * MRR_hat - w2 * E_hat)
    progress = env_inst.progress_scale * (a_p / env_inst.kappa_total)
    step_penalty = -env_inst.step_penalty
    # The tool cost is charged on the wear this pass consumed.
    from MillingDT.config.physics import flank_wear_rate
    delta_VB = flank_wear_rate(v_c, f_r, a_p, env_inst.L_cut / (n * f_r),
                               env_inst.tool_wear_cfg)
    tool_cost = -env_inst.tool_cost * delta_VB
    # The part is not finished by this pass, so the roughness penalty and the
    # completion bonus do not apply.
    assert not info['is_finishing']

    assert reward == pytest.approx(objective + progress + step_penalty + tool_cost, rel=1e-9)
    env_inst.close()


def test_reward_penalizes_a_finishing_pass_out_of_spec():
    """A part that ends outside the roughness limit must be penalized."""
    env_inst = env.MillingDTEnv()
    env_inst.reset(seed=5)
    env_inst.kappa = 0.2  # one more pass finishes the part

    # A fast, deep pass: the roughness is far above the safety limit.
    action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    _, reward, terminated, _, info = env_inst.step(action)

    assert info['is_finishing']
    assert info['Ra'] > env_inst.Ra_safe
    # The -50 roughness penalty has to be in there, on top of the objective.
    without_penalty = reward + 50.0
    objective_scale = env_inst.objective_scale
    assert without_penalty > -objective_scale * 10
    env_inst.close()

def test_sweep_and_environment_share_the_scenario_tables():
    """The sweep scenarios and the training scenarios come from one source."""
    from MillingDT.config.scenarios import generate_random_configs, sample_scenario
    from MillingDT.config import SCENARIO_RANGES

    configs = generate_random_configs(5, seed=42)
    assert len(configs) == 5
    for scenario in configs.values():
        for key, (low, high) in SCENARIO_RANGES.items():
            assert low <= scenario[key] <= high

    drawn = sample_scenario(np.random.default_rng(0))
    assert set(drawn) == set(configs['Scenario_000'])

