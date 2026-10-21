import pytest
import numpy as np
import gymnasium as gym
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from MillingDT import env
from MillingDT.env.milling_batch_env import BATCH_OBSERVATION_SIZE

# A constant action: the middle of the machine envelope. The batch is cut
# with it from the first pass to the last.
ACTION = np.array([0.0, 0.0, 0.0], dtype=np.float32)


class _AlwaysBreak:
    """A generator whose draws always fall below the breakage probability.

    The breakage of the per-part environment is a Bernoulli draw. Forcing the
    draw keeps the test on the batch logic and off the question of which seed
    happens to break a tool.
    """

    def random(self):
        return 0.0


def _break_always(env_inst):
    """Make the tool of the inner environment break on the first pass it can."""
    setattr(env_inst.unwrapped.inner, '_np_random', _AlwaysBreak())


def _rollout(env_inst, action, seed=42, after_reset=None):
    """Cut a whole batch with one action and collect what happened.

    Returns the observations, the rewards and the info payload of every step,
    with the observation of the reset at the head of the observations.
    """
    obs, _ = env_inst.reset(seed=seed)
    if after_reset is not None:
        after_reset(env_inst)

    observations = [obs]
    rewards = []
    infos = []
    terminated = truncated = False

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env_inst.step(action)
        observations.append(obs)
        rewards.append(reward)
        infos.append(info)
        assert len(infos) < 1000, "the batch did not finish"

    return observations, rewards, infos


def _part_starts(observations, infos):
    """The observation the tool saw when it entered each part of the batch."""
    starts = {0: observations[0]}
    for obs, info in zip(observations[1:], infos):
        starts.setdefault(info["parts_done"], obs)
    return starts


def test_batch_environment_has_the_batch_observation():
    """The observation is the observation of the part plus the two batch features."""
    env_inst = gym.make(env.BATCH_ENV_ID)
    obs, _ = env_inst.reset(seed=42)

    assert BATCH_OBSERVATION_SIZE == 13
    assert obs.shape == (13,)
    assert env_inst.observation_space.shape == (13,)
    assert env_inst.observation_space.contains(obs)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    env_inst.close()


def test_batch_observation_reports_the_tool_life_and_the_batch_left():
    """The last two features are the life of the tool and the batch left."""
    batch_parts = 3
    env_inst = env.MillingDTBatchEnv(batch_parts=batch_parts, randomize=False)
    observations, rewards, infos = _rollout(env_inst, ACTION)

    starts = _part_starts(observations, infos)

    # The tool starts fresh and the whole batch is left to cut.
    assert starts[0][11] == pytest.approx(1.0)
    assert starts[0][12] == pytest.approx(1.0)

    for part in range(1, batch_parts):
        # The wear the tool carries reduces the life it has left.
        assert starts[part][11] < starts[part - 1][11]
        # The fraction of the batch that is left falls by one part at a time.
        assert starts[part][12] == pytest.approx(1.0 - part / batch_parts)
    env_inst.close()


def test_batch_rollout_cuts_every_part_of_the_batch():
    """A constant action cuts the whole batch, and the batch ends with it."""
    batch_parts = 4
    env_inst = env.MillingDTBatchEnv(batch_parts=batch_parts, randomize=False)
    observations, rewards, infos = _rollout(env_inst, ACTION)

    assert infos[-1]["parts_done"] == batch_parts
    assert infos[-1]["batch_remaining"] == pytest.approx(0.0)
    # Every part of the batch was cut, and the last step ended the batch.
    assert sorted({info["part"] for info in infos}) == list(range(batch_parts))
    env_inst.close()


def test_a_batch_that_changes_the_tool_cuts_the_rest_with_the_new_one():
    """A batch whose tool is retired goes on with a fresh tool."""
    batch_parts = 6
    env_inst = env.MillingDTBatchEnv(batch_parts=batch_parts, randomize=False,
                                     scenarios={"Scenario_000": {"VB_0": 0.2}})
    observations, rewards, infos = _rollout(env_inst, ACTION, after_reset=_break_always)

    # The tool was changed once, and the batch was still cut to the end.
    assert infos[-1]["tool_changes"] == 1
    assert infos[-1]["parts_done"] == batch_parts

    # The part after the change is cut with a tool that has its full life.
    starts = _part_starts(observations, infos)
    assert starts[1][11] == pytest.approx(1.0)
    env_inst.close()


def test_the_wear_carries_over_from_one_part_to_the_next():
    """The second part is cut with the wear the first part left behind."""
    env_inst = env.MillingDTBatchEnv(batch_parts=3, randomize=False)
    observations, rewards, infos = _rollout(env_inst, ACTION)

    starts = _part_starts(observations, infos)

    # The first part is cut with a fresh tool, the second one with the wear
    # the first part left behind.
    assert starts[0][11] == pytest.approx(1.0)
    assert starts[1][11] < starts[0][11]
    assert starts[2][11] < starts[1][11]
    # The wear of the tool shows up in the observation of the part too.
    assert starts[1][0] > starts[0][0]
    env_inst.close()


def test_a_broken_tool_ends_the_part_and_triggers_a_tool_change():
    """A tool that breaks is replaced, and the replacement is paid for."""
    env_inst = env.MillingDTBatchEnv(batch_parts=2, randomize=False,
                                     scenarios={"Scenario_000": {"VB_0": 0.2}})
    observations, rewards, infos = _rollout(env_inst, ACTION, after_reset=_break_always)

    first = infos[0]

    # The tool broke, so the part ended before it was finished.
    assert first["broken"]
    assert not first["is_finishing"]
    assert first["tool_changed"]
    assert first["tool_changes"] == 1
    # The wear of the new tool starts at zero, and the change is charged.
    assert first["tool_VB"] == pytest.approx(0.0)
    assert first["tool_change_cost"] == pytest.approx(500.0)

    # The batch goes on with the new tool.
    assert first["parts_done"] == 1
    env_inst.close()


def test_a_tool_that_reaches_the_wear_limit_is_replaced():
    """A tool that is worn out is replaced, and the replacement is paid for."""
    env_inst = env.MillingDTBatchEnv(batch_parts=2, randomize=False, VB_max=0.05,
                                     scenarios={"Scenario_000": {"VB_0": 0.0495,
                                                                 "kappa_total": 15.0}})
    observations, rewards, infos = _rollout(env_inst, ACTION)

    worn_out = [info for info in infos if info["tool_changed"]]
    # The tool wears out during the first part. How many parts it survives
    # depends on the wear rate, so the count is not pinned here.
    assert len(worn_out) >= 1

    first = worn_out[0]

    # The wear of the tool passed the limit, so the tool was replaced.
    assert first["VB"] > 0.05
    assert not first["broken"]
    assert first["tool_changes"] >= 1
    assert first["tool_VB"] == pytest.approx(0.0)
    assert first["parts_done"] >= 1
    env_inst.close()


def test_batch_reward_is_the_part_rewards_plus_the_tool_changes():
    """The reward of the batch is the reward of the parts and the tool changes."""
    batch_parts = 3
    tool_change_cost = 500.0
    scenarios = {"Scenario_000": {"VB_0": 0.2}}

    env_inst = env.MillingDTBatchEnv(batch_parts=batch_parts, tool_change_cost=tool_change_cost,
                                     randomize=False, scenarios=scenarios)
    observations, rewards, infos = _rollout(env_inst, ACTION, after_reset=_break_always)

    changes = sum(1 for info in infos if info["tool_changed"])
    part_rewards = sum(info["inner_reward"] for info in infos)

    assert changes >= 1, "the batch has to change a tool for the accounting to mean anything"
    assert sum(rewards) == pytest.approx(part_rewards - tool_change_cost * changes)

    # The same batch with the tool changes for free: the difference between
    # the two runs is the cost of the changes and nothing else.
    free = env.MillingDTBatchEnv(batch_parts=batch_parts, tool_change_cost=0.0,
                                  randomize=False, scenarios=scenarios)
    _, free_rewards, free_infos = _rollout(free, ACTION, after_reset=_break_always)

    assert free_infos[-1]["tool_changes"] == changes
    assert sum(rewards) == pytest.approx(sum(free_rewards) - tool_change_cost * changes)
    env_inst.close()


def test_the_batch_is_reproducible_with_a_seed():
    """The same seed gives the same batch, and different seeds give different ones."""
    first = env.MillingDTBatchEnv(batch_parts=3)
    second = env.MillingDTBatchEnv(batch_parts=3)
    other = env.MillingDTBatchEnv(batch_parts=3)

    obs_a, rewards_a, _ = _rollout(first, ACTION, seed=7)
    obs_b, rewards_b, _ = _rollout(second, ACTION, seed=7)
    obs_c, rewards_c, _ = _rollout(other, ACTION, seed=8)

    assert rewards_a == rewards_b
    for a, b in zip(obs_a, obs_b):
        assert np.allclose(a, b)

    # A different seed draws a different batch of parts.
    assert not np.allclose(obs_a[0], obs_c[0])
    assert rewards_a != rewards_c
    first.close()
    second.close()
    other.close()


def test_the_energy_weight_is_the_same_for_every_part():
    """The weight of the parts is the weight the per-part environment uses."""
    env_inst = env.MillingDTBatchEnv(batch_parts=2, randomize=False, w2=0.75)
    observations, rewards, infos = _rollout(env_inst, ACTION)

    for obs in observations:
        # The ninth feature of the per-part observation is the energy weight.
        assert obs[8] == pytest.approx(0.75)
    assert env_inst.reward_cfg['w1'] == pytest.approx(1.0 - 0.75)
    env_inst.close()


def test_gymnasium_env_checker_passes():
    """The environment must satisfy the Gymnasium API contract."""
    from gymnasium.utils.env_checker import check_env

    check_env(env_inst := gym.make(env.BATCH_ENV_ID).unwrapped, skip_render_check=True)
    env_inst.close()
