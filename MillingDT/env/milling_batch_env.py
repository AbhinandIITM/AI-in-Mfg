"""
env/milling_batch_env.py

Digital Twin Batch Milling Environment (MillingDTBatchEnv)

One episode is a batch of parts cut with one tool. The tool is the resource
the parts share: the flank wear carries over from one part to the next, and
a tool that breaks or wears out is replaced and the replacement is paid for.

The environment wraps MillingDTEnv and adds nothing to the physics. The
per-part environment cuts the part, prices it and reports the state of the
cut. The batch environment decides when the part is over, moves the tool
into the next part, and keeps the count of the tool changes.

The classical controller solves one part at a time and has no notion of a
tool that is shared across the parts of a batch, so the batch objective is
where a learned policy can legitimately beat it.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any

from MillingDT.env.milling_env import MillingDTEnv, OBSERVATION_SIZE
from MillingDT.config.scenarios import generate_random_configs

# The features the batch adds on top of the observation of the part: the
# fraction of the tool life that is left, and the fraction of the batch that
# is still to cut.
BATCH_FEATURES = 2
BATCH_OBSERVATION_SIZE = OBSERVATION_SIZE + BATCH_FEATURES


class MillingDTBatchEnv(gym.Env):
    """Batch milling digital twin: one tool cuts a batch of parts.

    The observation has 13 features: the 11 features of the part the tool is
    cutting, then the fraction of the tool life that is left and the fraction
    of the batch that is still to cut.

    The reward is the reward of the part, as the per-part environment pays
    it, minus the cost of a tool change every time the tool is replaced. A
    tool is replaced when it breaks or when its wear passes VB_max, and the
    wear of the new tool starts at zero.

    Constructor arguments:
        batch_parts       the parts in one batch [count], default 10
        tool_change_cost  the cost of one tool change [reward units], default
                          500.0; a part finished within spec is worth the
                          completion bonus of 200
        randomize         draw the scenarios of the parts at every reset
        scenarios         the scenarios of the parts, as the dict of name to
                          scenario kwargs that generate_random_configs
                          returns; used when randomize is off
        w2                the energy weight of the parts; w1 = 1 - w2
        obs_noise         the noise on the measured state of the part

    Every other keyword is passed to the inner MillingDTEnv, so the machine
    limits, the engineering limits and the pinned physical values are set
    the same way as in the per-part environment.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, batch_parts: int = 10, tool_change_cost: float = 500.0,
                 randomize: bool = True,
                 scenarios: Optional[Dict[str, Dict[str, Any]]] = None,
                 render_mode: Optional[str] = None, **kwargs):
        super().__init__()

        self.batch_parts = int(batch_parts)
        if self.batch_parts < 1:
            raise ValueError("a batch holds at least one part")
        self.tool_change_cost = float(tool_change_cost)
        self.randomize = bool(randomize)
        self.render_mode = render_mode

        # The inner environment owns the physics, the reward shaping and the
        # observation of a part. It is never randomized itself: the batch
        # draws the scenario of every part and hands it over.
        self.inner = MillingDTEnv(randomize=False, **kwargs)
        self.inner.render_mode = render_mode

        # The energy weight of the parts and its complement. The inner
        # environment owns them, so the two can never disagree.
        self.reward_cfg = self.inner.reward_cfg

        # The scenarios of the parts, used when the batch is not randomized.
        self._fixed_scenarios = list(scenarios.values()) if scenarios else []
        self._scenarios = []

        self.action_space = self.inner.action_space
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(BATCH_OBSERVATION_SIZE,),
                                            dtype=np.float32)

        # The state of the batch: the wear of the tool, the number of parts
        # that are cut, the number of tool changes that are paid for and the
        # cycle time and the energy the batch has used so far.
        self.tool_VB = 0.0
        self.tool_changes = 0
        self.parts_done = 0
        self.part_index = 0
        self.batch_T_total = 0.0
        self.batch_E_total = 0.0
        self._inner_obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        # The seed of the episode. It fixes the scenarios of the parts and
        # the draws of the inner environment, so a batch replayed with the
        # same seed is the same batch.
        if seed is not None:
            episode_seed = int(seed)
        else:
            episode_seed = int(self.np_random.integers(0, 2 ** 31 - 1))

        if self.randomize:
            # A new batch of parts at every reset, drawn from the same tables
            # the per-part environment draws from.
            configs = generate_random_configs(self.batch_parts, seed=episode_seed)
            self._scenarios = list(configs.values())
        else:
            self._scenarios = list(self._fixed_scenarios)

        # The tool is new at the start of the batch, and so is the count.
        self.tool_VB = 0.0
        self.tool_changes = 0
        self.parts_done = 0
        self.batch_T_total = 0.0
        self.batch_E_total = 0.0

        # The inner environment draws the breakage from its own generator,
        # seeded here so that the batch is reproducible.
        self.inner.reset(seed=episode_seed)
        self._start_part(0)

        return self._get_obs(), {}

    def _start_part(self, index: int):
        """Set the inner environment up for the next part of the batch.

        The scenario of the part gives the geometry, the material and the
        coefficients of the tool. The flank wear is the wear the tool has
        when the part starts: for the first part it is the initial wear the
        scenario gives the tool, for the later parts it is the wear the tool
        carried over. A tool that was just changed starts fresh.
        """
        scenario = dict(self._scenarios[index]) if index < len(self._scenarios) else {}

        # The values pinned in the constructor win over the drawn ones.
        scenario.update(self.inner.fixed_scenario)

        if index == 0:
            # The tool arrives with the initial wear of the scenario, or
            # fresh if the scenario does not say.
            scenario.setdefault('VB_0', 0.0)
        else:
            # The wear of the tool is the state of the batch, so it wins
            # over anything the scenario of the part says.
            scenario['VB_0'] = self.tool_VB

        # The inner environment applies a scenario only when it randomizes
        # itself, so the scenario of the part is applied here.
        self.inner._apply_scenario(scenario)
        self._inner_obs, _ = self.inner.reset()
        self.part_index = index

    def step(self, action: np.ndarray):
        inner_obs, inner_reward, part_terminated, part_truncated, inner_info = self.inner.step(action)
        self._inner_obs = inner_obs

        # The wear of the tool is the wear the inner environment has.
        self.tool_VB = self.inner.VB

        reward = float(inner_reward)
        tool_changed = False
        terminated = False
        truncated = bool(part_truncated)

        if part_terminated:
            # The part is over: it is finished, or the tool broke, or the
            # wear of the tool passed the limit. A broken tool and a worn out
            # tool are both replaced, and the replacement is paid for.
            tool_changed = bool(self.inner.broken or self.inner.VB > self.inner.VB_max)
            if tool_changed:
                self.tool_changes += 1
                self.tool_VB = 0.0
                reward -= self.tool_change_cost

            self.parts_done += 1
            self.batch_T_total += self.inner.T_total
            self.batch_E_total += self.inner.E_total

            if self.parts_done < self.batch_parts:
                # The next part of the batch is cut with the tool that is
                # left, and with the wear that tool has.
                self._start_part(self.parts_done)
            else:
                # The last part of the batch is cut: the batch is over.
                terminated = True
        elif part_truncated:
            # The inner environment hit its limit on the passes of a part,
            # so the part was not finished and the batch cannot be counted.
            self.batch_T_total += self.inner.T_total
            self.batch_E_total += self.inner.E_total

        info = dict(inner_info)
        info.update({
            "part": self.part_index,
            "parts_done": self.parts_done,
            "tool_VB": self.tool_VB,
            "tool_changes": self.tool_changes,
            "tool_changed": tool_changed,
            "tool_change_cost": self.tool_change_cost if tool_changed else 0.0,
            "tool_life": self._tool_life(),
            "batch_remaining": self._batch_remaining(),
            # The reward of the pass, as the per-part environment paid it,
            # before the cost of the tool change.
            "inner_reward": float(inner_reward),
            "batch_T_total": self.batch_T_total,
            "batch_E_total": self.batch_E_total,
        })
        return self._get_obs(), reward, terminated, truncated, info

    def _tool_life(self) -> float:
        """The fraction of the tool life that is left."""
        return float(max(0.0, 1.0 - self.tool_VB / self.inner.VB_max))

    def _batch_remaining(self) -> float:
        """The fraction of the batch that is still to cut."""
        return float(1.0 - self.parts_done / self.batch_parts)

    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate([
            self._inner_obs,
            np.array([self._tool_life(), self._batch_remaining()], dtype=np.float32),
        ])
        return obs.astype(np.float32)

    def render(self):
        """Render the state of the part that is being cut."""
        return self.inner.render()
