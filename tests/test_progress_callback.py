import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.train_rl import ProgressQueueCallback


class _RecordingQueue:
    """Stands in for multiprocessing.Queue in the heartbeat test."""

    def __init__(self):
        self.payloads = []

    def put_nowait(self, payload):
        self.payloads.append(payload)


class _BrokenQueue:
    """A queue that refuses the payload, the way a closed pipe would."""

    def put_nowait(self, payload):
        raise RuntimeError("queue is closed")


def _armed_callback(queue, timesteps=1000, rewards=(1.0, 3.0)):
    callback = ProgressQueueCallback(queue)
    callback.init_callback(None)
    callback.num_timesteps = timesteps
    callback.locals = {"rewards": list(rewards), "infos": [{}]}
    return callback


def test_progress_queue_heartbeat_is_delivered():
    """The heartbeat must reach the supervisor.

    A dropped heartbeat is indistinguishable from a hung worker, and the
    stall detector terminates any job that goes 300 seconds without one.
    """
    queue = _RecordingQueue()
    callback = _armed_callback(queue)

    assert callback._on_step() is True
    assert len(queue.payloads) == 1

    payload = queue.payloads[0]
    assert payload['timesteps'] == 1000
    assert payload['reward'] == pytest.approx(2.0)
    assert 'timestamp' in payload


def test_progress_queue_heartbeat_reports_failures():
    """A queue failure must be reported, not swallowed by a bare except."""
    callback = _armed_callback(_BrokenQueue())

    # Training continues: a lost heartbeat must not kill the run.
    assert callback._on_step() is True


def test_progress_queue_heartbeat_respects_interval():
    """Heartbeats are sent on the report interval, not on every step."""
    queue = _RecordingQueue()
    callback = _armed_callback(queue, timesteps=1500)

    callback._on_step()
    assert queue.payloads == []

    callback.num_timesteps = 2000
    callback._on_step()
    assert len(queue.payloads) == 1
