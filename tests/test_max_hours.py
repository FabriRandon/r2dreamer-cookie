"""trainer.max_hours: a wall-clock limit for one session of training."""

import time
import types

from trainer import OnlineTrainer


def out_of_time(max_seconds, started):
    return OnlineTrainer._out_of_time(types.SimpleNamespace(_max_seconds=max_seconds), started)


def test_no_limit_never_stops():
    assert not out_of_time(None, time.time() - 10**6)


def test_stops_only_after_the_limit():
    assert not out_of_time(3600, time.time() - 10)
    assert out_of_time(3600, time.time() - 3601)
    assert out_of_time(0, time.time())
