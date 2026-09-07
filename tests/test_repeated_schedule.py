from types import SimpleNamespace

import pytest
import torch

from tinybench_lm.repeated_schedule import RepeatedScheduledStream
from tinybench_lm.schedule import ScheduleContractError


class SmallStream:
    content_hash = "fixture-schedule"
    schedule = SimpleNamespace(sequence_count=3)

    def __init__(self):
        self.cursor = SimpleNamespace(position=0)

    def get_batch(self, size, seq_len, device):
        values = torch.tensor([(self.cursor.position + i) % 3 for i in range(size)])
        self.cursor.position = (self.cursor.position + size) % 3
        return values, values + 1


def test_resume_across_epoch_boundary_preserves_absolute_position():
    a = RepeatedScheduledStream(SmallStream(), 2)
    assert a.get_batch(4, 1, "cpu")[0].tolist() == [0, 1, 2, 0]
    state = a.state_dict()
    b = RepeatedScheduledStream(SmallStream(), 2)
    b.load_state_dict(state)
    assert b.position == 4
    assert torch.equal(a.get_batch(2, 1, "cpu")[0], b.get_batch(2, 1, "cpu")[0])
    with pytest.raises(ScheduleContractError, match="exhausted"):
        b.get_batch(1, 1, "cpu")
    assert b.position == 6


def test_epoch_count_and_schedule_are_bound_to_resume_identity():
    a = RepeatedScheduledStream(SmallStream(), 2)
    b = RepeatedScheduledStream(SmallStream(), 3)
    assert a.content_hash != b.content_hash
    with pytest.raises(ScheduleContractError, match="identity"):
        b.load_state_dict(a.state_dict())
    state = a.state_dict()
    state["schedule_cursor"] = 7
    with pytest.raises(ScheduleContractError, match="cursor"):
        a.load_state_dict(state)
