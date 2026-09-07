"""Explicit finite repetition of a verified one-pass schedule with an absolute cursor."""

import hashlib
import json

from .schedule import CURSOR_STATE_KEY, ScheduleContractError


REPEAT_POLICY = "FINITE_IDENTICAL_SCHEDULE_PASSES_V1"


class RepeatedScheduledStream:
    """Repeat references without pretending repeated tokens are distinct corpus data."""

    def __init__(self, stream, epochs: int):
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 2:
            raise ValueError("Repeated schedules need at least two explicit epochs")
        if stream.schedule.sequence_count < 1 or stream.cursor.position != 0:
            raise ValueError("Repeated schedules require a nonempty base stream at its initial cursor")
        self.stream = stream
        self.stream.wrap = True
        self.schedule = stream.schedule
        self.epochs = int(epochs)
        self.position = 0
        self.sequence_count = self.schedule.sequence_count * self.epochs
        self.last_batch_reference_hash: str | None = None
        self.last_batch_entries = ()
        self.content_hash = hashlib.sha256(json.dumps({
            "policy": REPEAT_POLICY, "base_schedule_hash": stream.content_hash,
            "epochs": self.epochs,
        }, sort_keys=True).encode()).hexdigest()

    def state_dict(self):
        return {"format_version": 1, "schedule_content_hash": self.content_hash,
                CURSOR_STATE_KEY: self.position}

    def load_state_dict(self, state):
        position = state[CURSOR_STATE_KEY]
        if (state.get("format_version") != 1 or not isinstance(position, int)
                or isinstance(position, bool) or state["schedule_content_hash"] != self.content_hash
                or not 0 <= position <= self.sequence_count):
            raise ScheduleContractError("Repeated schedule identity or absolute cursor mismatch")
        self.position = position
        self.stream.cursor.position = position % self.schedule.sequence_count

    def get_batch(self, batch_size, seq_len, device):
        if self.position + batch_size > self.sequence_count:
            raise ScheduleContractError("Explicit repeated schedule is exhausted")
        result = self.stream.get_batch(batch_size, seq_len, device)
        self.last_batch_reference_hash = getattr(self.stream, "last_batch_reference_hash", None)
        self.last_batch_entries = getattr(self.stream, "last_batch_entries", ())
        self.position += batch_size
        return result

    def close(self):
        self.stream.close()
