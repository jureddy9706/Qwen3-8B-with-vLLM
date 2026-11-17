"""Bounded in-process admission. Limits each gateway replica, not an entire cluster."""

from __future__ import annotations

import asyncio
import time


class AdmissionRejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class AdmissionLimiter:
    """A small bounded waiting room around a fixed number of active requests.

    Condition wakeups are not guaranteed FIFO. Does not provide global cluster
    coordination. Each replica must be sized independently.
    """

    def __init__(self, max_active: int, max_waiting: int, wait_timeout_s: float):
        if max_active < 1 or max_waiting < 0 or wait_timeout_s <= 0:
            raise ValueError("Invalid admission limits")
        self.max_active = max_active
        self.max_waiting = max_waiting
        self.wait_timeout_s = wait_timeout_s
        self.active = 0
        self.waiting = 0
        self._condition = asyncio.Condition()

    async def acquire(self) -> float:
        started = time.perf_counter()
        async with self._condition:
            if self.active < self.max_active:
                self.active += 1
                return 0.0
            if self.waiting >= self.max_waiting:
                raise AdmissionRejected("queue_full")
            self.waiting += 1
            try:
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(lambda: self.active < self.max_active),
                        timeout=self.wait_timeout_s,
                    )
                except TimeoutError as exc:
                    raise AdmissionRejected("queue_timeout") from exc
                self.active += 1
                return time.perf_counter() - started
            finally:
                self.waiting -= 1

    async def release(self) -> None:
        async with self._condition:
            if self.active <= 0:
                raise RuntimeError("Unbalanced admission release")
            self.active -= 1
            self._condition.notify(1)
