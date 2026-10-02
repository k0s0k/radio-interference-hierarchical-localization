"""Offline task model, NOT an official HTTP simulator or submission adapter."""

from __future__ import annotations

import copy
import hashlib
import math
import random
import time
import numpy as np


def make_scenario(seed, distribution="uniform"):
    rng = random.Random(int(seed))
    n = rng.randint(10, 16)
    channels = rng.sample(range(1, 21), n)
    sources = []
    for i, ch in enumerate(channels):
        angle, radius = rng.uniform(0, 2 * math.pi), 1800 * math.sqrt(rng.random())
        if distribution == "boundary":
            radius = rng.uniform(1770, 1800)
        elif distribution == "cluster":
            angle, radius = rng.uniform(-0.05, 0.05), rng.uniform(1400, 1600)
        elif distribution == "collinear":
            angle, radius = (0 if i % 2 else math.pi), rng.uniform(100, 1799)
        elif distribution != "uniform":
            raise ValueError(distribution)
        kind = rng.choice(("omni",))  # preserve the original generator's RNG order
        reception = rng.uniform(1000, 1500)
        sources.append(
            dict(
                pos=(radius * math.cos(angle), radius * math.sin(angle)),
                channel=ch,
                kind=kind,
                reception=reception,
                cleared=False,
            )
        )
    return sources


class TaskSimulator:
    """Readings share a bounded fixed spatial error field across channels.

    Uniform source placement and this error field are modelling assumptions.
    Actual wall-clock time is separate from virtual task time. No artificial
    sleep or fixed per-request latency is substituted for wall-clock time.
    """

    def __init__(
        self,
        sources,
        seed=0,
        error_mode="hash",
        record=True,
        max_real=1200.0,
        max_virtual=360000.0,
    ):
        self._sources = copy.deepcopy(sources)
        for s in self._sources:
            s["cleared"] = False
        self.seed, self.error_mode, self.record = int(seed), error_mode, record
        self.time, self.n_requests = 0.0, 0
        self.robot_pos, self.robot_channel = np.zeros(2), 1
        self.entered, self.exited = False, False
        self.max_real, self.max_virtual = float(max_real), float(max_virtual)
        self.remaining_real, self.real_latency = self.max_real, 0.0
        self._started = None
        self._finished = None
        self.log = []
        self.metrics = dict(
            movement_s=0.0,
            switches=0,
            measurements=0,
            no_signal=0,
            clear_success=0,
            clear_failure=0,
        )

    @property
    def real_time(self):
        if self._started is None:
            return 0.0
        return (self._finished or time.perf_counter()) - self._started

    def enter(self, remaining_real=None):
        if not self.entered:
            self._started = time.perf_counter()
            self.entered = True
        if remaining_real is not None:
            self.remaining_real = min(self.max_real, max(0.0, float(remaining_real)))
        return dict(
            accepted=True,
            max_virtual_duration_s=self.max_virtual,
            max_real_duration_s=self.max_real,
            remaining_real_duration_s=self.remaining_real,
        )

    def alive(self):
        return (
            self.entered
            and not self.exited
            and self.time < self.max_virtual
            and self.real_time < self.remaining_real
        )

    def exit(self):
        self.exited = True
        self._finished = time.perf_counter()

    def can_execute(self, point, operation_s):
        return (
            self.alive()
            and self.time
            + np.linalg.norm(np.asarray(point) - self.robot_pos) / 5
            + operation_s
            <= self.max_virtual
        )

    def _error(self, pos):
        x, y = (round(float(v), 2) for v in pos)
        if self.error_mode == "smooth":
            return math.sin(x / 170 + self.seed * 0.11) * math.cos(y / 230)
        if self.error_mode == "extreme":
            return 1.0 if math.sin(x / 210 + y / 130 + self.seed) >= 0 else -1.0
        if self.error_mode != "hash":
            raise ValueError(self.error_mode)
        key = f"{self.seed}:{x:.2f}:{y:.2f}".encode()
        value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
        return 2 * value / (2**64 - 1) - 1

    def _source(self, channel):
        return next(
            (s for s in self._sources if s["channel"] == channel and not s["cleared"]),
            None,
        )

    def _travel(self, point):
        movement = float(np.linalg.norm(np.asarray(point) - self.robot_pos) / 5)
        self.time += movement
        self.metrics["movement_s"] += movement
        self.robot_pos = np.asarray(point, dtype=float).copy()
        self.n_requests += 1

    def measure(self, point, channel):
        switch = int(channel != self.robot_channel)
        if not self.can_execute(point, 5 + switch):
            return dict(type="closed", svd_deg=None)
        self._travel(point)
        self.time += 5 + switch
        self.robot_channel = int(channel)
        self.metrics["switches"] += switch
        self.metrics["measurements"] += 1
        source = self._source(channel)
        result = dict(type="no_signal", svd_deg=None)
        if source is not None:
            delta = np.asarray(source["pos"]) - self.robot_pos
            distance = float(np.linalg.norm(delta))
            visible = distance <= source["reception"]
            if source["kind"] == "dir":
                heading = math.radians(source["dir"])
                visible = (
                    visible
                    and float(-delta @ np.array([math.cos(heading), math.sin(heading)]))
                    >= 0
                )
            if visible:
                if distance <= 5:
                    result["type"] = "near"
                else:
                    theta = math.degrees(math.atan2(delta[1], delta[0]))
                    result = dict(
                        type="direction",
                        svd_deg=round((theta + self._error(point)) % 360, 2) % 360,
                    )
        self.metrics["no_signal"] += int(result["type"] == "no_signal")
        if self.record:
            self.log.append(
                dict(
                    action="measure",
                    point=self.robot_pos.tolist(),
                    channel=channel,
                    result=result,
                    virtual_s=self.time,
                )
            )
        return result

    def clear(self, point, channel):
        # Reserve the maximum clear-operation cost before issuing the request.
        if not self.can_execute(point, 5):
            return False
        self._travel(point)
        source = self._source(channel)
        success = (
            source is not None
            and np.linalg.norm(self.robot_pos - np.asarray(source["pos"])) <= 20
        )
        self.time += 5 if success else 3
        self.metrics["clear_success" if success else "clear_failure"] += 1
        if success:
            source["cleared"] = True
        if self.record:
            self.log.append(
                dict(
                    action="clear",
                    point=self.robot_pos.tolist(),
                    channel=channel,
                    result=bool(success),
                    virtual_s=self.time,
                )
            )
        return bool(success)

    def outcome(self):
        """Evaluator-only truth. The policy/environment never calls this method."""
        count = sum(s["cleared"] for s in self._sources)
        return dict(
            n_sources=len(self._sources),
            n_cleared=count,
            all_clear=count == len(self._sources),
            virtual_s=self.time,
            wall_s=self.real_time,
            requests=self.n_requests,
            **self.metrics,
        )


#     def truth_for_plot(self):
#         return copy.deepcopy(self._sources)
