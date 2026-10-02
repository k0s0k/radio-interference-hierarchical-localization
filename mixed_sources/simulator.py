"""Mixed-source Q4 scenarios and the shared offline task simulator."""

import math
import random
import numpy as np
from omnidirectional.simulator import TaskSimulator as BaseSimulator


def make_scenario(seed, distribution="uniform"):
    rng = random.Random(int(seed))
    n = rng.randint(10, 16)
    channels = rng.sample(range(1, 21), n)
    n_dir = rng.randint(1, n - 1)
    if distribution in ("boundary_outward", "cluster_away", "high_directional"):
        n_dir = n - 1
    dir_indices = set(rng.sample(range(n), n_dir))
    sources = []
    for i, ch in enumerate(channels):
        angle = rng.uniform(0, 2 * math.pi)
        radius = 1800 * math.sqrt(rng.random())
        heading = rng.uniform(0, 360)
        reception = rng.uniform(1000, 1500)
        if distribution == "boundary_outward":
            radius, heading, reception = (
                rng.uniform(1770, 1800),
                math.degrees(angle),
                1000.0,
            )
        elif distribution == "cluster_away":
            angle, radius, heading, reception = (
                rng.uniform(-0.08, 0.08),
                rng.uniform(1400, 1600),
                0.0,
                1000.0,
            )
        elif distribution == "collinear":
            angle, radius = (0 if i % 2 else math.pi), rng.uniform(100, 1799)
            heading, reception = rng.choice((90.0, 270.0)), 1000.0
        elif distribution == "high_directional":
            reception = 1000.0
        elif distribution != "uniform":
            raise ValueError(distribution)
        item = dict(
            pos=(radius * math.cos(angle), radius * math.sin(angle)),
            channel=ch,
            kind="dir" if i in dir_indices else "omni",
            reception=reception,
            cleared=False,
        )
        if item["kind"] == "dir":
            item["dir"] = heading % 360
        sources.append(item)
    return sources


class TaskSimulator(BaseSimulator):
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
            visible = distance <= source["reception"] + 1e-9
            if source["kind"] == "dir":
                angle = math.radians(source["dir"])
                front = float(-delta @ np.array([math.cos(angle), math.sin(angle)]))
                visible = visible and front >= -1e-8  # closed 180-degree sector
            if visible:
                if distance <= 5:
                    result["type"] = "near"
                else:
                    bearing = math.degrees(math.atan2(delta[1], delta[0]))
                    result = dict(
                        type="direction",
                        svd_deg=round((bearing + self._error(point)) % 360, 2) % 360,
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

    def outcome(self):
        result = super().outcome()
        result["n_directional"] = sum(s["kind"] == "dir" for s in self._sources)
        result["avg_localization_clear_s"] = (
            self.time / result["n_cleared"] if result["n_cleared"] else None
        )
        return result
