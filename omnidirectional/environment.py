"""Q3 macro-action environment. All decisions use observation history only."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from .geometry import (
    outer_disk,
    intersect_outer_disk,
    observe_direction,
    area,
    enclosing_circle,
    cover_grid,
)
from .simulator import TaskSimulator, make_scenario

GLOBAL_DIM, ACTION_DIM, MAX_ACTIONS = 10, 21, 96


@dataclass
class ChannelBelief:
    polygon: np.ndarray = field(default_factory=outer_disk)
    seen: bool = False
    cleared: bool = False
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 1803.0
    area: float = math.pi * 1800**2
    positive: list = field(default_factory=list)
    negative: list = field(default_factory=list)
    measured: set = field(default_factory=set)

    def update(self, point, result):
        self.measured.add(tuple(round(float(v), 2) for v in point))
        if result["type"] == "no_signal":
            self.negative.append(tuple(point))
            return
        if result["type"] not in ("direction", "near"):
            return
        self.seen = True
        if result["type"] == "near":
            self.polygon = intersect_outer_disk(self.polygon, point, 5.0)
        else:
            self.polygon = observe_direction(self.polygon, point, result["svd_deg"])
        if not len(self.polygon):
            raise RuntimeError(
                "Feasible set became empty; do not silently clear at an estimate"
            )
        self.positive.append((tuple(point), result["svd_deg"]))
        self.center, self.radius = enclosing_circle(self.polygon)
        self.area = area(self.polygon)


@dataclass
class MacroAction:
    kind: str
    point: np.ndarray
    channel: int = 0
    station: int = -1
    cost: float = 0.0
    radius: float = 0.0
    region_area: float = 0.0
    n_positive: int = 0
    n_negative: int = 0
    coverage_gain: float = 0.0
    variant: float = 0.0


def heuristic_index(actions):
    """Dynamic cost-based baseline; no source truth, learned weights or lookahead."""
    factors = {"scan": 1.0, "refine": 0.85, "clear": 0.7}
    return min(
        range(len(actions)), key=lambda i: actions[i].cost * factors[actions[i].kind]
    )


class SearchEnvironment:
    def __init__(self, cover=650.0, max_steps=160, record=False):
        if not 0 < cover <= 650:
            raise ValueError("This Q3 environment keeps covering radius in (0, 650]")
        self.stations = cover_grid(cover)
        if len(self.stations) + 60 > MAX_ACTIONS:
            raise ValueError("Increase MAX_ACTIONS before using a denser cover grid")
        self.max_steps, self.record = int(max_steps), record
        grid = np.array(
            [
                (x, y)
                for x in np.arange(-1800, 1801, 200)
                for y in np.arange(-1800, 1801, 200)
                if x * x + y * y <= 1800**2
            ]
        )
        # Approximation used ONLY as an actor feature, never as a stopping proof.
        self.cover_samples = grid
        self.station_sample_cover = (
            np.linalg.norm(self.stations[:, None, :] - grid[None, :, :], axis=2) <= 1000
        )

    def reset(self, seed=0, sources=None, distribution="uniform", error_mode="hash"):
        self.sim = TaskSimulator(
            make_scenario(seed, distribution) if sources is None else sources,
            seed=seed,
            error_mode=error_mode,
            record=self.record,
        )
        # 第三问的发现覆盖依据仅适用于全向源。
        if sources is not None and any(s["kind"] != "omni" for s in sources):
            raise ValueError("Q3 environment accepts omnidirectional sources only")
        self.sim.enter()
        self.beliefs = {ch: ChannelBelief() for ch in range(1, 21)}
        self.visited = np.zeros(len(self.stations), dtype=bool)
        self.sample_covered = np.zeros(len(self.cover_samples), dtype=bool)
        self.steps, self.fallback_steps = 0, 0
        self.done, self.success_certificate = False, False
        self.failure_reason = ""
        self.macro_log = []
        return self.observation()

    def unknown_channels(self):
        return [ch for ch, b in self.beliefs.items() if not b.seen and not b.cleared]

    def _candidates(self):
        actions, current = [], self.sim.robot_pos
        unknown = self.unknown_channels()
        if unknown:
            for idx in np.flatnonzero(~self.visited):
                point = self.stations[idx]
                cost = float(np.linalg.norm(point - current) / 5 + 6 * len(unknown))
                gain = float(
                    np.mean(self.station_sample_cover[idx] & ~self.sample_covered)
                )
                if self.sim.can_execute(point, 6 * len(unknown)):
                    actions.append(
                        MacroAction(
                            "scan",
                            point,
                            station=int(idx),
                            cost=cost,
                            coverage_gain=gain,
                        )
                    )
        for ch, belief in self.beliefs.items():
            if not belief.seen or belief.cleared:
                continue
            if belief.radius <= 20.0 - 1e-6:
                point = belief.center
                if self.sim.can_execute(point, 5):
                    actions.append(
                        MacroAction(
                            "clear",
                            point,
                            ch,
                            cost=float(np.linalg.norm(point - current) / 5 + 5),
                            radius=belief.radius,
                            region_area=belief.area,
                            n_positive=len(belief.positive),
                            n_negative=len(belief.negative),
                        )
                    )
                continue
            first_pos = np.asarray(belief.positive[0][0])
            axis = belief.center - first_pos
            axis /= max(float(np.linalg.norm(axis)), 1e-9)
            perpendicular = np.array([-axis[1], axis[0]])
            offset = min(200.0, max(30.0, 0.3 * belief.radius))
            for variant in (0, -1, 1):
                point = belief.center + variant * offset * perpendicular
                key = tuple(round(float(v), 2) for v in point)
                if key in belief.measured:
                    continue
                # Q3: positive reception is guaranteed for every feasible source
                # if the chosen point is within the minimum range of every vertex.
                if np.max(np.linalg.norm(belief.polygon - point, axis=1)) > 999.99:
                    continue
                extra = 5 + int(ch != self.sim.robot_channel)
                if self.sim.can_execute(point, extra):
                    actions.append(
                        MacroAction(
                            "refine",
                            point,
                            ch,
                            cost=float(np.linalg.norm(point - current) / 5 + extra),
                            radius=belief.radius,
                            region_area=belief.area,
                            n_positive=len(belief.positive),
                            n_negative=len(belief.negative),
                            variant=variant,
                        )
                    )
        if len(actions) > MAX_ACTIONS:
            raise RuntimeError("Candidate padding too small")
        # Budgeted rescue is visible in the action mask, so stored PPO actions
        # always equal executed actions. No hidden action substitution occurs.
        if self.steps >= 100 and actions:
            actions = [actions[heuristic_index(actions)]]
        return actions

    def observation(self):
        self.actions = [] if self.done else self._candidates()
        if not self.actions and not self.done:
            self.done = True
            self.failure_reason = "no_legal_candidate"
            self.sim.exit()
        n_seen = sum(b.seen for b in self.beliefs.values())
        n_cleared = sum(b.cleared for b in self.beliefs.values())
        global_features = np.array(
            [
                *list(self.sim.robot_pos / 1800),
                self.sim.robot_channel / 20,
                self.sim.time / 20000,
                float(self.visited.mean()),
                n_seen / 20,
                n_cleared / 20,
                len(self.unknown_channels()) / 20,
                self.sim.metrics["measurements"] / 500,
                self.steps / self.max_steps,
            ],
            dtype=np.float32,
        )
        candidates = np.zeros((MAX_ACTIONS, ACTION_DIM), dtype=np.float32)
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        for i, a in enumerate(self.actions):
            relative = a.point - self.sim.robot_pos
            features = [float(a.kind == kind) for kind in ("scan", "refine", "clear")]
            features += [
                *list(a.point / 1800),
                *list(relative / 1800),
                np.linalg.norm(relative) / 1800,
                a.cost / 500,
                a.radius / 1500,
                math.log1p(a.region_area) / 16,
                a.n_positive / 6,
                a.channel / 20,
                float(a.channel == self.sim.robot_channel),
                a.coverage_gain,
                len(self.unknown_channels()) / 20,
                a.variant,
                float(a.kind == "clear") + 0.5 * float(a.kind == "refine"),
                a.n_negative / 30,
                float(self.visited.mean()),
                float(a.radius <= 20 and a.kind != "scan"),
            ]
            candidates[i] = features
            mask[i] = True
        return dict(global_features=global_features, candidates=candidates, mask=mask)

    def step(self, index):
        if self.done or not 0 <= int(index) < len(self.actions):
            raise ValueError("Invalid or masked action")
        action = self.actions[int(index)]
        before = self.sim.time
        if self.steps >= 100:
            self.fallback_steps += 1
        if action.kind == "scan":
            complete = True
            for ch in self.unknown_channels():
                result = self.sim.measure(action.point, ch)
                if result["type"] == "closed":
                    complete = False
                    break
                self.beliefs[ch].update(action.point, result)
            if complete:
                self.visited[action.station] = True
                self.sample_covered |= self.station_sample_cover[action.station]
        elif action.kind == "refine":
            result = self.sim.measure(action.point, action.channel)
            self.beliefs[action.channel].update(action.point, result)
        else:
            if self.sim.clear(action.point, action.channel):
                self.beliefs[action.channel].cleared = True
            else:
                self.failure_reason = "certified_clear_failed"
        self.steps += 1
        observed_cleared = sum(b.cleared for b in self.beliefs.values())
        pending_seen = any(b.seen and not b.cleared for b in self.beliefs.values())
        # Unknown source count is never queried. The two valid Q3 certificates:
        # exhaust the covering stations and clear all detected channels; or clear
        # the stated maximum of 16 distinct sources.
        self.success_certificate = bool(
            (self.visited.all() and not pending_seen) or observed_cleared == 16
        )
        self.done = self.success_certificate or bool(self.failure_reason)
        if not self.done and (self.steps >= self.max_steps or not self.sim.alive()):
            self.done = True
            self.failure_reason = "step_or_time_budget"
        if self.record:
            self.macro_log.append(
                dict(
                    step=self.steps,
                    kind=action.kind,
                    point=action.point.tolist(),
                    channel=action.channel,
                    radius=action.radius,
                    virtual_s=self.sim.time,
                )
            )
        if self.done:
            self.sim.exit()
        observation = self.observation()
        reward = -(self.sim.time - before) / 1000
        if self.done and not self.success_certificate:
            reward -= 100.0
        return (
            observation,
            float(reward),
            self.done,
            dict(
                certificate=self.success_certificate, failure_reason=self.failure_reason
            ),
        )

    def summary(self):
        """Public observations only; evaluator combines this with TaskSimulator.outcome."""
        return dict(
            steps=self.steps,
            fallback_steps=self.fallback_steps,
            certificate=self.success_certificate,
            failure_reason=self.failure_reason,
        )
