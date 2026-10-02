"""Mixed-source macro decisions with certified clearance and sampled visibility."""

from dataclasses import dataclass
import math
import numpy as np

from omnidirectional.environment import MacroAction as BaseAction
from .belief import ChannelBelief
from .coverage import directional_cover, project_to_positive_hull
from .simulator import TaskSimulator, make_scenario

GLOBAL_DIM, ACTION_DIM, MAX_ACTIONS = 10, 29, 96


@dataclass
class MacroAction(BaseAction):
    visibility: float = 1.0
    p_directional: float = 0.5
    heading_x: float = 0.0
    heading_y: float = 0.0
    in_positive_hull: bool = False
    empty_hypotheses: bool = False
    consecutive_misses: int = 0


def heuristic_index(actions):
    def score(a):
        if a.kind == "clear":
            return 0.65 * a.cost
        if a.kind == "refine":
            return 0.85 * a.cost / max(0.15, a.visibility)
        return a.cost / (1 + 2 * a.coverage_gain)

    return min(range(len(actions)), key=lambda i: score(actions[i]))


class SearchEnvironment:
    def __init__(self, spacing=950.0, max_steps=360, record=False, use_visibility=True):
        self.stations, self.triangles = directional_cover(spacing)
        self.max_steps, self.record, self.use_visibility = (
            max_steps,
            record,
            use_visibility,
        )
        grid = np.array(
            [
                (x, y)
                for x in range(-1800, 1801, 300)
                for y in range(-1800, 1801, 300)
                if x * x + y * y <= 1800**2
            ]
        )
        headings = np.column_stack(
            (np.cos(np.arange(12) * np.pi / 6), np.sin(np.arange(12) * np.pi / 6))
        )
        delta = self.stations[:, None, :] - grid[None, :, :]
        self.sample_cover = (np.linalg.norm(delta, axis=-1) <= 1000)[:, :, None] & (
            np.einsum("spd,hd->sph", delta, headings) >= -1e-8
        )

    def reset(self, seed=0, sources=None, distribution="uniform", error_mode="hash"):
        self.sim = TaskSimulator(
            make_scenario(seed, distribution) if sources is None else sources,
            seed=seed,
            error_mode=error_mode,
            record=self.record,
        )
        self.sim.enter()
        self.beliefs = {ch: ChannelBelief() for ch in range(1, 21)}
        self.visited = np.zeros(len(self.stations), dtype=bool)
        self.sample_covered = np.zeros(self.sample_cover.shape[1:], dtype=bool)
        self.steps, self.fallback_steps = 0, 0
        self.done, self.success_certificate = False, False
        self.failure_reason = ""
        self.macro_log = []
        return self.observation()

    def unknown_channels(self):
        return [ch for ch, b in self.beliefs.items() if not b.seen and not b.cleared]

    def scan_channels(self):
        return [
            ch
            for ch, b in self.beliefs.items()
            if not b.cleared and (not b.seen or b.radius > 20 - 1e-6)
        ]

    def _probe_points(self, b):
        first = b.positive[0][1]
        theta = math.radians(first if first is not None else 0)
        axis = np.array([math.cos(theta), math.sin(theta)])
        normal = np.array([-axis[1], axis[0]])
        offset = min(200.0, max(30.0, 0.3 * b.radius))
        points = [b.center, b.center - offset * normal, b.center + offset * normal]
        variants = [0.0, -1.0, 1.0]
        safe = project_to_positive_hull(b.center, [p for p, _ in b.positive])
        if safe is not None:
            points.append(safe)
            variants.append(0.0)
        ring = min(650.0, max(70.0, 2 * b.radius + 20))
        for angle in np.arange(8) * math.pi / 4:
            points.append(
                b.center + ring * np.array([math.cos(angle), math.sin(angle)])
            )
            variants.append(0.0)
        # If opportunistic probes missed, add a finite local lattice around the
        # entire robust polygon. It changes only after positive set updates.
        if b.misses_since_positive >= 3 or self.visited.all():
            x, y = b.polygon @ axis, b.polygon @ normal
            for u in np.arange(
                math.floor((x.min() - 300) / 200) * 200, x.max() + 500, 200
            ):
                for v in np.arange(
                    math.floor((y.min() - 300) / 200) * 200, y.max() + 500, 200
                ):
                    points.append(u * axis + v * normal)
                    variants.append(0.0)
        unique, seen_keys = [], set()
        for point, variant in zip(points, variants):
            key = tuple(round(float(v), 2) for v in point)
            if key in b.measured or key in seen_keys:
                continue
            seen_keys.add(key)
            guaranteed = safe is not None and np.linalg.norm(point - safe) < 1e-6
            unique.append((point, variant, guaranteed))
        return unique

    def _candidates(self):
        actions = []
        current = self.sim.robot_pos
        channels = self.scan_channels()
        # The nearest 24 remaining stations are offered; visited ones leave the
        # list, so pruning does not remove stations from the discovery cover.
        station_ids = sorted(
            np.flatnonzero(~self.visited),
            key=lambda i: np.linalg.norm(self.stations[i] - current),
        )[:24]
        if channels:
            for idx in station_ids:
                point = self.stations[idx]
                cost = float(np.linalg.norm(point - current) / 5 + 6 * len(channels))
                gain = float(np.mean(self.sample_cover[idx] & ~self.sample_covered))
                if self.sim.can_execute(point, 6 * len(channels)):
                    actions.append(
                        MacroAction(
                            "scan",
                            point,
                            station=int(idx),
                            cost=cost,
                            coverage_gain=gain,
                        )
                    )
        for ch, b in self.beliefs.items():
            if not b.seen or b.cleared:
                continue
            if b.radius <= 20.0 - 1e-6:
                if self.sim.can_execute(b.center, 5):
                    actions.append(
                        MacroAction(
                            "clear",
                            b.center,
                            ch,
                            cost=float(np.linalg.norm(b.center - current) / 5 + 5),
                            radius=b.radius,
                            region_area=b.area,
                            n_positive=len(b.positive),
                            n_negative=len(b.negative),
                        )
                    )
                continue
            candidates = self._probe_points(b)
            if not candidates:
                continue
            probabilities, pdir, heading, empty = b.reception_features(
                [p for p, _, _ in candidates]
            )
            if not self.use_visibility:
                probabilities = np.full(len(candidates), 0.5)
                pdir = 0.5
                heading = np.zeros(2)
                empty = False
            local = []
            for (point, variant, guaranteed), probability in zip(
                candidates, probabilities
            ):
                if guaranteed:
                    probability = 1.0
                extra = 5 + int(ch != self.sim.robot_channel)
                if not self.sim.can_execute(point, extra):
                    continue
                local.append(
                    MacroAction(
                        "refine",
                        point,
                        ch,
                        cost=float(np.linalg.norm(point - current) / 5 + extra),
                        radius=b.radius,
                        region_area=b.area,
                        n_positive=len(b.positive),
                        n_negative=len(b.negative),
                        variant=variant,
                        visibility=float(probability),
                        p_directional=float(pdir),
                        heading_x=float(heading[0]),
                        heading_y=float(heading[1]),
                        in_positive_hull=guaranteed,
                        empty_hypotheses=empty,
                        consecutive_misses=b.misses_since_positive,
                    )
                )
            # Shared candidate budget. Probability is a ranking feature, not a
            # hard visibility mask: low-probability probes remain possible.
            local.sort(key=lambda a: a.cost / max(0.15, a.visibility))
            actions.extend(local[:4])
        if len(actions) > MAX_ACTIONS:
            raise RuntimeError("Candidate budget overflow")
        if self.steps >= 240 and actions:
            actions = [actions[heuristic_index(actions)]]
        return actions

    def observation(self):
        self.actions = [] if self.done else self._candidates()
        if not self.actions and not self.done:
            self.done = True
            self.failure_reason = "no_legal_candidate"
            self.sim.exit()
        seen = sum(b.seen for b in self.beliefs.values())
        cleared = sum(b.cleared for b in self.beliefs.values())
        global_features = np.array(
            [
                *list(self.sim.robot_pos / 1800),
                self.sim.robot_channel / 20,
                self.sim.time / 20000,
                float(self.visited.mean()),
                seen / 20,
                cleared / 20,
                len(self.unknown_channels()) / 20,
                self.sim.metrics["measurements"] / 500,
                self.steps / self.max_steps,
            ],
            dtype=np.float32,
        )
        features = np.zeros((MAX_ACTIONS, ACTION_DIM), dtype=np.float32)
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        for i, a in enumerate(self.actions):
            rel = a.point - self.sim.robot_pos
            row = [float(a.kind == kind) for kind in ("scan", "refine", "clear")]
            row += [
                *list(a.point / 1800),
                *list(rel / 1800),
                np.linalg.norm(rel) / 1800,
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
            row += [
                a.visibility,
                a.p_directional,
                a.heading_x,
                a.heading_y,
                math.hypot(a.heading_x, a.heading_y),
                float(a.in_positive_hull),
                float(a.empty_hypotheses),
                a.consecutive_misses / 10,
            ]
            features[i] = row
            mask[i] = True
        return dict(global_features=global_features, candidates=features, mask=mask)

    def step(self, index):
        if self.done or not 0 <= int(index) < len(self.actions):
            raise ValueError("Invalid or masked action")
        action = self.actions[int(index)]
        before = self.sim.time
        self.fallback_steps += int(self.steps >= 240)
        if action.kind == "scan":
            complete = True
            for ch in self.scan_channels():
                result = self.sim.measure(action.point, ch)
                if result["type"] == "closed":
                    complete = False
                    break
                self.beliefs[ch].update(action.point, result)
            if complete:
                self.visited[action.station] = True
                self.sample_covered |= self.sample_cover[action.station]
        elif action.kind == "refine":
            self.beliefs[action.channel].update(
                action.point, self.sim.measure(action.point, action.channel)
            )
        elif self.sim.clear(action.point, action.channel):
            self.beliefs[action.channel].cleared = True
        else:
            self.failure_reason = "certified_clear_failed"
        self.steps += 1
        count = sum(b.cleared for b in self.beliefs.values())
        pending = any(b.seen and not b.cleared for b in self.beliefs.values())
        # Uses only public maximum source count and the certified discovery cover.
        self.success_certificate = bool(
            (self.visited.all() and not pending) or count == 16
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
                    visibility=action.visibility,
                    p_directional=action.p_directional,
                    virtual_s=self.sim.time,
                )
            )
        if self.done:
            self.sim.exit()
        observation = self.observation()
        reward = -(self.sim.time - before) / 1000
        if self.done and not self.success_certificate:
            reward -= 100
        return (
            observation,
            float(reward),
            self.done,
            dict(
                certificate=self.success_certificate, failure_reason=self.failure_reason
            ),
        )

    def summary(self):
        return dict(
            steps=self.steps,
            fallback_steps=self.fallback_steps,
            certificate=self.success_certificate,
            failure_reason=self.failure_reason,
        )
