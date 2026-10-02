"""Conservative position set plus an approximate reception hypothesis cloud.

Only positive readings shrink the certified position polygon. Sampled position,
heading/type and feasible radius intervals rank probes; they NEVER certify
clearance or eliminate an unseen channel. Empty sampled support is not a proof.
"""

import numpy as np
from omnidirectional.environment import ChannelBelief as PositionBelief


class ChannelBelief(PositionBelief):
    def __init__(self):
        super().__init__()
        self.cloud = None
        self.misses_since_positive = 0
        self.refresh_count = 0

    def update(self, point, result):
        super().update(point, result)
        self.cloud = None
        if result["type"] == "no_signal":
            self.misses_since_positive += 1
        elif result["type"] in ("near", "direction"):
            self.misses_since_positive = 0

    def _refresh(self):
        if self.cloud is not None:
            return
        self.refresh_count += 1
        center = self.polygon.mean(0)
        positions = np.concatenate(
            ([center], self.polygon, (self.polygon + center) / 2)
        )
        if len(positions) > 25:
            positions = positions[np.linspace(0, len(positions) - 1, 25).astype(int)]
        angles = np.arange(24) * np.pi / 12
        headings = np.column_stack((np.cos(angles), np.sin(angles)))
        # Heading index 24 denotes an omnidirectional hypothesis.
        headings = np.concatenate((headings, [[0.0, 0.0]]))
        sites = np.asarray([p for p, _ in self.positive])
        rel = sites[None, :, :] - positions[:, None, :]
        distances = np.linalg.norm(rel, axis=-1)
        lower = np.maximum(1000.0, distances.max(1))[:, None] * np.ones((1, 25))
        positive_front = (np.einsum("qpd,hd->qph", rel, headings) >= -1e-7).all(1)
        upper = np.full_like(lower, 1500.0)
        if self.negative:
            neg = np.asarray(self.negative)[None, :, :] - positions[:, None, :]
            nd = np.linalg.norm(neg, axis=-1)
            nf = np.einsum("qnd,hd->qnh", neg, headings) >= -1e-7
            # For a negative reading in front, the radius must be below its
            # distance. Behind-source negatives impose no radius restriction.
            upper = np.minimum(
                upper, np.where(nf, nd[:, :, None] - 1e-7, 1500.0).min(1)
            )
        valid = positive_front & (lower <= upper)
        qi, hi = np.nonzero(valid)
        if not len(qi):
            self.cloud = dict(empty=True, p_directional=0.5, direction=np.zeros(2))
            return
        weights = np.where(hi == 24, 0.5, 0.5 / 24)
        weights /= weights.sum()
        directional = hi != 24
        p_directional = float(weights[directional].sum())
        direction = (weights[:, None] * headings[hi]).sum(0) / max(p_directional, 1e-12)
        self.cloud = dict(
            empty=False,
            positions=positions[qi],
            headings=headings[hi],
            directional=directional,
            lower=lower[qi, hi],
            upper=upper[qi, hi],
            weights=weights,
            p_directional=p_directional,
            direction=direction,
        )

    def reception_features(self, points):
        self._refresh()
        cloud = self.cloud
        if cloud["empty"]:
            return np.full(len(points), 0.5), 0.5, np.zeros(2), True
        delta = np.asarray(points)[:, None, :] - cloud["positions"][None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        front = np.einsum("mqd,qd->mq", delta, cloud["headings"]) >= -1e-7
        probability = np.clip(
            (cloud["upper"][None, :] - distance)
            / np.maximum(cloud["upper"] - cloud["lower"], 1e-7)[None, :],
            0,
            1,
        )
        probability[distance <= cloud["lower"][None, :]] = 1.0
        visibility = (front * probability * cloud["weights"][None, :]).sum(1)
        return visibility, cloud["p_directional"], cloud["direction"], False
