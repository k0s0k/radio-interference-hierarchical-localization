"""Conservative convex geometry. Distances in metres; angles in degrees."""

from __future__ import annotations

import math
import numpy as np

ANGLE_BOUND = 1.00500001  # bounded error plus 0.01-degree output rounding


def outer_disk(center=(0.0, 0.0), radius=1800.0, sides=64):
    """Circumscribed regular polygon, so the true disk is never cut away."""
    angles = (np.arange(sides) + 0.5) * 2 * np.pi / sides
    return np.asarray(center) + radius / math.cos(math.pi / sides) * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )


def halfplane(poly, normal, offset):
    """Clip a CCW convex polygon by dot(normal, x) >= offset."""
    if len(poly) == 0:
        return np.empty((0, 2))
    values = poly @ np.asarray(normal) - offset
    inside = values >= -1e-8
    if inside.all():
        return poly
    if not inside.any():
        return np.empty((0, 2))
    out = []
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        if inside[i]:
            out.append(poly[i])
        if inside[i] != inside[j]:
            t = values[i] / (values[i] - values[j])
            out.append(poly[i] + t * (poly[j] - poly[i]))
    return np.asarray(out)


def intersect_outer_disk(poly, center, radius, sides=32):
    center = np.asarray(center)
    for angle in np.arange(sides) * 2 * np.pi / sides:
        inward = -np.array([math.cos(angle), math.sin(angle)])
        poly = halfplane(poly, inward, float(inward @ center) - radius)
        if len(poly) == 0:
            break
    return poly


def observe_direction(poly, point, bearing):
    point = np.asarray(point)
    lo, hi = np.deg2rad([bearing - ANGLE_BOUND, bearing + ANGLE_BOUND])
    for normal in (
        np.array([-math.sin(lo), math.cos(lo)]),
        np.array([math.sin(hi), -math.cos(hi)]),
    ):
        poly = halfplane(poly, normal, float(normal @ point))
    # A positive reading also implies distance <= maximum effective range.
    return intersect_outer_disk(poly, point, 1500.0)


def area(poly):
    if len(poly) < 3:
        return 0.0
    return float(
        abs(
            np.sum(
                poly[:, 0] * np.roll(poly[:, 1], -1)
                - poly[:, 1] * np.roll(poly[:, 0], -1)
            )
        )
        / 2
    )


def _three_circle(a, b, c):
    ab, ac = b - a, c - a
    determinant = 2 * (ab[0] * ac[1] - ab[1] * ac[0])
    if abs(determinant) < 1e-9:
        points = [a, b, c]
        pair = max(
            ((points[i], points[j]) for i in range(3) for j in range(i)),
            key=lambda p: np.sum((p[0] - p[1]) ** 2),
        )
        center = (pair[0] + pair[1]) / 2
    else:
        u, v = ab @ ab, ac @ ac
        center = (
            a + np.array([ac[1] * u - ab[1] * v, ab[0] * v - ac[0] * u]) / determinant
        )
    return center, float(max(np.linalg.norm(p - center) for p in (a, b, c)))


def enclosing_circle(poly):
    """Incremental enclosing circle; final radius verifies *every* vertex.

    Final radius recomputation preserves the enclosure even in degenerate cases.
    No random state shared with the scenario or the learning policy is used.
    """
    if len(poly) == 0:
        raise ValueError("Empty feasible polygon: inconsistent observations")
    points = np.asarray(poly)[np.random.default_rng(17).permutation(len(poly))]
    center, radius = points[0].copy(), 0.0
    for i, p in enumerate(points):
        if np.linalg.norm(p - center) <= radius + 1e-8:
            continue
        center, radius = p.copy(), 0.0
        for j in range(i):
            q = points[j]
            if np.linalg.norm(q - center) <= radius + 1e-8:
                continue
            center, radius = (p + q) / 2, float(np.linalg.norm(p - q) / 2)
            for k in range(j):
                if np.linalg.norm(points[k] - center) > radius + 1e-8:
                    center, radius = _three_circle(p, q, points[k])
    radius = float(np.max(np.linalg.norm(np.asarray(poly) - center, axis=1))) + 1e-7
    return center, radius


def contains(poly, point, tolerance=1e-5):
    if len(poly) < 3:
        return False
    edges = np.roll(poly, -1, axis=0) - poly
    rel = np.asarray(point) - poly
    return bool(np.all(edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0] >= -tolerance))


def cover_grid(cover=650.0, radius=1800.0):
    """Triangular lattice covering the source disk, including boundary stations."""
    spacing = math.sqrt(3) * cover
    bound = math.ceil((radius + cover) / spacing) + 1
    points = []
    for i in range(-bound, bound + 1):
        for j in range(-bound, bound + 1):
            x, y = spacing * (i + j / 2), spacing * math.sqrt(3) * j / 2
            if math.hypot(x, y) <= radius + cover + 1e-8:
                points.append((x, y))
    return np.asarray(points)
