"""Heading-independent discovery cover using intersecting lattice triangles."""

import math
import numpy as np
from omnidirectional.geometry import contains


def nearest_on_segment(point, a, b):
    v = b - a
    t = np.clip(float((point - a) @ v) / max(float(v @ v), 1e-15), 0, 1)
    return a + t * v


def distance_to_triangle(point, triangle):
    if contains(triangle, point):
        return 0.0
    return min(
        float(np.linalg.norm(point - nearest_on_segment(point, a, b)))
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0))
    )


def directional_cover(spacing=650.0, radius=1800.0):
    """Keep all vertices of every equilateral triangle meeting the source disk.

    For any source s in a retained triangle and any heading h, at least one
    vertex v satisfies (v-s).h >= 0 because s is a convex combination of the
    three vertices. Every vertex is at distance <= spacing <= R_min from s.
    Therefore scanning these vertices discovers any 180-degree directional
    source, including sources on the region boundary and headings pointing out.
    This proves discovery, not a bound on the later localization time.
    """
    if not 0 < spacing <= 999.0:
        raise ValueError("Heading-independent cover requires spacing < 1000 m")

    def point(key):
        i, j = key
        return np.array([spacing * (i + j / 2), spacing * math.sqrt(3) * j / 2])

    bound = math.ceil(radius / spacing) + 3
    keys, triangles = set(), []
    for i in range(-bound, bound + 1):
        for j in range(-bound, bound + 1):
            for vertices in (
                ((i, j), (i + 1, j), (i, j + 1)),
                ((i + 1, j), (i + 1, j + 1), (i, j + 1)),
            ):
                tri = np.array([point(key) for key in vertices])
                if distance_to_triangle(np.zeros(2), tri) <= radius + 1e-6:
                    triangles.append(vertices)
                    keys.update(vertices)
    ordered = sorted(keys)
    index = {key: i for i, key in enumerate(ordered)}
    return np.array([point(key) for key in ordered]), np.array(
        [[index[key] for key in vertices] for vertices in triangles]
    )


def convex_hull(points):
    points = sorted(set(tuple(map(float, p)) for p in points))
    if len(points) <= 1:
        return np.asarray(points).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def project_to_positive_hull(point, sites):
    hull = convex_hull(sites)
    if not len(hull):
        return None
    if len(hull) == 1:
        return hull[0].copy()
    if len(hull) >= 3 and contains(hull, point):
        return point.copy()
    candidates = [
        nearest_on_segment(point, a, b) for a, b in zip(hull, np.roll(hull, -1, axis=0))
    ]
    projected = min(candidates, key=lambda p: np.linalg.norm(p - point))
    # Stay in the convex hull despite roundoff at a polygon edge.
    return projected * (1 - 1e-9) + hull.mean(0) * 1e-9
