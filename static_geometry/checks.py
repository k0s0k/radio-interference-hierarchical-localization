"""第一二问的独立数值校核，结果随交付保存。"""

from pathlib import Path
import json
import numpy as np
from scipy.spatial import ConvexHull
from static_geometry.models import (
    diameter_calipers,
    worst_area,
    q1_example,
    q2_candidate,
)
from omnidirectional.geometry import enclosing_circle


def main():
    rng = np.random.default_rng(20260915)
    errors = []
    for _ in range(100):
        points = rng.normal(size=(100, 2)) * rng.uniform(1, 1800)
        poly = points[ConvexHull(points).vertices]
        brute = np.linalg.norm(poly[:, None] - poly[None, :], axis=2).max()
        errors.append(abs(diameter_calipers(poly) - brute))
        c, r = enclosing_circle(poly)
        assert np.max(np.linalg.norm(poly - c, axis=1)) <= r
    assert max(errors) < 1e-7
    q1 = q1_example()
    q2 = q2_candidate()
    x, y = q2["point"]
    t = np.linspace(5, 1500, 100001)
    dense = np.max(4 * np.radians(1) ** 2 * t * (x * x + (y - t) ** 2) / x)
    assert abs(float(worst_area(x, y)) - dense) < 1e-4
    assert max(q2["endpoint_ranges"]) <= 1000 + 1e-7
    report = dict(
        random_convex_polygons=100,
        max_diameter_error=max(errors),
        q1_area=q1["area"],
        q1_radius=q1["radius"],
        q2_approx_worst_area=q2["approx_worst_area"],
        q2_dense_grid_difference=abs(q2["approx_worst_area"] - dense),
        passed=True,
    )
    out = Path(__file__).resolve().parents[1] / "outputs" / "static_geometry_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
