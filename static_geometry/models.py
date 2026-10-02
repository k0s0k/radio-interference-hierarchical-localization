"""问题一、二的可复现计算；几何精确式与面积近似式分开输出。"""

from pathlib import Path
import json
import math
import numpy as np
from scipy.optimize import minimize
from omnidirectional.geometry import (
    outer_disk,
    halfplane,
    area,
    enclosing_circle,
    contains,
)


def wedge(poly, site, bearing, epsilon=1.0):
    site = np.asarray(site)
    lo, hi = np.radians([bearing - epsilon, bearing + epsilon])
    for n in (np.array([-np.sin(lo), np.cos(lo)]), np.array([np.sin(hi), -np.cos(hi)])):
        poly = halfplane(poly, n, float(n @ site))
    return poly


def diameter_calipers(poly):
    """凸多边形旋转卡壳；与两两枚举作独立校核。"""
    n = len(poly)
    if n < 2:
        return 0.0
    if n == 2:
        return float(np.linalg.norm(poly[0] - poly[1]))
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    j = 1
    best = 0.0
    for i in range(n):
        k = (i + 1) % n
        edge = poly[k] - poly[i]
        for _ in range(n):
            j2 = (j + 1) % n
            if (
                abs(cross(edge, poly[j2] - poly[i]))
                > abs(cross(edge, poly[j] - poly[i])) + 1e-9
            ):
                j = j2
            else:
                break
        for a in (i, k):
            for b in (j, (j + 1) % n):
                best = max(best, float(np.linalg.norm(poly[a] - poly[b])))
    return best


def q1_example(epsilon=1.0):
    source = np.array([320.0, 540.0])
    sites = np.array([[0.0, 0.0], [1000.0, 0.0], [520.0, 820.0]])
    bearings = np.degrees(
        np.arctan2((source - sites)[:, 1], (source - sites)[:, 0])
    ) + np.array([0.7, -0.9, 0.4])
    p = outer_disk(sides=720)
    histories = []
    for s, b in zip(sites, bearings):
        p = wedge(p, s, b, epsilon)
        c, r = enclosing_circle(p)
        histories.append(
            dict(polygon=p.tolist(), area=area(p), center=c.tolist(), radius=r)
        )
    diameter = diameter_calipers(p)
    brute = float(np.linalg.norm(p[:, None] - p[None, :], axis=2).max())
    assert abs(diameter - brute) < 1e-7 and contains(p, source)
    assert max(np.linalg.norm(p - c, axis=1)) <= r
    return dict(
        source=source.tolist(),
        sites=sites.tolist(),
        bearings=bearings.tolist(),
        epsilon=epsilon,
        histories=histories,
        area=area(p),
        diameter=diameter,
        diameter_brute=brute,
        center=c.tolist(),
        radius=r,
    )


def worst_area(x, y, epsilon=1.0):
    """一阶面积近似的连续距离最坏值，检查端点及全部驻点。"""
    x, y = np.asarray(x), np.asarray(y)

    def f(t):
        return t * (x * x + (y - t) ** 2) / np.maximum(abs(x), 1e-9)

    vals = [f(5.0), f(1500.0)]
    disc = y * y - 3 * x * x
    for sign in [-1, 1]:
        t = (2 * y + sign * np.sqrt(np.maximum(disc, 0))) / 3
        vals.append(np.where((disc >= 0) & (t >= 5) & (t <= 1500), f(t), -np.inf))
    return 4 * math.radians(epsilon) ** 2 * np.maximum.reduce(vals)


def q2_candidate():
    # 使用网格初始化与 Nelder–Mead 局部搜索，并校核连续距离最坏值。
    tgrid = np.linspace(5, 1500, 400)
    x, y = np.meshgrid(np.linspace(5, 1000, 160), np.linspace(5, 1500, 220))
    feasible = (np.hypot(x, y - 5) <= 1000) & (np.hypot(x, y - 1500) <= 1000)
    objective = np.zeros_like(x)
    for t in tgrid:
        objective = np.maximum(objective, t * (x * x + (y - t) ** 2) / x)
    objective[~feasible] = np.inf
    idx = np.unravel_index(np.argmin(objective), objective.shape)

    def cost(xy):
        a, b = xy
        if a <= 1e-6 or max(math.hypot(a, b - 5), math.hypot(a, b - 1500)) > 1000:
            return 1e10
        return float(np.max(tgrid * (a * a + (b - tgrid) ** 2) / a))

    res = minimize(
        cost,
        [x[idx], y[idx]],
        method="Nelder-Mead",
        options=dict(xatol=1e-6, fatol=1e-6, maxiter=20000),
    )
    a, b = res.x
    return dict(
        point=[float(a), float(b)],
        approx_worst_area=float(worst_area(a, b)),
        endpoint_ranges=[math.hypot(a, b - 5), math.hypot(a, b - 1500)],
        scope="名义射线距离区间的数值极小极大候选；面积为小角度一阶近似。",
    )


def main(output=None):
    d = dict(q1=q1_example(), q2=q2_candidate())
    output = Path(
        output or Path(__file__).resolve().parents[1] / "outputs" / "static_geometry.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: {a: b for a, b in v.items() if a != "histories"} for k, v in d.items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return d


if __name__ == "__main__":
    main()
