"""
CUMCM 2026 Problem B - Q4 (mixed omnidirectional + directional sources).

Same validated integrated strategy as Q3, extended to directional sources:
  * detection is automatic: a directional source only returns a bearing from its
    front half-plane (heading +-90 deg), so the coverage sweep must be fine enough
    that every directional source's front sector contains at least one station;
  * localization uses every bearing the source emitted (omni from any station,
    dir only from front stations) in the same least-squares AOA fit;
  * clearing is distance-only in the protocol, so once localized we home in and
    clear regardless of which side of a directional source we end up on.

We also report the geometric detection ceiling: a source is 'detectable' iff at
least one grid station lies in its front half-plane within its own reception
radius.  Sources outside that ceiling (edge sources pointing radially outward)
cannot be found by any station-based sweep and are disclosed as a modelling limit.
"""

import math
import random
import argparse
import numpy as np

from . import core
from .core import (
    TaskSimulator,
    random_case,
    REGION_R,
    N_SRC_MIN,
    N_SRC_MAX,
    MAX_VIRTUAL_DURATION_S,
    MAX_REAL_DURATION_S,
)
from . import question_three

VIRTUAL_BUDGET = MAX_VIRTUAL_DURATION_S
REAL_BUDGET = MAX_REAL_DURATION_S
Q4_COVER = 350.0  # finer grid than Q3 (650 m) to cover directional front sectors


def solve(sim, cover=Q4_COVER, interleave=True, budget=VIRTUAL_BUDGET):
    return question_three.solve(
        sim, cover=cover, budget=budget, interleave=interleave, directional=True
    )


def detectable_fraction(sim, stations):
    """Fraction of sources with >=1 station in front half-plane within reception."""
    det = 0
    for s in sim.sources:
        for p in stations:
            if sim._in_coverage(s, p):
                det += 1
                break
    return det / max(1, len(sim.sources))


def run_mc(N=60, cover=Q4_COVER, seed=1, interleave=True):
    rng = random.Random(seed)
    stations = question_three.cover_grid(cover)
    times = []
    real_times = []
    cleared_frac = []
    avg_clear = []
    n_srcs = []
    nreqs = []
    det_frac = []
    for k in range(N):
        n = rng.randint(N_SRC_MIN, N_SRC_MAX)
        srcs = random_case(n, rng=rng, kinds=("omni", "dir"))
        sim = TaskSimulator(srcs, seed=1000 + k)
        solve(sim, cover=cover, interleave=interleave)
        nc = sim.clear_all()
        times.append(sim.time)
        real_times.append(sim.real_time)
        cleared_frac.append(nc / n)
        avg_clear.append(sim.time / nc if nc > 0 else float("nan"))
        n_srcs.append(n)
        nreqs.append(sim.n_requests)
        det_frac.append(detectable_fraction(sim, stations))
    return (
        np.array(times),
        np.array(real_times),
        np.array(cleared_frac),
        np.array(avg_clear),
        np.array(n_srcs),
        np.array(nreqs),
        np.array(det_frac),
    )


def summarize(name, times, real_times, frac, avgc, nsrc, nreqs, det):
    fin = np.isfinite(avgc)
    print(f"--- {name} ---")
    print(f"cases            : {len(times)}   mean n_src={nsrc.mean():.2f}")
    print(
        f"cleared fraction : mean={frac.mean():.3f} min={frac.min():.3f} "
        f"all-clear={np.mean(frac >= 0.999):.3f}"
    )
    print(f"detectable ceil  : mean={det.mean():.3f} min={det.min():.3f}")
    print(f"efficiency (clr/det): mean={(frac / np.maximum(det, 1e-9)).mean():.3f}")
    print(
        f"virtual time (s) : mean={times.mean():.1f} median={np.median(times):.1f} "
        f"max={times.max():.1f} under_360000={np.mean(times <= VIRTUAL_BUDGET) * 100:.1f}%"
    )
    print(
        f"real time (s)    : mean={real_times.mean():.2f} max={real_times.max():.2f} "
        f"requests mean={nreqs.mean():.1f}"
    )
    print(
        f"avg clear (s)    : mean={np.nanmean(avgc):.1f} median={np.nanmedian(avgc):.1f}"
    )
    return dict(
        name=name,
        frac=frac,
        times=times,
        real_times=real_times,
        avgc=avgc,
        nsrc=nsrc,
        nreqs=nreqs,
        det=det,
    )


# 绘图及报告导出源码已注释移至“辅助材料/注释代码”。
