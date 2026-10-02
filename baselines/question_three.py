"""
CUMCM 2026 Problem B - Q3 (omnidirectional sources)
Triangular-grid coverage sweep + least-squares bearing localization +
fine homing + walk-along-bearing clear, validated against the faithful simulator.

Budget model (protocol Appx 3):
  * VIRTUAL_BUDGET = 360000 s  -> accumulated virtual activity cap.
  * REAL_BUDGET    = 1200 s    -> wall-clock program runtime, read from /enter's
    remaining_real_duration_s and enforced separately via sim.real_time.

Coverage and localization strategy:
  1. Cover the 1800 m disk with a triangular lattice of stations whose covering
     radius is 'cover' (default 650 m).  Stations are kept out to R+cover so the
     disk boundary is fully covered; every source is therefore within cover of
     at least one station and hence within its 1000..1500 m reception radius.
  2. Visit stations in a nearest-neighbour order from the origin; on each station
     scan all 20 channels (skipping cleared channels and channels already well
     localised).  A 'direction' reading contributes a bearing; >=2 bearings are
     triangulated by least squares.  A 'near' reading is cleared immediately.
  3. After each station (interleave=True) clear the currently localised sources,
     nearest first, via fine homing + walk-along-bearing.
  4. After the sweep, resolve any single-bearing channels and clear the rest.
"""

import math
import random
import argparse
import numpy as np

from . import core
from .core import (
    TaskSimulator,
    random_case,
    nn_order,
    dist,
    uvec,
    SPEED,
    SWITCH_T,
    DETECT_T,
    CLEAR_OK_T,
    CLEAR_FAIL_T,
    REGION_R,
    R_MIN,
    R_MAX,
    N_CHANNEL,
    NEAR_R,
    CLEAR_R,
    MAX_VIRTUAL_DURATION_S,
    MAX_REAL_DURATION_S,
    N_SRC_MIN,
    N_SRC_MAX,
    ang_diff,
    deg2rad,
)

VIRTUAL_BUDGET = MAX_VIRTUAL_DURATION_S  # 360000 s virtual activity cap
REAL_BUDGET = MAX_REAL_DURATION_S  # 1200 s wall-clock runtime cap

# ---- strategy parameters ----
COVER_DEFAULT = 650.0  # covering radius of the triangular station lattice (m)
SKIP_SIN = 0.42  # skip channel once localization spread >= sin(25 deg)
SKIP_NDET = 20  # hard cap on direction detections per channel (bound requests)
HOMING_ITERS = 6  # max refine iterations in clear_source
WALK_SPREAD_MIN = 0.30  # sin(17.5 deg): geometry good enough to trust a short walk
WALK_STEP = 10.0  # m, fine step along a fresh bearing
WALK_MAX = 600.0  # m, max fine walk (adaptive: stop on bearing flip / signal loss)
FINE_RINGS = (12.0, 24.0, 36.0, 48.0)  # ring-search radii (m)
FINE_AZIM = 8  # azimuths per ring in the last-resort search
BUDGET_FRAC = 0.985  # stop the sweep at this fraction of the virtual cap
GRID_SEARCH_R = 200.0  # m, radius of dense clear-only fallback sweep
GRID_COVER = 19.0  # m, covering radius of the clear grid (<= CLEAR_R=20)


# ---------------- geometry helpers ----------------
def clip_inside(p, radius=REGION_R):
    """Clip a point into a disk of the given radius (default: source region)."""
    p = np.asarray(p, float)
    r = math.hypot(p[0], p[1])
    if r > radius and r > 0:
        p = p * (radius / r)
    return p


# Robot operational radius for homing/clear walks. Sources are strictly inside
# REGION_R, but the robot must reach within CLEAR_R of a boundary source, so the
# working disk is the source region expanded by one clear radius. Stations are
# already placed out to REGION_R+cover for full coverage, so this stays inside the
# robot's natural footprint.
OP_R = REGION_R + CLEAR_R


def clip_op(p):
    """Clip an operational (homing/clear) point into the robot's working disk."""
    return clip_inside(p, radius=OP_R)


def cover_grid(cover=COVER_DEFAULT, radius=REGION_R):
    """Triangular lattice whose covering radius is `cover`, clipped to |p| <= R+cover
    so the whole target disk is covered. The origin is a lattice point."""
    s = math.sqrt(3.0) * cover  # lattice spacing
    nmax = int(math.ceil((radius + cover) / s)) + 1
    pts = []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            x = s * (i + 0.5 * j)
            y = s * (math.sqrt(3.0) / 2.0) * j
            if math.hypot(x, y) <= radius + cover + 1e-6:
                pts.append((x, y))
    return pts


# ---------------- least-squares bearing localization ----------------
def lsq_triangulate(detections):
    """Fit best intersection of all bearing lines (bearing-only AOA LSQ).
    detections: list of (pos, bearing_deg). Returns (point, spread) where
    spread = max pairwise |sin(angle difference)| (0=parallel, 1=perpendicular)."""
    dets = [(np.asarray(p, float), th) for p, th in detections if th is not None]
    if len(dets) < 2:
        return None, 0.0
    spread = 0.0
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            s = abs(math.sin(deg2rad(ang_diff(dets[i][1], dets[j][1]))))
            spread = max(spread, s)
    A = []
    b = []
    for S, th in dets:
        u = uvec(th)
        A.append([-u[1], u[0]])
        b.append(u[0] * S[1] - u[1] * S[0])
    A = np.array(A)
    b = np.array(b)
    P, *_ = np.linalg.lstsq(A, b, rcond=None)
    return clip_inside(P), spread


# ---------------- budget helpers ----------------
def budget_ok(sim, target, extra, budget):
    """True if the planned travel+operation fits BOTH the virtual budget and the
    remaining real (wall-clock) budget."""
    if not sim.entered or sim.exited:
        return False
    if sim.time + dist(sim.robot_pos, target) / SPEED + extra > budget + 1e-9:
        return False
    if sim.real_time + sim.real_latency > sim.remaining_real:
        return False
    return True


def measure_at(sim, pos, ch, budget):
    if not budget_ok(
        sim, pos, (1.0 if ch != sim.robot_channel else 0.0) + DETECT_T, budget
    ):
        return None
    return sim.measure(pos, ch)


def do_clear(sim, pos, ch, cleared, budget):
    p = clip_op(np.asarray(pos, float))
    if not budget_ok(sim, p, CLEAR_OK_T, budget):
        return False
    ok = sim.clear(p, ch)
    if ok:
        cleared.add(ch)
    return ok


# ---------------- walk-along-bearing clear ----------------
def walk_along_clear(sim, ch, p, th, cleared, budget, max_len=WALK_MAX, step=WALK_STEP):
    """Walk from p along bearing th (deg) in fine steps. Adaptive: keep walking while
    the source keeps answering 'direction' from roughly the same direction; stop and
    clear once the bearing flips (>90 deg, we passed the source) or the signal is lost
    (directional back side). The 1-deg bearing error bounds the lateral offset to
    max_len*sin(1 deg) ~ 10 m, safely inside the 20 m clear radius."""
    th0 = th
    n = int(math.ceil(max_len / step))
    for k in range(1, n + 1):
        q = clip_op(np.asarray(p, float) + (k * step) * uvec(th0))
        res = measure_at(sim, q, ch, budget)
        if res is None:
            return False
        if res["type"] == "near":
            return do_clear(sim, q, ch, cleared, budget)
        if res["type"] == "direction":
            if abs(ang_diff(res["svd_deg"], th0)) > 90.0:
                # passed the source: clear at q and step back up to 2 steps
                for m in (0, 1, 2):
                    qm = clip_op(q - (m * step) * uvec(th0))
                    if do_clear(sim, qm, ch, cleared, budget):
                        return True
                return False
            continue
        # no_signal: directional source -> we passed out of its front half-plane
        for m in (0, 1, 2, 3):
            qm = clip_op(q - (m * step) * uvec(th0))
            if do_clear(sim, qm, ch, cleared, budget):
                return True
        return False
    return False


# ---------------- dense clear-only grid (distance-only fallback) ----------------
def clear_grid_search(
    sim, ch, p0, cleared, budget, r_search=GRID_SEARCH_R, cover=GRID_COVER
):
    """Dense triangular grid of `clear` attempts around p0. `clear` only needs the
    robot within CLEAR_R of the source, so unlike direction detection this works
    even on the back side of a directional source (where measure() returns
    no_signal but clear() still succeeds)."""
    if p0 is None:
        return False
    p0 = clip_op(np.asarray(p0, float))
    s = math.sqrt(3.0) * cover
    n = int(math.ceil(r_search / s)) + 1
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            q = clip_op(
                p0 + np.array([s * (i + 0.5 * j), s * (math.sqrt(3.0) / 2.0) * j])
            )
            if math.hypot(*(q - p0)) > r_search + 1e-6:
                continue
            if not sim.alive():
                return False
            if do_clear(sim, q, ch, cleared, budget):
                return True
    return False


# ---------------- last-resort fine search ----------------
def fine_search_and_clear(sim, ch, p0, cleared, budget, directional=False):
    """Ring search around a coarse estimate, then walk-in on any fresh front bearing."""
    if p0 is None:
        return False
    p0 = clip_op(np.asarray(p0, float))
    for R in FINE_RINGS:
        for k in range(FINE_AZIM):
            az = deg2rad(90.0 * k + (37.0 if directional else 11.0))
            q = clip_op(p0 + R * np.array([math.cos(az), math.sin(az)]))
            res = measure_at(sim, q, ch, budget)
            if res is None:
                return False
            if res["type"] == "near":
                return do_clear(sim, q, ch, cleared, budget)
            if res["type"] == "direction":
                if walk_along_clear(
                    sim, ch, q, res["svd_deg"], cleared, budget, max_len=2.0 * R + 60.0
                ):
                    return True
    return clear_grid_search(sim, ch, p0, cleared, budget, r_search=GRID_SEARCH_R)


# ---------------- single-bearing resolution ----------------
def hunt_single(sim, ch, bearings, est, spread, cleared, budget, directional=False):
    dets = [d for d in bearings.get(ch, []) if d[1] is not None]
    if not dets:
        return False
    S, th = dets[0]
    S = np.asarray(S, float)
    for t in (200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1500.0):
        q = clip_op(S + t * uvec(th))
        res = measure_at(sim, q, ch, budget)
        if res is None:
            return False
        if res["type"] == "near":
            return do_clear(sim, q, ch, cleared, budget)
        if res["type"] == "direction":
            bearings.setdefault(ch, []).append((tuple(q), res["svd_deg"]))
            # distance-only clear grid around this front-side point catches a
            # directional source even if the ray-walk drifts out of the half-plane
            if directional and clear_grid_search(
                sim, ch, q, cleared, budget, r_search=250.0
            ):
                return True
            # break the near-collinear ambiguity with a perpendicular offset
            for perp in (250.0, 500.0):
                for sgn in (1.0, -1.0):
                    q2 = clip_op(q + perp * uvec(res["svd_deg"] + 90.0 * sgn))
                    res2 = measure_at(sim, q2, ch, budget)
                    if res2 is None:
                        return False
                    if res2["type"] == "near":
                        return do_clear(sim, q2, ch, cleared, budget)
                    if res2["type"] == "direction":
                        bearings[ch].append((tuple(q2), res2["svd_deg"]))
                        p2, s = lsq_triangulate(bearings[ch])
                        if p2 is not None:
                            est[ch] = p2
                            spread[ch] = s
                            return clear_source(
                                sim,
                                ch,
                                est,
                                spread,
                                cleared,
                                bearings,
                                budget,
                                directional=directional,
                            )
        else:
            # no_signal: for directional we passed the source (back side); walk back finely
            if do_clear(sim, q, ch, cleared, budget):
                return True
            for tb in np.arange(max(0.0, t - 200.0), t + 5.0, 10.0):
                qb = clip_op(S + tb * uvec(th))
                resb = measure_at(sim, qb, ch, budget)
                if resb is None:
                    return False
                if resb["type"] == "near":
                    return do_clear(sim, qb, ch, cleared, budget)
                if resb["type"] == "direction":
                    return walk_along_clear(
                        sim, ch, qb, resb["svd_deg"], cleared, budget, max_len=60.0
                    )
            break
    return False


# ---------------- one-source clear with homing ----------------
def clear_source(sim, ch, est, spread, cleared, bearings, budget, directional=False):
    """Refine the current estimate with close-range bearings, walk in, and clear."""
    p = est.get(ch)
    if p is None:
        return hunt_single(
            sim, ch, bearings, est, spread, cleared, budget, directional=directional
        )

    walked = False
    for it in range(HOMING_ITERS + 1):
        p = clip_op(np.asarray(est.get(ch), float))
        res = measure_at(sim, p, ch, budget)
        if res is None:
            return False
        if res["type"] == "near":
            return do_clear(sim, p, ch, cleared, budget)
        if res["type"] == "direction":
            bearings.setdefault(ch, []).append((tuple(p), res["svd_deg"]))
            p2, s = lsq_triangulate(bearings[ch])
            if p2 is not None:
                est[ch] = p2
                spread[ch] = s
            if not walked:
                walked = True
                if walk_along_clear(sim, ch, p, res["svd_deg"], cleared, budget):
                    return True
            continue
        else:
            # no_signal at estimate (directional back side / out of range)
            if do_clear(sim, p, ch, cleared, budget):
                return True
            break

    p = est.get(ch)
    if p is not None:
        p = clip_op(np.asarray(p, float))
        if do_clear(sim, p, ch, cleared, budget):
            return True
    return fine_search_and_clear(
        sim, ch, est.get(ch), cleared, budget, directional=directional
    )


def clear_localized(
    sim, est, spread, cleared, bearings, budget, directional=False, stuck=None
):
    """Greedy nearest-first clearing of all currently localized sources.
    A source that fails to clear is marked stuck so it is not retried after every
    station (that retry storm previously burned the real-time budget)."""
    if stuck is None:
        stuck = set()
    while True:
        cands = [
            c for c in est if c not in cleared and est[c] is not None and c not in stuck
        ]
        if not cands:
            return
        cur = sim.robot_pos
        ch = min(cands, key=lambda c: dist(cur, est[c]))
        if not clear_source(
            sim, ch, est, spread, cleared, bearings, budget, directional=directional
        ):
            if not sim.alive() or sim.time >= budget * BUDGET_FRAC:
                return
            stuck.add(ch)


# ---------------- main strategy ----------------
def solve(
    sim,
    cover=COVER_DEFAULT,
    budget=VIRTUAL_BUDGET,
    interleave=True,
    directional=False,
    **kwargs,
):
    """Run the full sweep + localize + clear strategy. Returns the simulator."""
    if not sim.entered:
        sim.enter()
    stations = cover_grid(cover)
    order = nn_order(stations, (0.0, 0.0))
    bearings = {}
    est = {}
    spread = {}
    cleared = set()
    stuck = set()

    for st in order:
        if not sim.alive() or sim.time >= budget * BUDGET_FRAC:
            break
        for ch in range(1, N_CHANNEL + 1):
            if ch in cleared:
                continue
            if ch in est and est[ch] is not None:
                nd = len([d for d in bearings.get(ch, []) if d[1] is not None])
                if spread.get(ch, 0.0) >= SKIP_SIN or nd >= SKIP_NDET:
                    continue
            res = measure_at(sim, st, ch, budget)
            if res is None:
                break
            if res["type"] == "direction":
                bearings.setdefault(ch, []).append((tuple(st), res["svd_deg"]))
                if len([d for d in bearings[ch] if d[1] is not None]) >= 2:
                    p, s = lsq_triangulate(bearings[ch])
                    if p is not None:
                        est[ch] = p
                        spread[ch] = s
            elif res["type"] == "near":
                est[ch] = np.asarray(st, float)
                bearings.setdefault(ch, []).append((tuple(st), None))
                do_clear(sim, st, ch, cleared, budget)
        if interleave:
            clear_localized(
                sim,
                est,
                spread,
                cleared,
                bearings,
                budget,
                directional=directional,
                stuck=stuck,
            )

    # resolve single-bearing channels, then one final clearing pass (stuck sources
    # get one more chance now that all bearings from the full sweep are available)
    for ch in sorted(bearings.keys()):
        if ch not in cleared and (ch not in est or est[ch] is None):
            hunt_single(
                sim, ch, bearings, est, spread, cleared, budget, directional=directional
            )
    clear_localized(
        sim,
        est,
        spread,
        cleared,
        bearings,
        budget,
        directional=directional,
        stuck=set(),
    )
    return sim


# ---------------- Monte Carlo ----------------
def run_mc(
    N=100, n_src=None, cover=COVER_DEFAULT, interleave=True, seed=1, kinds=("omni",)
):
    rng = random.Random(seed)
    times = []
    real_times = []
    cleared_frac = []
    avg_clear = []
    n_srcs = []
    nreqs = []
    for k in range(N):
        n = n_src if n_src is not None else rng.randint(N_SRC_MIN, N_SRC_MAX)
        srcs = random_case(n, rng=rng, kinds=kinds)
        sim = TaskSimulator(srcs, seed=k)
        solve(sim, cover=cover, interleave=interleave, directional=("dir" in kinds))
        nc = sim.clear_all()
        times.append(sim.time)
        real_times.append(sim.real_time)
        cleared_frac.append(nc / n)
        avg_clear.append(sim.time / nc if nc > 0 else float("nan"))
        n_srcs.append(n)
        nreqs.append(sim.n_requests)
    return (
        np.array(times),
        np.array(real_times),
        np.array(cleared_frac),
        np.array(avg_clear),
        np.array(n_srcs),
        np.array(nreqs),
    )


def summarize(name, times, real_times, frac, avgc, nsrc, nreqs):
    fin = np.isfinite(avgc)
    print(f"--- {name} ---")
    print(f"cases            : {len(times)}")
    print(f"mean n_src       : {nsrc.mean():.2f}")
    print(
        f"cleared fraction : mean={frac.mean():.3f}  min={frac.min():.3f}  "
        f"all-clear rate={np.mean(frac >= 0.999):.3f}"
    )
    print(
        f"virtual time (s) : mean={times.mean():.1f}  median={np.median(times):.1f}  "
        f"max={times.max():.1f}  under_360000={np.mean(times <= VIRTUAL_BUDGET) * 100:.1f}%"
    )
    print(
        f"real time (s)    : mean={real_times.mean():.2f}  max={real_times.max():.2f}  "
        f"requests mean={nreqs.mean():.1f}"
    )
    print(
        f"avg clear (s)    : mean={np.nanmean(avgc):.1f}  median={np.nanmedian(avgc):.1f}"
    )
    return dict(
        name=name,
        frac=frac,
        times=times,
        real_times=real_times,
        avgc=avgc,
        nsrc=nsrc,
        nreqs=nreqs,
    )


# 绘图及报告导出源码已注释移至“辅助材料/注释代码”。
