"""
CUMCM 2026 Problem B -- core module
TaskSimulator (faithful to the official protocol) + computational geometry.
"""

import math
import random
import itertools
import numpy as np

# ---- constants (from problem statement) ----
SPEED = 5.0  # m/s
SWITCH_T = 1.0  # s (channel switch)
DETECT_T = 5.0  # s (one /measure detection)
CLEAR_OK_T = 5.0  # s (successful clear)
CLEAR_FAIL_T = 3.0  # s (failed clear)
BEARING_EPS = 1.0  # deg, half-width of direction error
REGION_R = 1800.0  # m, target circle radius
R_MIN, R_MAX = 1000.0, 1500.0  # reception radius bounds
CLEAR_R = 20.0  # m
NEAR_R = 5.0  # m
N_CHANNEL = 20
N_SRC_MIN, N_SRC_MAX = 10, 16

# ---- run-time budgets (protocol Appx 3) ----
MAX_VIRTUAL_DURATION_S = 360000.0  # 100 h virtual activity cap
MAX_REAL_DURATION_S = 1200.0  # 20 min real program runtime cap


def deg2rad(d):
    return d * math.pi / 180.0


def rad2deg(r):
    return r * 180.0 / math.pi


def uvec(deg):
    r = deg2rad(deg)
    return np.array([math.cos(r), math.sin(r)])


def norm_deg(d):
    d = d % 360.0
    return d


def ang_diff(a, b):
    """signed smallest difference a-b in deg, in [-180,180]"""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def ang_bearing(p, q):
    """bearing in deg (0=East, CCW) from point p to point q"""
    return norm_deg(rad2deg(math.atan2(q[1] - p[1], q[0] - p[0])))


def dist(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    return float(np.hypot(*(q - p)))


# ============================================================
# TaskSimulator (offline, faithful to official protocol)
# ============================================================
class TaskSimulator:
    """Faithful offline simulator.

    Protocol facts mirrored:
      * /measure(pos, channel): travel time |pos-cur|/5, +1 s if channel changes,
        +5 s detection. Returns no_signal / near / direction(svd_deg).
        direction error in [-1,1] deg and FIXED at a given point (location-based).
      * /clear(pos, channel): travel + 5 s (success, <=20 m) or 3 s (fail);
        /clear does NOT switch the receiver channel.
      * reception radius per-source uniform in [1000,1500]; directional source
        covers its heading +-90 deg (half-plane); near only if <=5 m AND in coverage.
      * sources on distinct channels; 10..16 sources; channels in 1..20.
      * already-cleared sources give no_signal on /measure and no_target on /clear.
      * /enter returns max_virtual_duration_s=360000 and max_real_duration_s=1200,
        remaining_real_duration_s; interface closes when either budget is spent.
    """

    def __init__(self, sources, reception=None, seed=0, real_latency=0.05):
        self.sources = list(sources)  # dicts: pos, channel, kind, dir(deg)
        self.reception = (
            reception if reception is not None else random.uniform(R_MIN, R_MAX)
        )
        self.rng = random.Random(seed)
        # per-source effective reception radius in [1000,1500] m (problem Appx 2(2))
        for _s in self.sources:
            _s.setdefault("reception", self.rng.uniform(R_MIN, R_MAX))
        self.time = 0.0
        self.real_time = 0.0
        self.real_latency = real_latency  # simulated HTTP round-trip per action (s)
        self.entered = False
        self.exited = False
        self.max_virtual = float(MAX_VIRTUAL_DURATION_S)
        self.max_real = float(MAX_REAL_DURATION_S)
        self.remaining_real = self.max_real
        self.n_requests = 0
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_channel = 1
        self._err = {}  # location key -> fixed error deg (location-based)
        self.log = []  # full action log (for support material)

    # ---- run-control (mirrors /enter, /exit, timeout handling) ----
    def enter(self, remaining_real=None):
        self.entered = True
        self.remaining_real = (
            self.max_real if remaining_real is None else float(remaining_real)
        )
        self.remaining_real = max(0.0, min(self.remaining_real, self.max_real))
        self.log.append(("enter", None, None, "accepted", None, 0.0))
        return {
            "accepted": True,
            "virtual_time_s": 0.0,
            "max_virtual_duration_s": int(self.max_virtual),
            "max_real_duration_s": int(self.max_real),
            "remaining_real_duration_s": int(self.remaining_real),
        }

    def exit(self):
        self.exited = True
        self.log.append(("exit", None, None, "user_exit", None, round(self.time, 2)))
        return {"accepted": True, "exit_reason": "user_exit"}

    def virtual_ok(self):
        return self.time < self.max_virtual

    def real_ok(self):
        return self.real_time < self.remaining_real

    def alive(self):
        # interface still open (entered, not exited, within both budgets)
        return self.entered and not self.exited and self.virtual_ok() and self.real_ok()

    def _charge(self):
        self.n_requests += 1
        self.real_time += self.real_latency

    # ---- internal: which source on a channel is relevant ----
    def _src_by_channel(self, channel):
        for i, s in enumerate(self.sources):
            if s["channel"] == channel and not s.get("cleared", False):
                return i, s
        return None, None

    def _in_coverage(self, s, p):
        """is point p inside source s's coverage (reception + angular)?"""
        d = dist(s["pos"], p)
        if d > s.get("reception", self.reception):
            return False
        if s["kind"] == "dir":
            # coverage: direction of source heading +-90 deg -> half-plane
            u = uvec(s["dir"])
            if float(np.dot(np.asarray(p, float) - np.asarray(s["pos"], float), u)) < 0:
                return False
        return True

    def _error(self, pos):
        key = (round(pos[0], 2), round(pos[1], 2))
        if key not in self._err:
            self._err[key] = self.rng.uniform(-BEARING_EPS, BEARING_EPS)
        return self._err[key]

    def measure(self, pos, channel):
        if not self.alive():
            return {"type": "closed", "svd_deg": None}
        pos = np.asarray(pos, float)
        dt = dist(self.robot_pos, pos) / SPEED
        if channel != self.robot_channel:
            dt += SWITCH_T
        dt += DETECT_T
        self.time += dt
        self._charge()
        self.robot_pos = pos.copy()
        self.robot_channel = channel
        i, s = self._src_by_channel(channel)
        res = {"type": "no_signal", "svd_deg": None}
        if s is not None:
            d = dist(s["pos"], pos)
            if self._in_coverage(s, pos):
                if d <= NEAR_R:
                    res = {"type": "near", "svd_deg": None}
                else:
                    th = ang_bearing(pos, s["pos"])
                    th = norm_deg(th + self._error(pos))
                    res = {"type": "direction", "svd_deg": th}
        self.log.append(
            (
                "measure",
                tuple(np.round(pos, 2)),
                channel,
                res["type"],
                None if res["svd_deg"] is None else round(res["svd_deg"], 3),
                round(self.time, 2),
            )
        )
        return res

    def clear(self, pos, channel):
        if not self.alive():
            return False
        pos = np.asarray(pos, float)
        dt = dist(self.robot_pos, pos) / SPEED
        self.time += dt
        i, s = self._src_by_channel(channel)
        ok = False
        if s is not None and dist(s["pos"], pos) <= CLEAR_R:
            ok = True
        self.time += CLEAR_OK_T if ok else CLEAR_FAIL_T
        self._charge()
        if ok and s is not None:
            s["cleared"] = True
        self.robot_pos = pos.copy()
        # NOTE: /clear does NOT change the receiver channel (protocol Appx 2)
        self.log.append(
            (
                "clear",
                tuple(np.round(pos, 2)),
                channel,
                "success" if ok else "fail",
                None,
                round(self.time, 2),
            )
        )
        return ok

    def clear_all(self):
        return sum(1 for s in self.sources if s["cleared"])


# ============================================================
# triangular (hexagonal) grid
# ============================================================
def build_triangular_grid(cover_radius, region_radius=REGION_R):
    """Triangular lattice with covering radius = cover_radius (inradius of hex cell).
    Lattice vectors: a=(s,0), b=(s/2, sqrt(3)/2*s) with s = sqrt(3)*cover_radius.
    Returns lattice points inside the region circle (with margin) and those used to
    cover the disk."""
    s = math.sqrt(3.0) * cover_radius
    pts = []
    R = region_radius
    # range of lattice indices
    nmax = int(math.ceil(R / s)) + 2
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            p = np.array([s * i + s * 0.5 * j, s * math.sqrt(3.0) / 2 * j])
            if np.hypot(p[0], p[1]) <= R + cover_radius * 0.5:
                pts.append(p)
    # keep points inside region (robot stays in target area)
    pts = [p for p in pts if np.hypot(p[0], p[1]) <= R + 1e-6]
    return pts


def grid_cover_check(pts, cover_radius, region_radius=REGION_R, n_samples=4000):
    """max distance from sampled region points to nearest grid point."""
    import random as _r

    _r.seed(1)
    worst = 0.0
    for k in range(n_samples):
        a = _r.uniform(0, 2 * math.pi)
        r = region_radius * math.sqrt(_r.random())
        q = np.array([r * math.cos(a), r * math.sin(a)])
        d = min(dist(q, p) for p in pts)
        worst = max(worst, d)
    return worst


def nn_order(pts, start=(0.0, 0.0)):
    """nearest-neighbor route through pts starting from start."""
    remain = [np.asarray(p, float) for p in pts]
    start = np.asarray(start, float)
    order = []
    cur = start
    while remain:
        i = min(range(len(remain)), key=lambda k: dist(cur, remain[k]))
        order.append(remain[i])
        cur = remain[i]
        del remain[i]
    return order


# ============================================================
# geometry: half-plane intersection, polygon ops, diameter, MEC
# ============================================================
def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def clip_poly_halfplane(poly, p, u, keep_left=True):
    """Clip polygon (list of 2d pts) by half-plane through p with inward normal u
    (keep points q with dot(q-p,u)>=0). Sutherland-Hodgman."""
    if len(poly) == 0:
        return []
    out = []
    n = len(poly)
    for i in range(n):
        a = np.asarray(poly[i], float)
        b = np.asarray(poly[(i + 1) % n], float)
        da = float(np.dot(a - p, u))
        db = float(np.dot(b - p, u))
        ain = da >= -1e-12
        bin = db >= -1e-12
        if ain:
            out.append(a)
        if ain != bin:
            t = da / (da - db)
            out.append(a + t * (b - a))
    return out


def bearing_wedge_halfplanes(S, theta_deg):
    """Return list of (point, normal) half-planes defining the wedge
    ang(S->P) in [theta-1, theta+1]."""
    lo = theta_deg - BEARING_EPS
    hi = theta_deg + BEARING_EPS
    ulo = uvec(lo)
    uhi = uvec(hi)
    S = np.asarray(S, float)
    # constraint cross(u(lo), P-S) >= 0  <=> normal = perpendicular to u(lo) on the left
    nlo = np.array([-ulo[1], ulo[0]])  # left normal of u(lo): dot(P-S, nlo)>=0
    # constraint cross(u(hi), P-S) <= 0  <=> dot(P-S, n_hi) >= 0 with n_hi = -left normal of u(hi)
    nhi = np.array([uhi[1], -uhi[0]])
    return [(S, nlo), (S, nhi)]


def localization_polygon(measures, region_radius=REGION_R):
    """Intersection of all bearing wedges + target disk (polygon approx)."""
    # start with large bounding square
    L = region_radius * 2 + 5000
    poly = [[-L, -L], [L, -L], [L, L], [-L, L]]
    for S, th in measures:
        for p, u in bearing_wedge_halfplanes(S, th):
            poly = clip_poly_halfplane(poly, p, u)
            if not poly:
                return []
    # clip to disk with a fine polygon
    circ = [
        [region_radius * math.cos(t), region_radius * math.sin(t)]
        for t in np.linspace(0, 2 * math.pi, 1080)
    ]
    poly = clip_polygon_by_polygon(poly, circ)
    return [tuple(p) for p in poly]


def clip_polygon_by_polygon(subject, clip):
    """Sutherland-Hodgman: clip subject polygon by convex clip polygon (CCW)."""
    result = [list(p) for p in subject]
    n = len(clip)
    for i in range(n):
        a = np.asarray(clip[i], float)
        b = np.asarray(clip[(i + 1) % n], float)
        e = b - a
        # inward normal for CCW polygon: (-ey, ex)
        u = np.array([-e[1], e[0]])
        result = clip_poly_halfplane(result, a, u)
        if not result:
            return []
    return result


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        s += a[0] * b[1] - a[1] * b[0]
    return abs(s) / 2.0


def polygon_diameter(poly):
    """diameter of a convex polygon (brute force O(n^2)); n is small."""
    best = 0.0
    pair = None
    for i in range(len(poly)):
        for j in range(i + 1, len(poly)):
            d = dist(poly[i], poly[j])
            if d > best:
                best = d
                pair = (poly[i], poly[j])
    return best, pair


def welzl(points):
    """Smallest enclosing circle (Welzl) over list of 2d points."""
    pts = [np.asarray(p, float) for p in points]
    rng = random.Random(0)

    def mec2(p, q):
        c = (p + q) / 2.0
        return c, dist(c, p)

    def mec3(p, q, r):
        ax, ay = p
        bx, by = q
        cx, cy = r
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-12:
            return mec2(p, q)
        ux = (
            (ax**2 + ay**2) * (by - cy)
            + (bx**2 + by**2) * (cy - ay)
            + (cx**2 + cy**2) * (ay - by)
        ) / d
        uy = (
            (ax**2 + ay**2) * (cx - bx)
            + (bx**2 + by**2) * (ax - cx)
            + (cx**2 + cy**2) * (bx - ax)
        ) / d
        c = np.array([ux, uy])
        return c, dist(c, p)

    def trivial(boundary):
        if len(boundary) == 0:
            return np.zeros(2), 0.0
        if len(boundary) == 1:
            return boundary[0], 0.0
        if len(boundary) == 2:
            return mec2(boundary[0], boundary[1])
        c, r = mec3(boundary[0], boundary[1], boundary[2])
        return c, r

    def welzl_rec(P, R):
        if not P or len(R) == 3:
            return trivial(R)
        idx = rng.randrange(len(P))
        p = P[idx]
        P2 = P[:idx] + P[idx + 1 :]
        c, r = welzl_rec(P2, R)
        if dist(c, p) <= r + 1e-9:
            return c, r
        return welzl_rec(P2, R + [p])

    order = pts[:]
    rng.shuffle(order)
    c, r = welzl_rec(order, [])
    return c, r


# ============================================================
# triangulation / homing helpers
# ============================================================
def intersect_lines(p1, th1, p2, th2):
    """intersection of ray from p1 at bearing th1 with ray from p2 at bearing th2."""
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    u1 = uvec(th1)
    u2 = uvec(th2)
    # p1 + t u1 = p2 + s u2
    A = np.array([[u1[0], -u2[0]], [u1[1], -u2[1]]])
    b = p2 - p1
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-12:
        return None
    ts = np.linalg.solve(A, b)
    return p1 + ts[0] * u1


def triangulate(detections):
    """detections: list of (pos, bearing_deg). Pick best pair and intersect."""
    best = None
    best_sin = -1.0
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            s = abs(math.sin(deg2rad(ang_diff(detections[i][1], detections[j][1]))))
            if s > best_sin:
                best_sin = s
                best = (i, j)
    if best is None:
        return None, 0.0
    i, j = best
    pt = intersect_lines(
        detections[i][0], detections[i][1], detections[j][0], detections[j][1]
    )
    return pt, best_sin


# ============================================================
# random case + covering stations (Q3/Q4)
# ============================================================
def random_case(n_src=None, rng=None, kinds=("omni",)):
    """Generate one random scenario: sources uniformly in the target disk,
    distinct channels in 1..20, per-source reception in [1000,1500].
    kinds: choose from {'omni','dir'}."""
    rng = rng or random.Random()
    if n_src is None:
        n_src = rng.randint(N_SRC_MIN, N_SRC_MAX)
    channels = rng.sample(range(1, N_CHANNEL + 1), n_src)
    srcs = []
    for c in channels:
        # uniform in disk
        a = rng.uniform(0, 2 * math.pi)
        r = REGION_R * math.sqrt(rng.random())
        pos = (r * math.cos(a), r * math.sin(a))
        kind = rng.choice(list(kinds))
        d = {
            "pos": pos,
            "channel": c,
            "kind": kind,
            "reception": rng.uniform(R_MIN, R_MAX),
            "cleared": False,
        }
        if kind == "dir":
            d["dir"] = rng.uniform(0, 360.0)
        srcs.append(d)
    return srcs


def hex_stations(r_station):
    """Covering stations: center + 6 points on a hexagon of radius r_station."""
    pts = [np.array([0.0, 0.0])]
    for k in range(6):
        a = deg2rad(60.0 * k)
        pts.append(np.array([r_station * math.cos(a), r_station * math.sin(a)]))
    return pts


def path_length(order):
    """total length of route given ordered points (first point is start)."""
    tot = 0.0
    for a, b in zip(order[:-1], order[1:]):
        tot += dist(a, b)
    return tot
