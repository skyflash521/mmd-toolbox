import math

import numpy as np
from scipy.optimize import least_squares

from vmd import interp

ControlPoints = tuple[int, int, int, int]

_ZERO_EPS = 1e-9

_CONST_EPS_SCALAR = 1e-9
_CONST_EPS_POS = 1e-9
_CONST_EPS_FOV = 1e-9
_CONST_EPS_RAD = 1e-9
_CONST_EPS_DEG = 1e-9


def _normalize(err, frame, tol):
    if tol == 0.0:
        return (math.inf, frame) if err > 0.0 else (0.0, None)
    return (err / tol, frame)


def _round_half_up(x):
    return math.floor(x + 0.5)


def _clip01(v):
    return min(1.0, max(0.0, v))


def _quantize_cp(v):
    return min(127, max(0, _round_half_up(_clip01(v) * 127)))


def _select_worst(errs, is_reversal):
    if not errs:
        return None
    max_err = max(errs.values())
    if max_err <= _ZERO_EPS:
        return None
    reversals = [f for f in errs if is_reversal(f)]
    candidates = reversals if reversals else list(errs)
    return max(candidates, key=lambda f: (errs[f], -f))


def _axis_curve(a0, a1, a, b, sample_fn, early_exit_err=None, category=None, skip_fastpath=False):
    span = b - a
    internal = range(a + 1, b)
    denom = a1 - a0
    if abs(denom) <= 1e-9 or not internal:
        return _BEZIER_LINEAR_CP
    xs = [(f - a) / span for f in internal]
    ys = [(sample_fn(f) - a0) / denom for f in internal]
    cp, _ = fit_bezier_curve(
        xs, ys, early_exit_err=early_exit_err, category=category, skip_fastpath=skip_fastpath
    )
    return cp


def _bezier_axis_pred(a0, a1, a, b, sample_fn, cp=None):
    span = b - a
    internal = range(a + 1, b)
    denom = a1 - a0
    if abs(denom) <= 1e-9:
        return {f: a0 for f in internal}
    if cp is None:
        cp = _axis_curve(a0, a1, a, b, sample_fn)
    return {f: a0 + denom * interp._solve_factor(*cp, (f - a) / span) for f in internal}


class LinearScalarChannel:
    def __init__(self, frame_start, values, tol, mode="linear", force_bezier=False):
        self.frame_start = frame_start
        self.values = list(values)
        self.tol = float(tol)
        self.mode = mode
        self._force_bezier = force_bezier
        self._cp_cache: dict[tuple[int, int], ControlPoints] = {}

    def _value(self, frame):
        return self.values[frame - self.frame_start]

    def _axis_cp(self, a, b):
        cp = self._cp_cache.get((a, b))
        if cp is None:
            a0, a1 = self._value(a), self._value(b)
            denom = abs(a1 - a0)
            ee = self.tol / denom if denom > 1e-9 else None
            cp = _axis_curve(
                a0, a1, a, b, self._value, early_exit_err=ee,
                category=getattr(self, "label", None), skip_fastpath=self._force_bezier,
            )
            self._cp_cache[(a, b)] = cp
        return cp

    def residual(self, a, b):
        if self.mode == "bezier":
            pred = _bezier_axis_pred(
                self._value(a), self._value(b), a, b, self._value, cp=self._axis_cp(a, b)
            )
        else:
            va, vb, span = self._value(a), self._value(b), b - a
            pred = {f: va + (vb - va) * (f - a) / span for f in range(a + 1, b)}
        errs = {f: abs(self._value(f) - pred[f]) for f in pred}
        if not errs:
            return (0.0, None)
        max_err = max(errs.values())
        if max_err <= _ZERO_EPS:
            return (0.0, None)
        return (max_err, _select_worst(errs, self._is_reversal))

    def _is_reversal(self, frame):
        d_prev = self._value(frame) - self._value(frame - 1)
        d_next = self._value(frame + 1) - self._value(frame)
        return d_prev * d_next < 0.0

    def normalized(self, a, b):
        err, frame = self.residual(a, b)
        return _normalize(err, frame, self.tol)

    def curve(self, a, b) -> ControlPoints:
        if self.mode != "bezier":
            return _BEZIER_LINEAR_CP
        return self._axis_cp(a, b)

    def is_constant(self, a, b):
        lo = hi = self._value(a)
        for f in range(a + 1, b + 1):
            v = self._value(f)
            if v < lo:
                lo = v
            elif v > hi:
                hi = v
            if hi - lo > _CONST_EPS_SCALAR:
                return False
        return True


class EuclideanVectorChannel:
    def __init__(self, frame_start, vectors, tol, mode="linear", force_bezier=False):
        self.frame_start = frame_start
        self.vectors = [tuple(float(c) for c in v) for v in vectors]
        self.tol = float(tol)
        self.mode = mode
        self._force_bezier = force_bezier
        self._cp_cache: dict[tuple[int, int, int], ControlPoints] = {}

    def _vec(self, frame):
        return self.vectors[frame - self.frame_start]

    def _axis_cp(self, a, b, i):
        cp = self._cp_cache.get((a, b, i))
        if cp is None:
            va, vb = self._vec(a), self._vec(b)
            denom = abs(vb[i] - va[i])
            ee = self.tol / (math.sqrt(3.0) * denom) if denom > 1e-9 else None
            cp = _axis_curve(
                va[i], vb[i], a, b, lambda f: self._vec(f)[i], early_exit_err=ee,
                category=getattr(self, "label", None), skip_fastpath=self._force_bezier,
            )
            self._cp_cache[(a, b, i)] = cp
        return cp

    def residual(self, a, b):
        va = self._vec(a)
        vb = self._vec(b)
        span = b - a
        internal = list(range(a + 1, b))
        if not internal:
            return (0.0, None)
        if self.mode == "bezier":
            axis_pred = [
                _bezier_axis_pred(
                    va[i], vb[i], a, b, lambda f, i=i: self._vec(f)[i], cp=self._axis_cp(a, b, i)
                )
                for i in range(3)
            ]
            errs = {
                f: math.dist(self._vec(f), tuple(axis_pred[i][f] for i in range(3)))
                for f in internal
            }
        else:
            errs = {}
            for f in internal:
                t = (f - a) / span
                pred = tuple(va[i] + (vb[i] - va[i]) * t for i in range(3))
                errs[f] = math.dist(self._vec(f), pred)
        max_err = max(errs.values())
        if max_err <= _ZERO_EPS:
            return (0.0, None)
        return (max_err, _select_worst(errs, self._is_reversal))

    def _is_reversal(self, frame):
        prev = self._vec(frame - 1)
        cur = self._vec(frame)
        nxt = self._vec(frame + 1)
        return any((cur[i] - prev[i]) * (nxt[i] - cur[i]) < 0.0 for i in range(3))

    def normalized(self, a, b):
        err, frame = self.residual(a, b)
        return _normalize(err, frame, self.tol)

    def curve(self, a, b) -> tuple[ControlPoints, ControlPoints, ControlPoints]:
        if self.mode != "bezier":
            return (_BEZIER_LINEAR_CP, _BEZIER_LINEAR_CP, _BEZIER_LINEAR_CP)
        return tuple(self._axis_cp(a, b, i) for i in range(3))

    def is_constant(self, a, b):
        v0 = self._vec(a)
        lo = list(v0)
        hi = list(v0)
        for f in range(a + 1, b + 1):
            v = self._vec(f)
            for i in range(3):
                if v[i] < lo[i]:
                    lo[i] = v[i]
                elif v[i] > hi[i]:
                    hi[i] = v[i]
                if hi[i] - lo[i] > _CONST_EPS_POS:
                    return False
        return True


class FovChannel:
    def __init__(self, frame_start, values, tol, mode="linear", force_bezier=False):
        self.frame_start = frame_start
        self.values = [float(v) for v in values]
        self.tol = float(tol)
        self.mode = mode
        self._force_bezier = force_bezier
        self._cp_cache: dict[tuple[int, int], ControlPoints] = {}

    def _value(self, frame):
        return self.values[frame - self.frame_start]

    def _axis_cp(self, a, b):
        cp = self._cp_cache.get((a, b))
        if cp is None:
            a0, a1 = self._value(a), self._value(b)
            denom = abs(a1 - a0)
            ee = self.tol / denom if denom > 1e-9 else None
            cp = _axis_curve(
                a0, a1, a, b, self._value, early_exit_err=ee,
                category=getattr(self, "label", None), skip_fastpath=self._force_bezier,
            )
            self._cp_cache[(a, b)] = cp
        return cp

    def residual(self, a, b):
        if self.mode == "bezier":
            pred = _bezier_axis_pred(
                self._value(a), self._value(b), a, b, self._value, cp=self._axis_cp(a, b)
            )
        else:
            va, vb, span = self._value(a), self._value(b), b - a
            pred = {f: va + (vb - va) * (f - a) / span for f in range(a + 1, b)}
        errs = {f: abs(_round_half_up(pred[f]) - self._value(f)) for f in pred}
        if not errs:
            return (0.0, None)
        max_err = max(errs.values())
        if max_err <= _ZERO_EPS:
            return (0.0, None)
        return (max_err, _select_worst(errs, self._is_reversal))

    def _is_reversal(self, frame):
        d_prev = self._value(frame) - self._value(frame - 1)
        d_next = self._value(frame + 1) - self._value(frame)
        return d_prev * d_next < 0.0

    def normalized(self, a, b):
        err, frame = self.residual(a, b)
        return _normalize(err, frame, self.tol)

    def curve(self, a, b) -> ControlPoints:
        if self.mode != "bezier":
            return _BEZIER_LINEAR_CP
        return self._axis_cp(a, b)

    def is_constant(self, a, b):
        lo = hi = self._value(a)
        for f in range(a + 1, b + 1):
            v = self._value(f)
            if v < lo:
                lo = v
            elif v > hi:
                hi = v
            if hi - lo > _CONST_EPS_FOV:
                return False
        return True


def _quat_dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def _quat_normalize(q):
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def _quat_conj(q):
    return (-q[0], -q[1], -q[2], q[3])


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_slerp(q0, q1, t):
    """q0・q1 は単位クォータニオンであること。"""
    d = _quat_dot(q0, q1)
    if d < 0.0:
        q1 = tuple(-c for c in q1)
        d = -d
    if d > 0.9995:
        r = tuple(q0[i] + t * (q1[i] - q0[i]) for i in range(4))
        return _quat_normalize(r)
    th0 = math.acos(d)
    th = th0 * t
    s0 = math.sin(th0 - th) / math.sin(th0)
    s1 = math.sin(th) / math.sin(th0)
    return tuple(s0 * q0[i] + s1 * q1[i] for i in range(4))


def _quat_angle_deg(a, b):
    """a・b は単位クォータニオンであること。"""
    r = _quat_mul(b, _quat_conj(a))
    v = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    # 2·acos(|dot|) は微小角で丸め誤差が増幅し、libm の実装差で結果が揺れる。
    return math.degrees(2.0 * math.atan2(v, abs(r[3])))


class CameraRotationChannel:
    def __init__(self, frame_start, eulers, tol, mode="linear", force_bezier=False):
        self.frame_start = frame_start
        arr = np.asarray(eulers, dtype=float)
        self.eulers = np.column_stack([np.unwrap(arr[:, i]) for i in range(3)])
        self.tol = float(tol)
        self.mode = mode
        self._force_bezier = force_bezier
        self._cp_cache: dict[tuple[int, int], ControlPoints] = {}

    def _euler(self, frame):
        return self.eulers[frame - self.frame_start]

    def residual(self, a, b):
        ea = self._euler(a)
        eb = self._euler(b)
        span = b - a
        internal = range(a + 1, b)
        if self.mode == "bezier":
            cp = self.curve(a, b)
            errs = {}
            for f in internal:
                y = interp._solve_factor(*cp, (f - a) / span)
                ef = self._euler(f)
                errs[f] = max(
                    abs(math.degrees(ef[i] - (ea[i] + (eb[i] - ea[i]) * y)))
                    for i in range(3)
                )
        else:
            errs = {}
            for f in internal:
                t = (f - a) / span
                ef = self._euler(f)
                errs[f] = max(
                    abs(math.degrees(ef[i] - (ea[i] + (eb[i] - ea[i]) * t)))
                    for i in range(3)
                )
        if not errs:
            return (0.0, None)
        max_err = max(errs.values())
        if max_err <= _ZERO_EPS:
            return (0.0, None)
        reversals = [f for f in errs if self._is_reversal(f)]
        candidates = reversals if reversals else list(errs)
        worst = max(candidates, key=lambda f: (errs[f], -f))
        return (max_err, worst)

    def _is_reversal(self, frame):
        prev = self._euler(frame - 1)
        cur = self._euler(frame)
        nxt = self._euler(frame + 1)
        for i in range(3):
            d_prev = cur[i] - prev[i]
            d_next = nxt[i] - cur[i]
            if abs(d_prev) > _ZERO_EPS and abs(d_next) > _ZERO_EPS and d_prev * d_next < 0.0:
                return True
        return False

    def normalized(self, a, b):
        err, frame = self.residual(a, b)
        return _normalize(err, frame, self.tol)

    def curve(self, a, b) -> ControlPoints:
        if self.mode != "bezier":
            return _BEZIER_LINEAR_CP
        cached = self._cp_cache.get((a, b))
        if cached is not None:
            return cached
        ea = self._euler(a)
        eb = self._euler(b)
        span = b - a
        internal = range(a + 1, b)
        if not internal:
            return _BEZIER_LINEAR_CP

        def _resid_at(coeff):
            out = []
            for f in internal:
                y = coeff((f - a) / span)
                ef = self._euler(f)
                out.extend(
                    math.degrees(ef[i] - (ea[i] + (eb[i] - ea[i]) * y)) for i in range(3)
                )
            return out

        cp = _fit_coeff_curve(
            [(f - a) / span for f in internal], _resid_at, early_exit_err=self.tol,
            category=getattr(self, "label", None), skip_fastpath=self._force_bezier,
        )
        self._cp_cache[(a, b)] = cp
        return cp

    def is_constant(self, a, b):
        e0 = self._euler(a)
        lo = [e0[i] for i in range(3)]
        hi = [e0[i] for i in range(3)]
        for f in range(a + 1, b + 1):
            e = self._euler(f)
            for i in range(3):
                if e[i] < lo[i]:
                    lo[i] = e[i]
                elif e[i] > hi[i]:
                    hi[i] = e[i]
                if hi[i] - lo[i] > _CONST_EPS_RAD:
                    return False
        return True


class BoneRotationChannel:
    def __init__(self, frame_start, quats, tol, mode="linear"):
        self.frame_start = frame_start
        aligned = []
        for q in quats:
            qn = _quat_normalize(q)
            if aligned and _quat_dot(qn, aligned[-1]) < 0.0:
                qn = tuple(-c for c in qn)
            aligned.append(qn)
        self.quats = aligned
        self.tol = float(tol)
        self.mode = mode
        self._cp_cache: dict[tuple[int, int], ControlPoints] = {}

    def _q(self, frame):
        return self.quats[frame - self.frame_start]

    def residual(self, a, b):
        q0 = self._q(a)
        q1 = self._q(b)
        span = b - a
        internal = range(a + 1, b)
        if self.mode == "bezier":
            cp = self.curve(a, b)
            errs = {
                f: _quat_angle_deg(
                    self._q(f), _quat_slerp(q0, q1, interp._solve_factor(*cp, (f - a) / span))
                )
                for f in internal
            }
        else:
            errs = {}
            for f in internal:
                t = (f - a) / span
                errs[f] = _quat_angle_deg(self._q(f), _quat_slerp(q0, q1, t))
        if not errs:
            return (0.0, None)
        max_err = max(errs.values())
        if max_err <= _ZERO_EPS:
            return (0.0, None)
        reversals = [f for f in errs if self._is_reversal(f)]
        candidates = reversals if reversals else list(errs)
        worst = max(candidates, key=lambda f: (errs[f], -f))
        return (max_err, worst)

    def _rel_axis(self, frame):
        rel = _quat_mul(self._q(frame), _quat_conj(self._q(frame - 1)))
        return rel[:3]

    def _is_reversal(self, frame):
        ax0 = self._rel_axis(frame)
        ax1 = self._rel_axis(frame + 1)
        n0 = math.sqrt(sum(c * c for c in ax0))
        n1 = math.sqrt(sum(c * c for c in ax1))
        if n0 <= _ZERO_EPS or n1 <= _ZERO_EPS:
            return False
        return sum(ax0[i] * ax1[i] for i in range(3)) < 0.0

    def normalized(self, a, b):
        err, frame = self.residual(a, b)
        return _normalize(err, frame, self.tol)

    def curve(self, a, b) -> ControlPoints:
        if self.mode != "bezier":
            return _BEZIER_LINEAR_CP
        cached = self._cp_cache.get((a, b))
        if cached is not None:
            return cached
        q0 = self._q(a)
        q1 = self._q(b)
        span = b - a
        internal = range(a + 1, b)
        if not internal:
            return _BEZIER_LINEAR_CP

        def _resid_at(coeff):
            return [
                _quat_angle_deg(self._q(f), _quat_slerp(q0, q1, coeff((f - a) / span)))
                for f in internal
            ]

        cp = _fit_coeff_curve(
            [(f - a) / span for f in internal], _resid_at, early_exit_err=self.tol,
            category=getattr(self, "label", None),
        )
        self._cp_cache[(a, b)] = cp
        return cp

    def is_constant(self, a, b):
        q0 = self._q(a)
        for f in range(a + 1, b + 1):
            if _quat_angle_deg(self._q(f), q0) > _CONST_EPS_DEG:
                return False
        return True


_INIT_LINEAR = (20.0 / 127, 20.0 / 127, 107.0 / 127, 107.0 / 127)
_INIT_EASE_IN = (0.42, 0.0, 1.0, 1.0)
_INIT_EASE_OUT = (0.0, 0.0, 0.58, 1.0)
_INIT_EASE_IN_OUT = (0.42, 0.0, 0.58, 1.0)
_BEZIER_INITS = (_INIT_LINEAR, _INIT_EASE_IN, _INIT_EASE_OUT, _INIT_EASE_IN_OUT)
_BEZIER_INITS_LARGE = (_INIT_LINEAR, _INIT_EASE_IN_OUT)
_BEZIER_LINEAR_CP = (20, 20, 107, 107)
_CHEAP_EASE_CPS = tuple(
    tuple(_quantize_cp(v) for v in init) for init in (_INIT_EASE_IN, _INIT_EASE_OUT, _INIT_EASE_IN_OUT)
)

_FIT_COUNTERS = {"fit_calls": 0, "lsq_calls": 0, "fastpath_linear": 0, "cheap_accept": 0}
_FIT_COUNTERS_BY_CAT = {}


def _bump(field, category):
    _FIT_COUNTERS[field] += 1
    if category is not None:
        bucket = _FIT_COUNTERS_BY_CAT.get(category)
        if bucket is None:
            bucket = {k: 0 for k in _FIT_COUNTERS}
            _FIT_COUNTERS_BY_CAT[category] = bucket
        bucket[field] += 1


def reset_fit_counters():
    for k in _FIT_COUNTERS:
        _FIT_COUNTERS[k] = 0
    _FIT_COUNTERS_BY_CAT.clear()


def read_fit_counters():
    return dict(_FIT_COUNTERS)


def read_fit_counters_by_category():
    return {cat: dict(counts) for cat, counts in _FIT_COUNTERS_BY_CAT.items()}


_LSQ_LOOSE_TOL = 1e-3
_LSQ_LOOSEN_MIN_SAMPLES = 30


def _lsq_kwargs(n_samples):
    if n_samples < _LSQ_LOOSEN_MIN_SAMPLES:
        return {}
    return {"ftol": _LSQ_LOOSE_TOL, "xtol": _LSQ_LOOSE_TOL, "gtol": _LSQ_LOOSE_TOL}


def _bezier_inits(n_samples):
    return _BEZIER_INITS_LARGE if n_samples >= _LSQ_LOOSEN_MIN_SAMPLES else _BEZIER_INITS


def _bez(s, c1, c2):
    u = 1.0 - s
    return 3 * u * u * s * c1 + 3 * u * s * s * c2 + s * s * s


def _bezier_y_at(px1, py1, px2, py2, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    def fx(s):
        return _bez(s, px1, px2)

    def dfx(s):
        u = 1.0 - s
        return 3.0 * (px1 * u * u + 2.0 * (px2 - px1) * u * s + (1.0 - px2) * s * s)

    s = x
    converged = False
    for _ in range(20):
        err = fx(s) - x
        if abs(err) < 1e-9:
            converged = True
            break
        d = dfx(s)
        if d <= 1e-12:
            break
        s -= err / d
        if s < 0.0 or s > 1.0:
            break
    if not converged or s < 0.0 or s > 1.0 or abs(fx(s) - x) > 1e-6:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if fx(mid) < x:
                lo = mid
            else:
                hi = mid
        s = (lo + hi) / 2.0
    return _bez(s, py1, py2)


def _initial_solution(init):
    ix1, iy1, ix2, iy2 = init
    t0 = (ix2 - ix1) / (1.0 - ix1) if ix1 < 1.0 else 0.0
    return [_clip01(ix1), _clip01(t0), _clip01(iy1), _clip01(iy2)]


def _quantize_solution(sol_x):
    x1, t, y1, y2 = sol_x
    x2 = x1 + (1.0 - x1) * t
    x1q = _quantize_cp(x1)
    x2q = _quantize_cp(x2)
    y1q = _quantize_cp(y1)
    y2q = _quantize_cp(y2)
    if x1q > x2q:
        x2q = x1q
    return (x1q, y1q, x2q, y2q)


def fit_bezier_curve(xs, ys, early_exit_err=None, category=None, skip_fastpath=False):
    """(量子化した制御点 (x1, y1, x2, y2), 最大絶対誤差) を返す。

    xs は正規化時間、ys は正規化値。誤差と early_exit_err は正規化値の単位。early_exit_err を渡すと、
    量子化後の誤差がそれ以下の候補が見つかった時点で探索を打ち切る。skip_fastpath=True では、
    線形と固定 ease の候補を最適化の前に試さない。
    """
    xs = list(xs)
    ys = list(ys)
    if not xs:
        return (_BEZIER_LINEAR_CP, 0.0)
    _bump("fit_calls", category)

    def residual(v):
        x1, t, y1, y2 = v
        x2 = x1 + (1.0 - x1) * t
        return [_bezier_y_at(x1, y1, x2, y2, x) - y for x, y in zip(xs, ys, strict=True)]

    def quantized_err(cp):
        return max(abs(interp._solve_factor(*cp, x) - y) for x, y in zip(xs, ys, strict=True))

    if early_exit_err is not None and not skip_fastpath:
        lin_err = quantized_err(_BEZIER_LINEAR_CP)
        if lin_err <= early_exit_err:
            _bump("fastpath_linear", category)
            return (_BEZIER_LINEAR_CP, lin_err)
        for cand in _CHEAP_EASE_CPS:
            cand_err = quantized_err(cand)
            if cand_err <= early_exit_err:
                _bump("cheap_accept", category)
                return (cand, cand_err)

    lsq_kw = _lsq_kwargs(len(xs))
    best_cost = math.inf
    best_cp = None
    best_err = None
    for init in _bezier_inits(len(xs)):
        x0 = _initial_solution(init)
        try:
            _bump("lsq_calls", category)
            sol = least_squares(residual, x0, bounds=([0.0] * 4, [1.0] * 4), **lsq_kw)
        except Exception:
            continue
        cost = float(np.sum(np.square(residual(sol.x))))
        if cost < best_cost:
            best_cost = cost
            best_cp = _quantize_solution(sol.x)
            best_err = quantized_err(best_cp)
            if early_exit_err is not None and best_err <= early_exit_err:
                return (best_cp, best_err)
    if best_cp is None:
        best_cp = _quantize_solution(_initial_solution(_INIT_LINEAR))
        best_err = quantized_err(best_cp)
    return (best_cp, best_err)


def _fit_coeff_curve(xs, resid_at, early_exit_err=None, category=None, skip_fastpath=False):
    """量子化した制御点 (x1, y1, x2, y2) を返す。

    resid_at(coeff) は、正規化時間 x から係数 y を返す関数 coeff を受け取り、残差の列を返すこと。
    early_exit_err は残差と同じ単位。early_exit_err と skip_fastpath の意味は fit_bezier_curve と同じ。
    """
    if not xs:
        return _BEZIER_LINEAR_CP
    _bump("fit_calls", category)

    def residual(v):
        x1, t, y1, y2 = v
        x2 = x1 + (1.0 - x1) * t
        return resid_at(lambda x: _bezier_y_at(x1, y1, x2, y2, x))

    def quantized_err(cp):
        res = resid_at(lambda x: interp._solve_factor(*cp, x))
        return max((abs(r) for r in res), default=0.0)

    if early_exit_err is not None and not skip_fastpath:
        if quantized_err(_BEZIER_LINEAR_CP) <= early_exit_err:
            _bump("fastpath_linear", category)
            return _BEZIER_LINEAR_CP
        for cand in _CHEAP_EASE_CPS:
            if quantized_err(cand) <= early_exit_err:
                _bump("cheap_accept", category)
                return cand

    lsq_kw = _lsq_kwargs(len(xs))
    best_cost = math.inf
    best_cp = None
    for init in _bezier_inits(len(xs)):
        x0 = _initial_solution(init)
        try:
            _bump("lsq_calls", category)
            sol = least_squares(residual, x0, bounds=([0.0] * 4, [1.0] * 4), **lsq_kw)
        except Exception:
            continue
        cost = float(np.sum(np.square(residual(sol.x))))
        if cost < best_cost:
            best_cost = cost
            best_cp = _quantize_solution(sol.x)
            if early_exit_err is not None and quantized_err(best_cp) <= early_exit_err:
                return best_cp
    if best_cp is None:
        best_cp = _quantize_solution(_initial_solution(_INIT_LINEAR))
    return best_cp
