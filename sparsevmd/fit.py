"""補間曲線フィット・誤差評価(sparsevmd.md §5.2, §7.2)。

本モジュールはチャンネル単位の誤差評価を担う。reduce.py は分割戦略に専念し、
各チャンネルの「区間 [a,b] を表現したときの正規化誤差と最大誤差フレーム」を
このモジュールの評価器から得る(チャンネルは normalized(a,b) を持つダックタイプ)。

linear mode のスカラー評価器をまず提供する。ベジェ曲線フィット・回転評価器は
後続ステップで追加する。
"""

import math

import numpy as np
from scipy.optimize import least_squares

from mmd_toolbox.vmd import interp

# 実質ゼロ誤差の閾値(回転は unwrap/slerp の浮動小数誤差で厳密0にならないため)。
_ZERO_EPS = 1e-9


def _normalize(err, frame, tol):
    """誤差・フレームを正規化誤差へ変換する(§5.5)。

    許容0は誤差0で (0.0, None)、誤差が正で (inf, frame)。
    """
    if tol == 0.0:
        return (math.inf, frame) if err > 0.0 else (0.0, None)
    return (err / tol, frame)


def _round_half_up(x):
    """四捨五入(0.5切り上げ)。視野角は非負なので floor(x+0.5) で表せる(§3.2)。"""
    return math.floor(x + 0.5)


def _select_worst(errs, is_reversal):
    """誤差辞書から分割候補フレームを選ぶ(§5.5)。

    速度符号反転(局所極値)が区間内にあればその中で誤差最大、無ければ全内部の誤差最大。
    同点は先頭(小さいフレーム)。errs が空または最大が実質0なら None を返す。
    """
    if not errs:
        return None
    max_err = max(errs.values())
    if max_err <= _ZERO_EPS:
        return None
    reversals = [f for f in errs if is_reversal(f)]
    candidates = reversals if reversals else list(errs)
    return max(candidates, key=lambda f: (errs[f], -f))


def _bezier_axis_pred(a0, a1, a, b, sample_fn):
    """1軸の内部フレーム予測値を返す(ベジェ近似)。端点同値は平坦(a0固定)。

    sample_fn(frame) は当該軸のサンプル値。戻り値は {frame: 予測値}。
    """
    span = b - a
    internal = range(a + 1, b)
    denom = a1 - a0
    if abs(denom) <= 1e-9:
        return {f: a0 for f in internal}
    xs = [(f - a) / span for f in internal]
    ys = [(sample_fn(f) - a0) / denom for f in internal]
    cp, _ = fit_bezier_curve(xs, ys)
    return {f: a0 + denom * interp._solve_factor(*cp, (f - a) / span) for f in internal}


class LinearScalarChannel:
    """1スカラーチャンネルを評価する(§5.2, §7.2)。

    values[i] はフレーム frame_start + i のサンプル値。mode="linear" は両端を結ぶ直線で、
    mode="bezier" は1本のベジェ曲線で内部フレームを予測し、元サンプルとの最大絶対誤差を測る。
    分割候補フレームは§5.5(速度符号反転優先)。
    """

    def __init__(self, frame_start, values, tol, mode="linear"):
        self.frame_start = frame_start
        self.values = list(values)
        self.tol = float(tol)
        self.mode = mode

    def _value(self, frame):
        return self.values[frame - self.frame_start]

    def residual(self, a, b):
        if self.mode == "bezier":
            pred = _bezier_axis_pred(self._value(a), self._value(b), a, b, self._value)
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
        """frame で速度の符号が反転する(局所極値・切り返し)か。"""
        d_prev = self._value(frame) - self._value(frame - 1)
        d_next = self._value(frame + 1) - self._value(frame)
        return d_prev * d_next < 0.0

    def normalized(self, a, b):
        """正規化誤差(誤差/許容)と最大誤差フレームを返す(§5.5)。

        許容0のチャンネルは、誤差0なら (0.0, None)、誤差が正なら (inf, frame)。
        """
        err, frame = self.residual(a, b)
        return _normalize(err, frame, self.tol)


class EuclideanVectorChannel:
    """カメラ中心位置などのベクトルチャンネル(§4.2, §7.2)。

    各軸を線形補間し、採否・分割はサンプルベクトルとのユークリッド距離で測る。
    分割候補は最大ユークリッド誤差フレーム(§5.1 の基本)。
    """

    def __init__(self, frame_start, vectors, tol, mode="linear"):
        self.frame_start = frame_start
        self.vectors = [tuple(float(c) for c in v) for v in vectors]
        self.tol = float(tol)
        self.mode = mode

    def _vec(self, frame):
        return self.vectors[frame - self.frame_start]

    def residual(self, a, b):
        va = self._vec(a)
        vb = self._vec(b)
        span = b - a
        internal = list(range(a + 1, b))
        if not internal:
            return (0.0, None)
        if self.mode == "bezier":
            # 各軸を個別にベジェ近似し(§4.2)、採否はユークリッド距離(§7.2)。
            axis_pred = [
                _bezier_axis_pred(va[i], vb[i], a, b, lambda f, i=i: self._vec(f)[i])
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


class FovChannel:
    """視野角チャンネル(§4.2, §7.2)。

    出力は整数度保存のため、線形補間値を四捨五入した整数で再評価し、元サンプルとの
    差(丸めを含む総誤差)を測る。許容は §2.4 で0.5度以上に制限される。
    分割候補は速度符号反転(局所極値)を優先し、無ければ最大誤差フレーム(§5.5)。
    """

    def __init__(self, frame_start, values, tol, mode="linear"):
        self.frame_start = frame_start
        self.values = [float(v) for v in values]
        self.tol = float(tol)
        self.mode = mode

    def _value(self, frame):
        return self.values[frame - self.frame_start]

    def residual(self, a, b):
        if self.mode == "bezier":
            pred = _bezier_axis_pred(self._value(a), self._value(b), a, b, self._value)
        else:
            va, vb, span = self._value(a), self._value(b), b - a
            pred = {f: va + (vb - va) * (f - a) / span for f in range(a + 1, b)}
        # 出力は整数度保存。丸めを含む総誤差で測る(§7.2)。
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


# ---------------------------------------------------------------------------
# 回転ユーティリティ(quaternion)
# ---------------------------------------------------------------------------


def _quat_dot(a, b):
    return sum(x * y for x, y in zip(a, b))


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
    """同一半球前提の球面線形補間(端点は正規化・整列済みを渡す)。"""
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
    """2つの単位quaternion間の角度距離(度)。符号不変(|dot|)。"""
    d = min(1.0, abs(_quat_dot(a, b)))
    return math.degrees(2.0 * math.acos(d))


class CameraRotationChannel:
    """カメラ回転(3軸Euler共通曲線、§5.3, §7.2)。

    各軸を線形補間で評価し、誤差は unwrap 後の軸別角度誤差(度)の最大。360度境界の
    ラップは __init__ の軸別 unwrap で除去する。分割候補は軸別速度反転を優先(§5.5)。
    """

    def __init__(self, frame_start, eulers, tol, mode="linear"):
        self.frame_start = frame_start
        arr = np.asarray(eulers, dtype=float)
        self.eulers = np.column_stack([np.unwrap(arr[:, i]) for i in range(3)])
        self.tol = float(tol)
        self.mode = mode

    def _euler(self, frame):
        return self.eulers[frame - self.frame_start]

    def residual(self, a, b):
        ea = self._euler(a)
        eb = self._euler(b)
        span = b - a
        internal = range(a + 1, b)
        if self.mode == "bezier":
            # 3軸が1本の共通係数曲線 y(x)∈[0,1] を共有する(§5.3)。各軸の予測は
            # ea[i]+(eb[i]-ea[i])*y。軸別角度誤差(度)の二乗和を最小化して曲線を合わせ、
            # 量子化後の曲線で誤差を再評価する。
            def _resid_at(coeff):
                out = []
                for f in internal:
                    y = coeff((f - a) / span)
                    ef = self._euler(f)
                    out.extend(
                        math.degrees(ef[i] - (ea[i] + (eb[i] - ea[i]) * y))
                        for i in range(3)
                    )
                return out

            cp = _fit_coeff_curve([(f - a) / span for f in internal], _resid_at)
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
        # 微小ジッタを反転と誤認しないよう、両側の速度が有意な軸のみで判定する。
        for i in range(3):
            d_prev = cur[i] - prev[i]
            d_next = nxt[i] - cur[i]
            if abs(d_prev) > _ZERO_EPS and abs(d_next) > _ZERO_EPS and d_prev * d_next < 0.0:
                return True
        return False

    def normalized(self, a, b):
        err, frame = self.residual(a, b)
        return _normalize(err, frame, self.tol)


class BoneRotationChannel:
    """ボーン回転(quaternion slerp、§5.3, §7.2)。

    端点 quaternion の slerp(線形係数)で予測し、サンプルとの角度距離(度)で誤差を測る。
    __init__ で正規化と同一半球整列を行う。分割候補は回転方向反転(相対回転軸の符号
    反転)を優先する(§5.5)。
    """

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

    def _q(self, frame):
        return self.quats[frame - self.frame_start]

    def residual(self, a, b):
        q0 = self._q(a)
        q1 = self._q(b)
        span = b - a
        internal = range(a + 1, b)
        if self.mode == "bezier":
            # slerp 係数 y(x)∈[0,1] を1本のベジェ曲線で表す(§5.3)。予測は
            # slerp(q0,q1,y)。角度距離(度)の二乗和を最小化して係数曲線を合わせ、
            # 量子化後の曲線で誤差を再評価する。
            def _resid_at(coeff):
                return [
                    _quat_angle_deg(self._q(f), _quat_slerp(q0, q1, coeff((f - a) / span)))
                    for f in internal
                ]

            cp = _fit_coeff_curve([(f - a) / span for f in internal], _resid_at)
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
        # 微小回転は軸が不定なので、両側の回転軸が有意な場合のみ符号反転で判定する。
        n0 = math.sqrt(sum(c * c for c in ax0))
        n1 = math.sqrt(sum(c * c for c in ax1))
        if n0 <= _ZERO_EPS or n1 <= _ZERO_EPS:
            return False
        return sum(ax0[i] * ax1[i] for i in range(3)) < 0.0

    def normalized(self, a, b):
        err, frame = self.residual(a, b)
        return _normalize(err, frame, self.tol)


# ---------------------------------------------------------------------------
# スカラー ベジェ曲線フィット(§5.2, §5.4)
# ---------------------------------------------------------------------------

# 制御点探索の初期値(正規化 [0,1] の (x1,y1,x2,y2))。§5.4 の固定順:
# 線形 / ease-in / ease-out / ease-in-out。先勝ち選択のため順序を仕様に合わせる。
_BEZIER_INITS = (
    (20.0 / 127, 20.0 / 127, 107.0 / 127, 107.0 / 127),  # 線形
    (0.42, 0.0, 1.0, 1.0),   # ease-in
    (0.0, 0.0, 0.58, 1.0),   # ease-out
    (0.42, 0.0, 0.58, 1.0),  # ease-in-out
)
_BEZIER_LINEAR_CP = (20, 20, 107, 107)


def _bez(s, c1, c2):
    u = 1.0 - s
    return 3 * u * u * s * c1 + 3 * u * s * s * c2 + s * s * s


def _bezier_y_at(px1, py1, px2, py2, x):
    """正規化制御点(px* in [0,1])・正規化時間 x で y を返す(連続版。最適化用)。

    interp._solve_factor と同じく X(s)=x をニュートン法+二分法で解く。
    """
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


def fit_bezier_curve(xs, ys):
    """正規化サンプル (xs, ys) に VMD補間曲線をフィットする(§5.2, §5.4)。

    制御点 (x1,y1,x2,y2) を 0..127 整数に量子化して返し、最大絶対誤差は量子化後の曲線を
    interp._solve_factor で再評価して測る(正規化y単位)。内部点が無ければ線形・誤差0。
    最適化は x2 = x1 + (1-x1)*t の再パラメータ化で全変数をボックス境界 [0,1] に収め、
    X単調(x1<=x2)を保証する。複数初期値を決定論的に試して最良を採る。
    """
    xs = list(xs)
    ys = list(ys)
    if not xs:
        return (_BEZIER_LINEAR_CP, 0.0)

    def residual(v):
        x1, t, y1, y2 = v
        x2 = x1 + (1.0 - x1) * t
        return [_bezier_y_at(x1, y1, x2, y2, x) - y for x, y in zip(xs, ys)]

    best = None
    best_cost = math.inf
    for ix1, iy1, ix2, iy2 in _BEZIER_INITS:
        t0 = (ix2 - ix1) / (1.0 - ix1) if ix1 < 1.0 else 0.0
        x0 = [_clip01(ix1), _clip01(t0), _clip01(iy1), _clip01(iy2)]
        try:
            sol = least_squares(residual, x0, bounds=([0.0] * 4, [1.0] * 4))
        except Exception:
            continue
        cost = float(np.sum(np.square(residual(sol.x))))
        if cost < best_cost:
            best_cost = cost
            best = sol.x
    if best is None:
        best = [20.0 / 127, _clip01((107 - 20) / (127 - 20)), 20.0 / 127, 107.0 / 127]

    x1, t, y1, y2 = best
    x2 = x1 + (1.0 - x1) * t
    x1q = _quantize_cp(x1)
    x2q = _quantize_cp(x2)
    y1q = _quantize_cp(y1)
    y2q = _quantize_cp(y2)
    if x1q > x2q:  # 量子化後の X 単調を担保(§5.4)
        x2q = x1q
    cp = (x1q, y1q, x2q, y2q)

    max_err = max(abs(interp._solve_factor(*cp, x) - y) for x, y in zip(xs, ys))
    return (cp, max_err)


def _fit_coeff_curve(xs, resid_at):
    """共通の係数曲線 y(x)∈[0,1] をフィットし量子化制御点を返す(§5.3)。

    回転チャンネル用。fit_bezier_curve がスカラー (xs,ys) を直接合わせるのに対し、
    こちらは「曲線係数 y を介した誤差」を resid_at(coeff_fn) で受け取り最小化する
    (カメラ3軸共通・ボーン slerp 係数のように y が複数量へ非線形に効く場合)。
    coeff_fn(x) は正規化時間 x∈[0,1] に対する曲線値 y を返す。fit_bezier_curve と同じ
    再パラメータ化 x2=x1+(1-x1)*t でボックス境界に収め、複数初期値を決定論的に試す。
    内部点が無ければ線形制御点を返す。
    """
    if not xs:
        return _BEZIER_LINEAR_CP

    def residual(v):
        x1, t, y1, y2 = v
        x2 = x1 + (1.0 - x1) * t
        return resid_at(lambda x: _bezier_y_at(x1, y1, x2, y2, x))

    best = None
    best_cost = math.inf
    for ix1, iy1, ix2, iy2 in _BEZIER_INITS:
        t0 = (ix2 - ix1) / (1.0 - ix1) if ix1 < 1.0 else 0.0
        x0 = [_clip01(ix1), _clip01(t0), _clip01(iy1), _clip01(iy2)]
        try:
            sol = least_squares(residual, x0, bounds=([0.0] * 4, [1.0] * 4))
        except Exception:
            continue
        cost = float(np.sum(np.square(residual(sol.x))))
        if cost < best_cost:
            best_cost = cost
            best = sol.x
    if best is None:
        best = [20.0 / 127, _clip01((107 - 20) / (127 - 20)), 20.0 / 127, 107.0 / 127]

    x1, t, y1, y2 = best
    x2 = x1 + (1.0 - x1) * t
    x1q = _quantize_cp(x1)
    x2q = _quantize_cp(x2)
    y1q = _quantize_cp(y1)
    y2q = _quantize_cp(y2)
    if x1q > x2q:  # 量子化後の X 単調を担保(§5.4)
        x2q = x1q
    return (x1q, y1q, x2q, y2q)


def _clip01(v):
    return min(1.0, max(0.0, v))


def _quantize_cp(v):
    return min(127, max(0, _round_half_up(_clip01(v) * 127)))
