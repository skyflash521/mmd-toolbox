"""補間曲線フィット・誤差評価(sparsevmd.md §5.2, §7.2)。

本モジュールはチャンネル単位の誤差評価を担う。reduce.py は分割戦略に専念し、
各チャンネルの「区間 [a,b] を表現したときの正規化誤差と最大誤差フレーム」を
このモジュールの評価器から得る(チャンネルは normalized(a,b) を持つダックタイプ)。

スカラー(線形)評価器・ベジェ曲線フィット・回転評価器を提供する。
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


def _axis_curve(a0, a1, a, b, sample_fn, early_exit_err=None):
    """1軸の量子化ベジェ制御点 (x1,y1,x2,y2) を返す(§5.4)。

    端点同値(正規化不能)や内部点なしは線形制御点。sample_fn(frame) は当該軸のサンプル値。
    採否(_bezier_axis_pred)と出力(curve)が同一の制御点を使うよう、両者はこれを共有する。
    early_exit_err(正規化y単位)は fit_bezier_curve の早期終了閾値へ渡す。
    """
    span = b - a
    internal = range(a + 1, b)
    denom = a1 - a0
    if abs(denom) <= 1e-9 or not internal:
        return _BEZIER_LINEAR_CP
    xs = [(f - a) / span for f in internal]
    ys = [(sample_fn(f) - a0) / denom for f in internal]
    cp, _ = fit_bezier_curve(xs, ys, early_exit_err=early_exit_err)
    return cp


def _bezier_axis_pred(a0, a1, a, b, sample_fn, cp=None):
    """1軸の内部フレーム予測値を返す(ベジェ近似)。端点同値は平坦(a0固定)。

    sample_fn(frame) は当該軸のサンプル値。戻り値は {frame: 予測値}。
    cp を渡すと量子化済み制御点の再計算(_axis_curve)を省く(メモ化)。端点同値の
    平坦ケースは cp を使わないため、cp 有無で結果は変わらない。
    """
    span = b - a
    internal = range(a + 1, b)
    denom = a1 - a0
    if abs(denom) <= 1e-9:
        return {f: a0 for f in internal}
    if cp is None:
        cp = _axis_curve(a0, a1, a, b, sample_fn)
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
        self._cp_cache = {}  # (a, b) -> 量子化済みベジェ制御点(区間フィットのメモ化)

    def _value(self, frame):
        return self.values[frame - self.frame_start]

    def _axis_cp(self, a, b):
        """区間 [a,b] の量子化ベジェ制御点を計算しインスタンスにキャッシュする。

        早期終了閾値は許容誤差を正規化y単位へ換算した tol/|denom| を渡す(区間が許容内に
        フィットできた時点で残り初期値を打ち切る)。
        """
        cp = self._cp_cache.get((a, b))
        if cp is None:
            a0, a1 = self._value(a), self._value(b)
            denom = abs(a1 - a0)
            ee = self.tol / denom if denom > 1e-9 else None
            cp = _axis_curve(a0, a1, a, b, self._value, early_exit_err=ee)
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

    def curve(self, a, b):
        """区間 [a,b] の出力用制御点 (x1,y1,x2,y2) を返す(§5.4)。"""
        if self.mode != "bezier":
            return _BEZIER_LINEAR_CP
        return self._axis_cp(a, b)


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
        self._cp_cache = {}  # (a, b, axis) -> 量子化済みベジェ制御点(区間フィットのメモ化)

    def _vec(self, frame):
        return self.vectors[frame - self.frame_start]

    def _axis_cp(self, a, b, i):
        """区間 [a,b]・軸 i の量子化ベジェ制御点を計算しインスタンスにキャッシュする。

        採否はユークリッド距離(3軸合成)が許容内かで判定するため、軸別の早期終了閾値は
        各軸が tol/√3 以内なら合成 <= tol になるよう tol/(√3·|denom_i|) を渡す。
        """
        cp = self._cp_cache.get((a, b, i))
        if cp is None:
            va, vb = self._vec(a), self._vec(b)
            denom = abs(vb[i] - va[i])
            ee = self.tol / (math.sqrt(3.0) * denom) if denom > 1e-9 else None
            cp = _axis_curve(va[i], vb[i], a, b, lambda f: self._vec(f)[i], early_exit_err=ee)
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
            # 各軸を個別にベジェ近似し(§4.2)、採否はユークリッド距離(§7.2)。
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

    def curve(self, a, b):
        """各軸の出力用制御点を (cp_x, cp_y, cp_z) で返す(§4.2, §5.4)。"""
        if self.mode != "bezier":
            return (_BEZIER_LINEAR_CP, _BEZIER_LINEAR_CP, _BEZIER_LINEAR_CP)
        return tuple(self._axis_cp(a, b, i) for i in range(3))


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
        self._cp_cache = {}  # (a, b) -> 量子化済みベジェ制御点(区間フィットのメモ化)

    def _value(self, frame):
        return self.values[frame - self.frame_start]

    def _axis_cp(self, a, b):
        """区間 [a,b] の量子化ベジェ制御点を計算しインスタンスにキャッシュする。

        早期終了閾値は許容(度)を正規化y単位へ換算した tol/|denom| を渡す。FOV は出力時に
        整数度へ丸めるため丸め分(最大0.5度)の上振れがありうるが、採否は丸め込みの residual で
        測られ、超過すれば reduce 側で分割されるためフィット品質は担保される。
        """
        cp = self._cp_cache.get((a, b))
        if cp is None:
            a0, a1 = self._value(a), self._value(b)
            denom = abs(a1 - a0)
            ee = self.tol / denom if denom > 1e-9 else None
            cp = _axis_curve(a0, a1, a, b, self._value, early_exit_err=ee)
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

    def curve(self, a, b):
        """区間 [a,b] の出力用制御点 (x1,y1,x2,y2) を返す(§5.4)。"""
        if self.mode != "bezier":
            return _BEZIER_LINEAR_CP
        return self._axis_cp(a, b)


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
    """2つの単位quaternion間の角度距離(度)。符号不変。

    相対回転 r=b·conj(a) の (||虚部||, |実部|) から 2·atan2(||xyz||,|w|) で測る。
    2·acos(|dot|) は dot≈1(微小角)で悪条件になり libm 差で偽差が出るが、atan2 形は
    全域で安定。|w| を取ることで q と -q(同一回転)は厳密に角度0、最短弧(≤180度)を返す。
    """
    r = _quat_mul(b, _quat_conj(a))
    v = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    return math.degrees(2.0 * math.atan2(v, abs(r[3])))


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
        self._cp_cache = {}  # (a, b) -> 共通係数曲線の量子化制御点(区間フィットのメモ化)

    def _euler(self, frame):
        return self.eulers[frame - self.frame_start]

    def residual(self, a, b):
        ea = self._euler(a)
        eb = self._euler(b)
        span = b - a
        internal = range(a + 1, b)
        if self.mode == "bezier":
            # 3軸が1本の共通係数曲線を共有する(§5.3)。出力と同一の量子化制御点
            # (self.curve)で各軸予測 ea[i]+(eb[i]-ea[i])*y を再評価し誤差を測る。
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

    def curve(self, a, b):
        """3軸共通の出力用制御点 (x1,y1,x2,y2) を返す(§5.3, §5.4)。

        各軸予測 ea[i]+(eb[i]-ea[i])*y(x) の軸別角度誤差(度)の二乗和を最小化して
        共通係数曲線 y(x) をフィットする。内部点なしは線形。
        """
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

        # 早期終了閾値は回転許容(度)。係数曲線の残差は度単位なので直接渡す。
        cp = _fit_coeff_curve([(f - a) / span for f in internal], _resid_at, early_exit_err=self.tol)
        self._cp_cache[(a, b)] = cp
        return cp


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
        self._cp_cache = {}  # (a, b) -> slerp 係数曲線の量子化制御点(区間フィットのメモ化)

    def _q(self, frame):
        return self.quats[frame - self.frame_start]

    def residual(self, a, b):
        q0 = self._q(a)
        q1 = self._q(b)
        span = b - a
        internal = range(a + 1, b)
        if self.mode == "bezier":
            # slerp 係数を1本のベジェ曲線で表す(§5.3)。出力と同一の量子化制御点
            # (self.curve)で予測 slerp(q0,q1,y) を再評価し角度距離(度)で誤差を測る。
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
        # 微小回転は軸が不定なので、両側の回転軸が有意な場合のみ符号反転で判定する。
        n0 = math.sqrt(sum(c * c for c in ax0))
        n1 = math.sqrt(sum(c * c for c in ax1))
        if n0 <= _ZERO_EPS or n1 <= _ZERO_EPS:
            return False
        return sum(ax0[i] * ax1[i] for i in range(3)) < 0.0

    def normalized(self, a, b):
        err, frame = self.residual(a, b)
        return _normalize(err, frame, self.tol)

    def curve(self, a, b):
        """slerp 係数の出力用制御点 (x1,y1,x2,y2) を返す(§5.3, §5.4)。

        予測 slerp(q0,q1,y(x)) とサンプルの角度距離(度)の二乗和を最小化して係数曲線
        y(x) をフィットする。内部点なしは線形。
        """
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

        # 早期終了閾値は回転許容(度)。係数曲線の残差は度単位なので直接渡す。
        cp = _fit_coeff_curve([(f - a) / span for f in internal], _resid_at, early_exit_err=self.tol)
        self._cp_cache[(a, b)] = cp
        return cp


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


def _bezier_y_at_many(px1, py1, px2, py2, xs):
    """_bezier_y_at の numpy 配列一括版(§5.2 のベクトル化)。

    正規化制御点(px* in [0,1])と正規化時間配列 xs に対し、各要素で X(s)=x を
    単点版 _bezier_y_at と同じニュートン法+二分法フォールバックで解き y を返す。
    戻り値は xs と同形の ndarray。内部点ごとの Python 呼び出しを消して
    fit_bezier_curve の残差評価を高速化する(単点版と同一アルゴリズムなので結果は一致)。
    """
    xs = np.asarray(xs, dtype=float)
    out = np.empty_like(xs)
    out[xs <= 0.0] = 0.0
    out[xs >= 1.0] = 1.0
    interior = (xs > 0.0) & (xs < 1.0)
    if not np.any(interior):
        return out

    xi = xs[interior]
    s = xi.copy()
    converged = np.zeros(xi.shape, dtype=bool)
    dead = np.zeros(xi.shape, dtype=bool)  # 発散・微小導関数で Newton 打ち切り → 二分法へ
    for _ in range(20):
        active = ~converged & ~dead
        if not np.any(active):
            break
        err = _bez(s, px1, px2) - xi
        converged |= active & (np.abs(err) < 1e-9)
        active &= ~converged
        u = 1.0 - s
        d = 3.0 * (px1 * u * u + 2.0 * (px2 - px1) * u * s + (1.0 - px2) * s * s)
        dead |= active & (d <= 1e-12)
        active &= ~dead
        s_cand = s - err / np.where(d > 1e-12, d, 1.0)
        diverged = active & ((s_cand < 0.0) | (s_cand > 1.0))
        dead |= diverged
        s = np.where(active & ~diverged, s_cand, s)

    need_bis = ~converged
    if np.any(need_bis):
        lo = np.zeros(xi.shape)
        hi = np.ones(xi.shape)
        for _ in range(60):
            mid = (lo + hi) / 2.0
            go_lo = _bez(mid, px1, px2) < xi
            lo = np.where(need_bis & go_lo, mid, lo)
            hi = np.where(need_bis & ~go_lo, mid, hi)
        s = np.where(need_bis, (lo + hi) / 2.0, s)

    out[interior] = _bez(s, py1, py2)
    return out


def _quantize_solution(sol_x):
    """最適化解 (x1,t,y1,y2) を 0..127 整数の制御点 (x1,y1,x2,y2) へ量子化する(§5.4)。"""
    x1, t, y1, y2 = sol_x
    x2 = x1 + (1.0 - x1) * t
    x1q = _quantize_cp(x1)
    x2q = _quantize_cp(x2)
    y1q = _quantize_cp(y1)
    y2q = _quantize_cp(y2)
    if x1q > x2q:  # 量子化後の X 単調を担保(§5.4)
        x2q = x1q
    return (x1q, y1q, x2q, y2q)


def fit_bezier_curve(xs, ys, early_exit_err=None):
    """正規化サンプル (xs, ys) に VMD補間曲線をフィットする(§5.2, §5.4)。

    制御点 (x1,y1,x2,y2) を 0..127 整数に量子化して返し、最大絶対誤差は量子化後の曲線を
    interp._solve_factor で再評価して測る(正規化y単位)。内部点が無ければ線形・誤差0。
    最適化は x2 = x1 + (1-x1)*t の再パラメータ化で全変数をボックス境界 [0,1] に収め、
    X単調(x1<=x2)を保証する。複数初期値を決定論的に試して最良(コスト最小)を採る。

    early_exit_err(正規化y単位)を渡すと、各初期値の評価後に現在の最良の量子化誤差がそれ以下
    なら残りの初期値を試さず打ち切る。閾値は呼び出し側(チャンネル)が許容誤差から算出して渡す
    (tol / |denom|): 区間が許容内にフィットできた時点で打ち切るため、採否(誤差 <= 許容)は
    変わらず、出力は全初期値試行と許容内一致になる。
    None なら早期終了せず全初期値を試す。
    """
    xs = list(xs)
    ys = list(ys)
    if not xs:
        return (_BEZIER_LINEAR_CP, 0.0)

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)

    def residual(v):
        x1, t, y1, y2 = v
        x2 = x1 + (1.0 - x1) * t
        return _bezier_y_at_many(x1, y1, x2, y2, xs_arr) - ys_arr

    def quantized_err(cp):
        return max(abs(interp._solve_factor(*cp, x) - y) for x, y in zip(xs, ys))

    best_cost = math.inf
    best_cp = None
    best_err = None
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
            best_cp = _quantize_solution(sol.x)
            best_err = quantized_err(best_cp)
            if early_exit_err is not None and best_err <= early_exit_err:
                return (best_cp, best_err)  # 許容内にフィット済み。残り初期値は不要
    if best_cp is None:
        best_cp = _quantize_solution(
            [20.0 / 127, _clip01((107 - 20) / (127 - 20)), 20.0 / 127, 107.0 / 127]
        )
        best_err = quantized_err(best_cp)
    return (best_cp, best_err)


def _fit_coeff_curve(xs, resid_at, early_exit_err=None):
    """共通の係数曲線 y(x)∈[0,1] をフィットし量子化制御点を返す(§5.3)。

    回転チャンネル用。fit_bezier_curve がスカラー (xs,ys) を直接合わせるのに対し、
    こちらは「曲線係数 y を介した誤差」を resid_at(coeff_fn) で受け取り最小化する
    (カメラ3軸共通・ボーン slerp 係数のように y が複数量へ非線形に効く場合)。
    coeff_fn(x) は正規化時間 x∈[0,1] に対する曲線値 y を返す。fit_bezier_curve と同じ
    再パラメータ化 x2=x1+(1-x1)*t でボックス境界に収め、複数初期値を決定論的に試す。
    内部点が無ければ線形制御点を返す。

    early_exit_err を渡すと、現在の最良の量子化後残差(resid_at の単位=回転では度)の最大値が
    それ以下なら残りの初期値を試さず打ち切る。閾値は呼び出し側が許容誤差(度)から渡す。
    None なら早期終了せず全初期値を試す。
    """
    if not xs:
        return _BEZIER_LINEAR_CP

    def residual(v):
        x1, t, y1, y2 = v
        x2 = x1 + (1.0 - x1) * t
        return resid_at(lambda x: _bezier_y_at(x1, y1, x2, y2, x))

    def quantized_err(cp):
        res = resid_at(lambda x: interp._solve_factor(*cp, x))
        return max((abs(r) for r in res), default=0.0)

    best_cost = math.inf
    best_cp = None
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
            best_cp = _quantize_solution(sol.x)
            if early_exit_err is not None and quantized_err(best_cp) <= early_exit_err:
                return best_cp  # 許容内にフィット済み。残り初期値は不要
    if best_cp is None:
        best_cp = _quantize_solution(
            [20.0 / 127, _clip01((107 - 20) / (127 - 20)), 20.0 / 127, 107.0 / 127]
        )
    return best_cp


def _clip01(v):
    return min(1.0, max(0.0, v))


def _quantize_cp(v):
    return min(127, max(0, _round_half_up(_clip01(v) * 127)))
