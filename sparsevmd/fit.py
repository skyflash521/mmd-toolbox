"""補間曲線フィット・誤差評価(sparsevmd.md §5.2, §7.2)。

本モジュールはチャンネル単位の誤差評価を担う。reduce.py は分割戦略に専念し、
各チャンネルの「区間 [a,b] を表現したときの正規化誤差と最大誤差フレーム」を
このモジュールの評価器から得る(チャンネルは normalized(a,b) を持つダックタイプ)。

linear mode のスカラー評価器をまず提供する。ベジェ曲線フィット・回転評価器は
後続ステップで追加する。
"""

import math

import numpy as np

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


class LinearScalarChannel:
    """1スカラーチャンネルを線形補間で評価する(§5.2 の線形ケース, §7.2)。

    values[i] はフレーム frame_start + i のサンプル値。区間 [a,b] は両端の値を結ぶ
    線形補間で内部フレームを予測し、元サンプルとの最大絶対誤差を測る。
    """

    def __init__(self, frame_start, values, tol):
        self.frame_start = frame_start
        self.values = list(values)
        self.tol = float(tol)

    def _value(self, frame):
        return self.values[frame - self.frame_start]

    def residual(self, a, b):
        """区間 [a,b] の線形補間による (最大絶対誤差, 分割候補フレーム) を返す。

        内部フレーム a<f<b のみを評価する。内部が無い(隣接)区間は (0.0, None)。
        最大絶対誤差は採否判定(§7.2)に用いる真の最大値。分割候補フレームは§5.5に従い、
        速度符号反転(局所極値・切り返し)が区間内にあればその中で誤差最大のものを優先し、
        無ければ全内部フレームの誤差最大フレームとする。
        """
        va = self._value(a)
        vb = self._value(b)
        span = b - a
        errs = {}
        for f in range(a + 1, b):
            pred = va + (vb - va) * (f - a) / span
            errs[f] = abs(self._value(f) - pred)
        if not errs:
            return (0.0, None)
        max_err = max(errs.values())
        if max_err <= _ZERO_EPS:
            return (0.0, None)
        # 速度符号反転(局所極値)を優先候補にする(§5.5)。無ければ全内部。
        reversals = [f for f in errs if self._is_reversal(f)]
        candidates = reversals if reversals else list(errs)
        # 候補のうち誤差最大、同点は先頭(小さいフレーム)を選ぶ。
        worst = max(candidates, key=lambda f: (errs[f], -f))
        return (max_err, worst)

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

    def __init__(self, frame_start, vectors, tol):
        self.frame_start = frame_start
        self.vectors = [tuple(float(c) for c in v) for v in vectors]
        self.tol = float(tol)

    def _vec(self, frame):
        return self.vectors[frame - self.frame_start]

    def residual(self, a, b):
        va = self._vec(a)
        vb = self._vec(b)
        span = b - a
        errs = {}
        for f in range(a + 1, b):
            t = (f - a) / span
            pred = tuple(va[i] + (vb[i] - va[i]) * t for i in range(3))
            errs[f] = math.dist(self._vec(f), pred)
        if not errs:
            return (0.0, None)
        max_err = max(errs.values())
        if max_err <= _ZERO_EPS:
            return (0.0, None)
        # いずれかの軸で速度が反転する切り返し点を優先候補にする(§5.5)。
        reversals = [f for f in errs if self._is_reversal(f)]
        candidates = reversals if reversals else list(errs)
        worst = max(candidates, key=lambda f: (errs[f], -f))
        return (max_err, worst)

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

    def __init__(self, frame_start, values, tol):
        self.frame_start = frame_start
        self.values = [float(v) for v in values]
        self.tol = float(tol)

    def _value(self, frame):
        return self.values[frame - self.frame_start]

    def residual(self, a, b):
        va = self._value(a)
        vb = self._value(b)
        span = b - a
        errs = {}
        for f in range(a + 1, b):
            pred = va + (vb - va) * (f - a) / span
            errs[f] = abs(_round_half_up(pred) - self._value(f))
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

    def __init__(self, frame_start, eulers, tol):
        self.frame_start = frame_start
        arr = np.asarray(eulers, dtype=float)
        self.eulers = np.column_stack([np.unwrap(arr[:, i]) for i in range(3)])
        self.tol = float(tol)

    def _euler(self, frame):
        return self.eulers[frame - self.frame_start]

    def residual(self, a, b):
        ea = self._euler(a)
        eb = self._euler(b)
        span = b - a
        errs = {}
        for f in range(a + 1, b):
            t = (f - a) / span
            ef = self._euler(f)
            errs[f] = max(
                abs(math.degrees(ef[i] - (ea[i] + (eb[i] - ea[i]) * t))) for i in range(3)
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

    def __init__(self, frame_start, quats, tol):
        self.frame_start = frame_start
        aligned = []
        for q in quats:
            qn = _quat_normalize(q)
            if aligned and _quat_dot(qn, aligned[-1]) < 0.0:
                qn = tuple(-c for c in qn)
            aligned.append(qn)
        self.quats = aligned
        self.tol = float(tol)

    def _q(self, frame):
        return self.quats[frame - self.frame_start]

    def residual(self, a, b):
        q0 = self._q(a)
        q1 = self._q(b)
        span = b - a
        errs = {}
        for f in range(a + 1, b):
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
