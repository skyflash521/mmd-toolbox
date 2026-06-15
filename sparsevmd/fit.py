"""補間曲線フィット・誤差評価(sparsevmd.md §5.2, §7.2)。

本モジュールはチャンネル単位の誤差評価を担う。reduce.py は分割戦略に専念し、
各チャンネルの「区間 [a,b] を表現したときの正規化誤差と最大誤差フレーム」を
このモジュールの評価器から得る(チャンネルは normalized(a,b) を持つダックタイプ)。

linear mode のスカラー評価器をまず提供する。ベジェ曲線フィット・回転評価器は
後続ステップで追加する。
"""

import math


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
        if max_err == 0.0:
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
        if self.tol == 0.0:
            return (math.inf, frame) if err > 0.0 else (0.0, None)
        return (err / self.tol, frame)
