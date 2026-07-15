"""任意の伸び表現のテスト。

長く伸ばす母音の実効保持プラトーに、保持値を中心とした正弦波の揺らぎを極値のみ疎に乗せる。位相は
プラトー始端基準、節点は t_k = plateau_start + period·(1/4 + k/2) の厳密内側、open を正弦波で変調して
同じ合成で重みを出し、最後に量子化へ渡す。プラトー長が閾値以下のとき・揺らぎ無効のときは
揺らぎを置かない。
"""

import pytest

import lipsync
from lipsync import ConsonantClass, GenerationParams, MouthEvent, MouthShape


def _envelope(events, params=None):
    params = params or GenerationParams()
    by_morph: dict[str, list[tuple[int, float]]] = {}
    for k in lipsync.generate_morph_keys(events, params):
        by_morph.setdefault(k.name, []).append((k.frame, k.weight))
    for name in by_morph:
        by_morph[name].sort()
    return by_morph


def _approx_envelope(actual, expected):
    assert [f for f, _ in actual] == [f for f, _ in expected]
    for (_, aw), (_, ew) in zip(actual, expected):
        assert aw == pytest.approx(ew)


def test_long_vowel_vibrato_known_values():
    # あ[0,40]op0.5、既定(threshold18・amp0.05・period15・attack2・release2)。
    # プラトー[2,38]、長さ36>18。hold=0.5、amp_eff=min(0.05,0.5)=0.05。
    # 極値 t_k=2+15·(1/4+k/2)=5.75+7.5k、厳密内側(<38)の k=0..4 で sin=+1,-1,+1,-1,+1。
    # open_v=0.55/0.45/0.55/0.45/0.55、半上げ量子化で 6/13/21/28/36。公称(0,2,38,40)と衝突なし。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 40.0, 0.5)], GenerationParams())
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.5),
            (6, 0.55),
            (13, 0.45),
            (21, 0.55),
            (28, 0.45),
            (36, 0.55),
            (38, 0.5),
            (40, 0.0),
        ],
    )


def test_short_plateau_no_vibrato():
    # あ[0,20]op0.5、プラトー[2,18]長さ16<=threshold18。揺らぎを置かず公称のみ(非適用時と不変)。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 20.0, 0.5)], GenerationParams())
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (18, 0.5), (20, 0.0)])


def test_vibrato_disabled_when_amp_zero():
    # 長い母音でも vibrato_amp=0 なら揺らぎ無効。公称のみ(非適用時と不変)。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 40.0, 0.5)], GenerationParams(vibrato_amp=0.0))
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (38, 0.5), (40, 0.0)])


def test_vibrato_follows_interpolated_aperture_scale():
    # あ[0,20]op0.5(ApertureClass.NONE=1.0)・あ[20,40]op0.5(ApertureClass.FIRM_CLOSURE=0.75)を連結。
    # 開き量は両区間とも0.5で一定だが、開口減衰aperture_v(t)はmid1=10からmid2=30にかけて1.0→0.75へ
    # 線形補間される。境界の子音がFIRM_CLOSURE(NONEでない)なので谷が生成される。mora_valley_frames
    # を小さく指定し(既定値では両隣のモーラが長く谷が完全飽和して肩キーが消え、揺らぎ間引き幅も
    # period/2側でなくhw+2側が支配的になって検証が弱まるため)、谷の肩キー・揺らぎ間引きの両方を
    # 独立に検証できる形にする。
    events = [
        MouthEvent(
            MouthShape.A, 0.0, 20.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.NONE
        ),
        MouthEvent(
            MouthShape.A, 20.0, 40.0, 0.5, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
    ]
    env = _envelope(events, GenerationParams(mora_valley_frames=2.0))
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.5),  # アタック到達(先頭小区間の最終重み 0.5*1.0)
            (6, 0.55),  # 揺らぎの正極値(残存)
            (10, 0.5),  # mid1 強弱節点(先頭小区間の最終重み 0.5*1.0)
            (18, 0.45),  # 谷の左肩
            (20, 0.328125),  # 谷の中央
            (22, 0.425),  # 谷の右肩
            (30, 0.375),  # mid2 強弱節点(後方小区間の最終重み 0.5*0.75)
            (36, 0.4125),  # 揺らぎの正極値(残存)
            (38, 0.375),  # 実効リリース開始(後方小区間の最終重み 0.5*0.75)
            (40, 0.0),
        ],
    )


def test_vibrato_aperture_decay_applied_after_shrink_not_before():
    # あ[0,40]op0.6(SPREAD、ApertureClass.NARROW_CHANNEL=0.85)。有効プロファイル{あ:1.0,い:0.3}。
    # 揺らぎの山(open_v=0.65)では raw 総量 0.65*1.3=0.845 が cap(0.8)を超え比例縮小が発動するが、
    # 谷(open_v=0.55)では raw 総量 0.715<0.8 で発動しない。開口減衰を比例縮小の前に適用する誤った
    # 実装だと、山では decayed_open_v=0.65*0.85=0.5525 となり raw 総量 0.5525*1.3=0.71825<0.8 で
    # 縮小自体が発動しなくなり、下記の期待値(山:0.523077/0.156923、谷:0.4675/0.14025)とは異なる
    # (0.5525/0.16575)になる。この違いを既知値で検出する。
    events = [
        MouthEvent(MouthShape.A, 0.0, 40.0, 0.6, ConsonantClass.SPREAD, lipsync.ApertureClass.NARROW_CHANNEL)
    ]
    env = _envelope(events)
    assert set(env) == {"あ", "い"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.51),  # アタック到達(境界の最終重み 0.6*0.85、総量0.78<capで縮小非発動)
            (6, 0.5230769230769231),  # t=5.75: open_v=0.65(山)、比例縮小あり
            (13, 0.4675),  # t=13.25: open_v=0.55(谷)、比例縮小なし
            (21, 0.5230769230769231),
            (28, 0.4675),
            (36, 0.5230769230769231),
            (38, 0.51),
            (40, 0.0),
        ],
    )
    _approx_envelope(
        env["い"],
        [
            (0, 0.0),
            (2, 0.153),
            (6, 0.15692307692307692),
            (13, 0.14025),
            (21, 0.15692307692307692),
            (28, 0.14025),
            (36, 0.15692307692307692),
            (38, 0.153),
            (40, 0.0),
        ],
    )


def test_vibrato_extremum_exactly_at_half_period_margin_is_suppressed():
    # あ[0,20]op0.5(NONE)・あ[20,40]op0.5(FIRM_CLOSURE)を連結。mora_valley_framesを小さく指定し
    # (理由は前テストと同じ)、vibrato_period=24にする。正方向の間引き幅max(hw+2,period/2)は
    # period/2=12で支配され、ちょうど揺らぎの正極値(t=8,32)までの距離(12)と一致する境界値になる。
    # 仕様は`≤`(境界を含めて間引く)なので、この2つの正極値はどちらも間引かれ、揺らぎ由来のキーは
    # 1つも残らない。厳密不等号`<`で間引く誤った実装だと、t=8(重み0.55)・t=32(重み0.4125)が
    # 残ってしまい、この既知値とは異なる。
    events = [
        MouthEvent(
            MouthShape.A, 0.0, 20.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.NONE
        ),
        MouthEvent(
            MouthShape.A, 20.0, 40.0, 0.5, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
    ]
    env = _envelope(events, GenerationParams(mora_valley_frames=2.0, vibrato_period=24))
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.5),
            (10, 0.5),
            (18, 0.45),
            (20, 0.328125),
            (22, 0.425),
            (30, 0.375),
            (38, 0.375),
            (40, 0.0),
        ],
    )


def test_mora_valley_none_boundary_does_not_disturb_vibrato():
    # あ[0,20]op0.5(ApertureClass.NONE)・あ[20,40]op0.5(ApertureClass.NONE)を連結。境界(b=20)の
    # ApertureClassがNONEなので、モーラ境界の谷は一切生成されず(3点自体が出力されない)、揺らぎの
    # 極値も一切間引かれず通常どおり生成される。両小区間の保持値・ApertureClassが同一
    # (開き量0.5・aperture_scale=1.0)なので、揺らぎは公称保持値0.5で一定のプラトーとして、既存の
    # 単一区間の揺らぎ(あ[0,40]op0.5)と同じ極値(t=5.75,13.25,20.75,28.25,35.75の全5点)を持つ。
    # 中央の強弱節点(mid1=10, mid2=30)は追加されるが値は既存の保持値と同じ0.5なので、それ自体は
    # 谷でも揺らぎの片側消失でもない(NONEでは谷も間引きも一切行わないことの回帰)。
    events = [
        MouthEvent(
            MouthShape.A, 0.0, 20.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.NONE
        ),
        MouthEvent(
            MouthShape.A, 20.0, 40.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.NONE
        ),
    ]
    env = _envelope(events)
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.5),
            (6, 0.55),
            (10, 0.5),  # mid1 強弱節点(谷ではない。単なる保持値どうしの一致)
            (13, 0.45),
            (21, 0.55),
            (28, 0.45),
            (30, 0.5),  # mid2 強弱節点
            (36, 0.55),
            (38, 0.5),
            (40, 0.0),
        ],
    )
