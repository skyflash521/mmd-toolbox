"""母音合成プロファイルのテスト。

純母音(子音なし/唇を動かさない子音)は主モーフ単独。子音種別 ConsonantClass で唇方向を変調する
(ROUNDED→丸めの う・お、SPREAD→横引きの い)。誇張係数・母音別倍率・保持値クランプ・合成後総量の
比例縮小が確定算法どおり適用されることを既知値で検証する。子音変調のゲインは lipsync 内の定数。
"""

import pytest

import lipsync
from lipsync import ConsonantClass, GenerationParams, MouthEvent, MouthShape


def _peak_weights(events, params=None):
    """単一区間のイベント列を変換し、モーフ名→ピーク(保持)ウェイトの辞書を返す。

    形状で各モーフは 0→保持値→保持値→0 のエンベロープになるため、各モーフの最大重み
    (=プラトーの保持値=合成重み)を取り出す。単一キーのみのときは恒等。
    """
    params = params or GenerationParams()
    keys = lipsync.generate_morph_keys(events, params)
    weights: dict[str, float] = {}
    for k in keys:
        weights[k.name] = max(weights.get(k.name, 0.0), k.weight)
    return weights


def _vowel(shape, open_amount, start=0.0, end=10.0, consonant=ConsonantClass.NONE):
    return [MouthEvent(shape, start, end, open_amount, consonant)]


def test_single_morph_vowel_a():
    # あ: プロファイル {あ:1.0}。hold=0.5、総量0.5<cap でクランプ無し。
    assert _peak_weights(_vowel(MouthShape.A, 0.5)) == pytest.approx({"あ": 0.5})


@pytest.mark.parametrize(
    "shape,main",
    [
        (MouthShape.A, "あ"),
        (MouthShape.I, "い"),
        (MouthShape.U, "う"),
        (MouthShape.E, "え"),
        (MouthShape.O, "お"),
        (MouthShape.N, "ん"),
    ],
)
def test_pure_vowel_is_main_morph_only(shape, main):
    # 純母音(子音なし)は主モーフ単独。open_amount=0.4 でどの母音も総量<cap で縮小なし。
    assert _peak_weights(_vowel(shape, 0.4)) == pytest.approx({main: 0.4})


def test_neutral_consonant_is_pure_vowel():
    # NEUTRAL(唇を動かさない子音=か/さ/た/は行 等)は合成上 NONE と同値(純母音)。
    assert _peak_weights(_vowel(MouthShape.A, 0.5, consonant=ConsonantClass.NEUTRAL)) == pytest.approx(
        {"あ": 0.5}
    )


def test_rounded_consonant_adds_u_morph():
    # ROUNDED(ふ・わ)は丸め(う方向)の補助を主モーフへ足す。あ: {あ:1.0, う:0.3}。
    # hold=0.5 → {あ:0.5, う:0.15}、総量0.65<cap で縮小なし。
    w = _peak_weights(_vowel(MouthShape.A, 0.5, consonant=ConsonantClass.ROUNDED))
    assert set(w) == {"あ", "う"}
    assert w["あ"] == pytest.approx(0.5)
    assert w["う"] == pytest.approx(0.15)


def test_spread_consonant_adds_i_morph():
    # SPREAD(し・ち・じ・拗音)は横引き(い方向)の補助を主モーフへ足す。あ: {あ:1.0, い:0.3}。
    # hold=0.5 → {あ:0.5, い:0.15}。
    w = _peak_weights(_vowel(MouthShape.A, 0.5, consonant=ConsonantClass.SPREAD))
    assert set(w) == {"あ", "い"}
    assert w["あ"] == pytest.approx(0.5)
    assert w["い"] == pytest.approx(0.15)


def test_consonant_aux_not_double_counted_on_main_morph():
    # 母音が既に補助と同方向のときは二重計上しない。ROUNDED×う・SPREAD×い は主モーフ単独のまま。
    assert _peak_weights(_vowel(MouthShape.U, 0.5, consonant=ConsonantClass.ROUNDED)) == pytest.approx(
        {"う": 0.5}
    )
    assert _peak_weights(_vowel(MouthShape.I, 0.5, consonant=ConsonantClass.SPREAD)) == pytest.approx(
        {"い": 0.5}
    )


def test_hold_clamped_to_open_cap():
    # あ: open_amount=1.0 でも hold=clamp(1.0, 0, 0.8)=0.8。
    assert _peak_weights(_vowel(MouthShape.A, 1.0)) == pytest.approx({"あ": 0.8})


def test_vowel_scale_applied_then_clamped():
    # あ: open_amount=0.5, vowel_scale[a]=2 → hold=clamp(1.0, 0, 0.8)=0.8。
    p = GenerationParams(vowel_scale=(2.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    assert _peak_weights(_vowel(MouthShape.A, 0.5), p) == pytest.approx({"あ": 0.8})


def test_total_clamp_proportional_scaling():
    # 子音変調で総量が cap を超えるとき比例縮小する。ROUNDED あ open=1.0 → hold=0.8。
    # raw={あ:0.8, う:0.24}, 総量1.04>cap → 全体に factor=0.8/1.04。
    w = _peak_weights(_vowel(MouthShape.A, 1.0, consonant=ConsonantClass.ROUNDED))
    factor = 0.8 / 1.04
    assert w["あ"] == pytest.approx(0.8 * factor)
    assert w["う"] == pytest.approx(0.24 * factor)
    assert sum(w.values()) == pytest.approx(0.8)


def test_open_cap_param_used_for_hold_clamp():
    # open_cap=0.6 を渡すと保持値クランプの上限が 0.6 になる(0.8 ハードコードでないこと)。
    # あ: open_amount=1.0 → hold=clamp(1.0, 0, 0.6)=0.6 → {あ:0.6}。
    p = GenerationParams(open_cap=0.6)
    assert _peak_weights(_vowel(MouthShape.A, 1.0), p) == pytest.approx({"あ": 0.6})


def test_open_cap_param_used_for_total_clamp():
    # open_cap=0.6 が合成後総量の比例縮小の上限にも使われる。ROUNDED あ open=1.0 →
    # hold=clamp(1.0,0,0.6)=0.6 → raw={あ:0.6, う:0.18}, 総量0.78>0.6 → factor=0.6/0.78。
    p = GenerationParams(open_cap=0.6)
    w = _peak_weights(_vowel(MouthShape.A, 1.0, consonant=ConsonantClass.ROUNDED), p)
    factor = 0.6 / 0.78
    assert w["あ"] == pytest.approx(0.6 * factor)
    assert w["う"] == pytest.approx(0.18 * factor)
    assert sum(w.values()) == pytest.approx(0.6)


def test_exaggeration_scales_consonant_aux_only():
    # 誇張は子音変調の補助重みに掛け、主モーフは不変。ROUNDED あ exaggeration=2.0 →
    # 有効 {あ:1.0, う:0.6}。hold=0.5 → {あ:0.5, う:0.3}、総量0.8=cap で縮小なし。
    p = GenerationParams(exaggeration=2.0)
    w = _peak_weights(_vowel(MouthShape.A, 0.5, consonant=ConsonantClass.ROUNDED), p)
    assert w["あ"] == pytest.approx(0.5)  # 主モーフは誇張対象外
    assert w["う"] == pytest.approx(0.3)  # 補助は誇張で 0.15→0.3


def test_exaggerated_aux_included_in_total_clamp():
    # 誇張後の補助も総量クランプに含む。ROUNDED あ exaggeration=2.0 open=1.0 → hold=0.8。
    # 有効{あ:1.0, う:0.6}, raw={あ:0.8, う:0.48}, 総量1.28>cap → factor=0.8/1.28。
    p = GenerationParams(exaggeration=2.0)
    w = _peak_weights(_vowel(MouthShape.A, 1.0, consonant=ConsonantClass.ROUNDED), p)
    factor = 0.8 / 1.28
    assert w["あ"] == pytest.approx(0.8 * factor)
    assert w["う"] == pytest.approx(0.48 * factor)
    assert sum(w.values()) == pytest.approx(0.8)


@pytest.mark.parametrize(
    "shape,index,main",
    [
        (MouthShape.A, 0, "あ"),
        (MouthShape.I, 1, "い"),
        (MouthShape.U, 2, "う"),
        (MouthShape.E, 3, "え"),
        (MouthShape.O, 4, "お"),
    ],
)
def test_vowel_scale_indexed_per_vowel(shape, index, main):
    # 母音別倍率 vowel_scale=(a,i,u,e,o,n) が母音ごとに正しい添字で hold に効く。当該母音の倍率だけ
    # 1.5、open_amount=0.3 で hold=clamp(0.45, 0, 0.8)=0.45(cap未満で縮小なし)。純母音は {主:0.45}。
    scale = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    scale[index] = 1.5
    p = GenerationParams(vowel_scale=tuple(scale))
    assert _peak_weights(_vowel(shape, 0.3), p) == pytest.approx({main: 0.45})


@pytest.mark.parametrize(
    "aperture_name,scale",
    [
        ("NONE", 1.0),
        ("FIRM_CLOSURE", 0.75),
        ("NARROW_CHANNEL", 0.85),
        ("SLIGHT_CLOSURE", 0.92),
    ],
)
def test_aperture_scale_applied_to_pure_vowel(aperture_name, scale):
    # 純母音(ConsonantClass.NONE、比例縮小非発動)で open_amount=0.5 → hold=0.5(cap未満)。
    # 最終重み = hold × aperture_scale。
    aperture = getattr(lipsync.ApertureClass, aperture_name)
    ev = MouthEvent(MouthShape.A, 0.0, 10.0, 0.5, ConsonantClass.NONE, aperture)
    assert _peak_weights([ev]) == pytest.approx({"あ": 0.5 * scale})


def test_aperture_decay_after_hold_clamp_not_before():
    # pop プリセット相当(vowel_scale[お]=1.70, open_cap=0.90)。open_amount=0.75・純母音(比例縮小
    # 非発動)・aperture_class=FIRM_CLOSURE(0.75)。hold=clamp(0.75×1.70,0,0.90)=0.90(クランプ発動)。
    # 正しい式(クランプ後に減衰を掛ける)の最終重みは 0.90×0.75=0.675。開口減衰をクランプの内側
    # (clamp(open_amount×vowel_scale×aperture_scale,0,open_cap))で先に掛ける誤り方式なら
    # clamp(0.75×1.70×0.75,0,0.90)=0.90 のまま aperture_scale に依らず減衰が消える。
    p = GenerationParams(vowel_scale=(1.0, 1.0, 1.0, 1.0, 1.70, 1.0), open_cap=0.90)
    ev = MouthEvent(
        MouthShape.O, 0.0, 10.0, 0.75, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE
    )
    assert _peak_weights([ev], p) == pytest.approx({"お": 0.675})


def test_aperture_decay_survives_total_clamp_shrink():
    # SPREAD(唇形変調)併用で hold が高く合成後総量の比例縮小が発動する条件で、異なる
    # ApertureClass の最終重みが異なる値になることを検証する回帰。
    # あ+SPREAD open_amount=1.0 → hold=0.8。raw={あ:0.8,い:0.24}、総量1.04>cap=0.8 →
    # factor=0.8/1.04。開口減衰(aperture_scale)を比例縮小より前に適用する誤った実装では、この
    # 条件で両者の最終重みが一致してしまう(aperture_scale が factor の分子・分母から相殺されるため)。
    factor = 0.8 / 1.04
    shrunk_a, shrunk_i = 0.8 * factor, 0.24 * factor
    w_none = _peak_weights(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 1.0, ConsonantClass.SPREAD, lipsync.ApertureClass.NONE)]
    )
    w_narrow = _peak_weights(
        [
            MouthEvent(
                MouthShape.A, 0.0, 10.0, 1.0, ConsonantClass.SPREAD,
                lipsync.ApertureClass.NARROW_CHANNEL,
            )
        ]
    )
    assert w_none == pytest.approx({"あ": shrunk_a, "い": shrunk_i})
    assert w_narrow == pytest.approx({"あ": shrunk_a * 0.85, "い": shrunk_i * 0.85})


def test_consonant_and_aperture_independent_composition():
    # ConsonantClass(SPREAD、唇形混合)と ApertureClass(NARROW_CHANNEL、開口減衰)を同一イベントに
    # 設定し、唇形混合(補助モーフ「い」)と開口減衰(最終重みの縮小)が両方反映されることを検証する。
    # open_amount=0.5 → hold=0.5(cap未満で比例縮小非発動)。有効プロファイル{あ:1.0,い:0.3}、
    # raw={あ:0.5,い:0.15}(総量0.65<0.8)。最終重み=raw×aperture_scale(NARROW_CHANNEL=0.85)。
    ev = MouthEvent(
        MouthShape.A, 0.0, 10.0, 0.5, ConsonantClass.SPREAD, lipsync.ApertureClass.NARROW_CHANNEL
    )
    w = _peak_weights([ev])
    assert w["あ"] == pytest.approx(0.5 * 0.85)
    assert w["い"] == pytest.approx(0.15 * 0.85)
