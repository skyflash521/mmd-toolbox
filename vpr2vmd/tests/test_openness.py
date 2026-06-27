"""ベロシティ→開き量写像のテスト(vpr2vmd.md §3、実装計画 §5.3)。

写像式 open = lo + (hi - lo) * (velocity/127) ** gamma を [lo, hi] にクランプし open_max を
上限とする。全ノートのベロシティが一様(強弱差 0)のときは写像式が定数に退化するため、全モーラへ
既定開き量 default_open を用いる。
"""

import pytest

from vpr2vmd import openness

_PENDING = "impl pending: P-3 開き量写像"

# pop プリセット相当のレンジ(出発点)。default_open は中央 (lo+hi)/2。
_LO, _HI = 0.30, 0.75
_OPEN_MAX = 0.90
_DEFAULT = (_LO + _HI) / 2  # 0.525


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_velocity_max_maps_to_high():
    # velocity=127 → n=1 → open = hi(open_max が hi 以上なら hi のまま)。
    assert openness.velocity_to_open(127, lo=_LO, hi=_HI, open_max=_OPEN_MAX) == pytest.approx(_HI)


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_velocity_zero_maps_to_low():
    # velocity=0 → n=0 → open = lo。
    assert openness.velocity_to_open(0, lo=_LO, hi=_HI, open_max=_OPEN_MAX) == pytest.approx(_LO)


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_velocity_mid_follows_power_curve():
    # 累乗カーブ: open = lo + (hi-lo) * (64/127)**0.6。
    n = 64 / 127
    expected = _LO + (_HI - _LO) * (n ** openness.GAMMA_DEFAULT)
    assert openness.velocity_to_open(64, lo=_LO, hi=_HI, open_max=_OPEN_MAX) == pytest.approx(expected)


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_velocity_gamma_parameter_changes_curve():
    # gamma を変えると写像が変わる(実装が gamma 引数を無視し固定 0.6 を使う誤りを落とす)。
    n = 64 / 127
    linear = _LO + (_HI - _LO) * n  # gamma=1.0 なら累乗が恒等で線形
    assert openness.velocity_to_open(
        64, lo=_LO, hi=_HI, open_max=_OPEN_MAX, gamma=1.0
    ) == pytest.approx(linear)
    # 既定 gamma(0.6)は弱音側を持ち上げるので線形とは異なる値になる。
    assert openness.velocity_to_open(
        64, lo=_LO, hi=_HI, open_max=_OPEN_MAX
    ) != pytest.approx(linear)


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_velocity_monotonic_increasing():
    vals = [openness.velocity_to_open(v, lo=_LO, hi=_HI, open_max=_OPEN_MAX) for v in (0, 32, 64, 96, 127)]
    assert vals == sorted(vals)
    assert vals[0] < vals[-1]


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_open_max_caps_below_high():
    # open_max が hi より小さいとき、強音側は open_max で頭打ちになる。
    assert openness.velocity_to_open(127, lo=_LO, hi=_HI, open_max=0.50) == pytest.approx(0.50)


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_open_amounts_empty():
    assert openness.open_amounts([], lo=_LO, hi=_HI, open_max=_OPEN_MAX, default_open=_DEFAULT) == []


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_open_amounts_uniform_uses_default():
    # 全ベロシティ一様(強弱差 0)→ 写像式が定数に退化するため全モーラへ既定開き量。
    result = openness.open_amounts(
        [64, 64, 64], lo=_LO, hi=_HI, open_max=_OPEN_MAX, default_open=_DEFAULT
    )
    assert result == [pytest.approx(_DEFAULT)] * 3


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_open_amounts_single_note_is_uniform_default():
    # 単一音符は強弱差を持たないので既定開き量。
    assert openness.open_amounts(
        [80], lo=_LO, hi=_HI, open_max=_OPEN_MAX, default_open=_DEFAULT
    ) == [pytest.approx(_DEFAULT)]


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_open_amounts_default_not_capped_by_open_max():
    # uniform 経路の既定開き量は vpr2vmd 側で open_max クランプしない(写像式側だけが open_max で
    # 頭打ち。既定開き量の最終上限は lipsync の open_cap が担う)。default_open=0.8 > open_max=0.5
    # でも 0.8 をそのまま返す。
    result = openness.open_amounts(
        [70, 70], lo=_LO, hi=_HI, open_max=0.50, default_open=0.80
    )
    assert result == [pytest.approx(0.80), pytest.approx(0.80)]


@pytest.mark.xfail(reason=_PENDING, strict=True)
def test_open_amounts_varying_maps_each_velocity():
    # 強弱差があれば各ベロシティを写像式で個別に開き量へ写す。
    result = openness.open_amounts(
        [0, 127], lo=_LO, hi=_HI, open_max=_OPEN_MAX, default_open=_DEFAULT
    )
    assert result == [pytest.approx(_LO), pytest.approx(_HI)]
