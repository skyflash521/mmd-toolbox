"""疎キー配置・30fps量子化のテスト(lipsync.md §3/§4)。

生成側が出した float 目標位置を、四捨五入(floor(x+0.5)・0.5は切り上げ)で整数フレーム化し、同一モーフ・
同一フレームへ潰れた目標値を量子化前 float が最も後ろの値へ統合して、量子化後に重複キーが出ないことを
検証する。協調調音の遷移長 T が奇数(=1)になる overlap_max=1 の境界で b±0.5 が潰れる衝突を扱う。
協調調音が T=4 になる overlap_max=4 では衝突が起きないことを回帰ガードで確認する(協調調音と同じ既知値)。
"""

import pytest

import lipsync
from lipsync import GenerationParams, MouthEvent, MouthShape


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


def test_t1_coartic_collision_resolved():
    # あ[0,10]op0.5・う[10,20]op0.5、overlap_max=1 で T=min(1, 5)=1、境界10。
    # 遷移目標は s=9.5・b=10.0・e=10.5。四捨五入(半上げ)で 9.5→10・10.0→10・10.5→11。
    # 同一(モーフ,フレーム10)へ潰れた s と b は量子化前 float が後ろの b(10.0)へ統合する。
    p = GenerationParams(coartic_overlap_max=1)
    keys = lipsync.generate_morph_keys(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 20.0, 0.5)],
        p,
    )
    # 量子化後に同一(モーフ名,フレーム)の重複キーが無い。
    assert len({(k.name_raw, k.frame) for k in keys}) == len(keys)
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 20.0, 0.5)],
        p,
    )
    assert set(env) == {"あ", "う", "お"}
    # あ: アタック後、境界10で中間口形0.25へ統合、e は半上げで11に分離し0へ。
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (10, 0.25), (11, 0.0)])
    # う: 境界10で0.25、11で0.5に達し保持、末尾リリースで0。
    _approx_envelope(env["う"], [(10, 0.25), (11, 0.5), (18, 0.5), (20, 0.0)])
    # お(うの補助0.1): 境界10で0.05、11で0.1、保持、リリースで0。
    _approx_envelope(env["お"], [(10, 0.05), (11, 0.1), (18, 0.1), (20, 0.0)])


def test_no_collision_overlap4_unchanged():
    # overlap_max=4 では T=min(4, 5)=4、窓[8,12]が整数で衝突なし。協調調音の既知値と同じ(量子化後も不変)。
    p = GenerationParams(coartic_overlap_max=4)
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 20.0, 0.5)], p
    )
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.25), (12, 0.0)])
    _approx_envelope(env["う"], [(8, 0.0), (10, 0.25), (12, 0.5), (18, 0.5), (20, 0.0)])
    _approx_envelope(env["お"], [(8, 0.0), (10, 0.05), (12, 0.1), (18, 0.1), (20, 0.0)])
