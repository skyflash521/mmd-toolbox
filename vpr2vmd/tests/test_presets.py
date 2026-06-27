"""口パクスタイルプリセット解決のテスト(vpr2vmd.md §4)。

`presets.resolve` はスタイル名と任意の上書き(--open-max/--default-open)から、開き量写像の
パラメータ(OpennessParams)と lipsync の生成パラメータ(GenerationParams)を返す。プリセット
具体値は presets モジュールが出発点値として持つ。default_open 既定は開き量レンジの中央 (lo+hi)/2、
--open-max は写像上限かつ lipsync の open_cap。
"""

import pytest

from lipsync import GenerationParams

from vpr2vmd import presets

_STYLES = ("pop", "ballad", "powerful", "whisper", "rap")


def test_all_styles_resolve_to_param_pair():
    for style in _STYLES:
        op, gen = presets.resolve(style)
        assert isinstance(op, presets.OpennessParams)
        assert isinstance(gen, GenerationParams)
        # gamma は全スタイル共通の初期値 0.6。pop 以外の設定漏れを落とす。
        assert op.gamma == 0.6
        # --open-max 既定はそのスタイルの写像上限であり lipsync の open_cap にも渡る。
        assert gen.open_cap == op.open_max


def test_pop_openness_params():
    op, _ = presets.resolve("pop")
    assert (op.lo, op.hi, op.open_max, op.gamma) == (0.30, 0.75, 0.90, 0.6)
    assert op.default_open == pytest.approx((0.30 + 0.75) / 2)


def test_pop_generation_params():
    _, gen = presets.resolve("pop")
    assert gen.open_cap == 0.90
    assert gen.attack_frames == 2
    assert gen.release_frames == 2
    assert gen.coartic_overlap_max == 2
    assert gen.anticipation_frames == 1
    assert gen.min_hold_frames == 3
    assert gen.exaggeration == 1.0


def test_ballad_values():
    op, gen = presets.resolve("ballad")
    assert (op.lo, op.hi, op.open_max) == (0.20, 0.55, 0.70)
    assert (gen.attack_frames, gen.release_frames) == (3, 3)
    assert (gen.coartic_overlap_max, gen.anticipation_frames, gen.min_hold_frames) == (3, 1, 4)
    assert gen.exaggeration == 0.8


def test_powerful_values():
    op, gen = presets.resolve("powerful")
    assert (op.lo, op.hi, op.open_max) == (0.40, 0.95, 0.97)
    assert (gen.attack_frames, gen.release_frames) == (1, 1)
    assert (gen.coartic_overlap_max, gen.anticipation_frames, gen.min_hold_frames) == (2, 2, 3)
    assert gen.exaggeration == 1.3


def test_whisper_values():
    op, gen = presets.resolve("whisper")
    assert (op.lo, op.hi, op.open_max) == (0.10, 0.35, 0.50)
    assert (gen.attack_frames, gen.release_frames) == (2, 2)
    assert (gen.coartic_overlap_max, gen.anticipation_frames, gen.min_hold_frames) == (2, 1, 3)
    assert gen.exaggeration == 0.7


def test_rap_values():
    op, gen = presets.resolve("rap")
    assert (op.lo, op.hi, op.open_max) == (0.30, 0.70, 0.85)
    assert (gen.attack_frames, gen.release_frames) == (1, 1)
    assert (gen.coartic_overlap_max, gen.anticipation_frames, gen.min_hold_frames) == (1, 1, 2)
    assert gen.exaggeration == 1.0


def test_default_open_defaults_to_range_midpoint():
    # 未指定の default_open は開き量レンジの中央 (lo+hi)/2(スタイルごと)。
    op, _ = presets.resolve("whisper")
    assert op.default_open == pytest.approx((0.10 + 0.35) / 2)


def test_open_max_override_replaces_preset_and_open_cap():
    # --open-max は写像上限(OpennessParams.open_max)かつ lipsync の open_cap を上書きする。
    op, gen = presets.resolve("pop", open_max=0.50)
    assert op.open_max == 0.50
    assert gen.open_cap == 0.50


def test_default_open_override_replaces_midpoint():
    op, _ = presets.resolve("pop", default_open=0.80)
    assert op.default_open == 0.80
