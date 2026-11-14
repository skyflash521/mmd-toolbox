import pytest

from vpr2vmd import presets


@pytest.mark.parametrize(
    ("style", "described"),
    [
        ("pop", (0.90, 0.525, 0.45, 0.30, 0.02, 6, 11)),
        ("ballad", (0.70, 0.375, 0.40, 0.20, 0.025, 3, 1)),
        ("powerful", (0.97, 0.675, 0.40, 0.20, 0.025, 2, 2)),
        ("whisper", (0.50, 0.225, 0.40, 0.20, 0.025, 2, 1)),
        ("rap", (0.85, 0.50, 0.40, 0.20, 0.025, 1, 1)),
    ],
)
def test_described_preset_values(style, described):
    op, gen = presets.resolve(style)
    assert (
        op.open_max, op.default_open, gen.legato_valley_shallow, gen.legato_valley_deep,
        gen.legato_valley_slope, gen.coartic_overlap_max, gen.anticipation_frames,
    ) == pytest.approx(described)


def test_open_max_override_replaces_preset_and_open_cap():
    op, gen = presets.resolve("pop", open_max=0.50)
    assert op.open_max == 0.50
    assert gen.open_cap == 0.50


def test_default_open_override_replaces_midpoint():
    op, _ = presets.resolve("pop", default_open=0.80)
    assert op.default_open == 0.80
