import random

import pytest

from vmd import interp
from vmd.types import BoneKey, CameraKey, bone_interp_bytes, camera_interp_bytes

LINEAR = (20, 20, 107, 107)
CURVE_A = (5, 122, 122, 5)
CURVE_B = (64, 0, 64, 127)
CURVE_C = (0, 64, 127, 64)

CAMERA_CHANNELS = ["pos_x", "pos_y", "pos_z", "rot", "distance", "fov"]
BONE_CHANNELS = ["pos_x", "pos_y", "pos_z", "rot"]


def camera_interp(curve) -> bytes:
    b = bytearray(24)
    for off in interp.CAMERA_CHANNEL_OFFSET.values():
        x1, y1, x2, y2 = curve
        b[off : off + 4] = bytes([x1, x2, y1, y2])
    return bytes(b)


def bone_interp(curve) -> bytes:
    b = bytearray(64)
    x1, y1, x2, y2 = curve
    b[0], b[4], b[8], b[12] = x1, y1, x2, y2
    b[1], b[5], b[9], b[13] = x1, y1, x2, y2
    b[17], b[6], b[10], b[14] = x1, y1, x2, y2
    b[18], b[7], b[11], b[15] = x1, y1, x2, y2
    return bytes(b)


def cam_key(frame, seed, curve=LINEAR, raw=None):
    rng = random.Random(seed)
    return CameraKey(
        frame,
        rng.uniform(5.0, 50.0),
        (rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(-10, 10)),
        (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3)),
        camera_interp(curve) if raw is None else raw,
        rng.randint(10, 90),
        0,
    )


def bone_key(frame, seed, curve=LINEAR, raw=None):
    rng = random.Random(seed)
    q = [rng.uniform(-1, 1) for _ in range(4)]
    n = sum(c * c for c in q) ** 0.5
    return BoneKey(
        b"\x00" * 15,
        frame,
        (rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(-10, 10)),
        tuple(c / n for c in q),
        bone_interp(curve) if raw is None else raw,
    )


def camera_track(frames, curves=None):
    curves = curves or {}
    return [cam_key(f, 100 + i, curves.get(i, LINEAR)) for i, f in enumerate(frames)]


def bone_track(frames, curves=None):
    curves = curves or {}
    return [bone_key(f, 200 + i, curves.get(i, LINEAR)) for i, f in enumerate(frames)]


def assert_range_matches_sample(keys, channel, f0, f1):
    got = interp.sample_range(keys, channel, f0, f1)
    frames = list(range(f0, f1 + 1))
    assert len(got) == len(frames)
    for value, f in zip(got, frames, strict=True):
        expected = interp.sample(keys, channel, f)
        if isinstance(expected, tuple):
            assert tuple(value) == pytest.approx(expected, abs=1e-9), f"frame {f}"
        else:
            assert float(value) == pytest.approx(expected, abs=1e-9), f"frame {f}"


def many_frames():
    rng = random.Random(7)
    return sorted(rng.sample(range(0, 200), 14))


MIXED = [LINEAR, CURVE_A, CURVE_B, CURVE_C]


class TestCameraChannels:
    @pytest.mark.parametrize("channel", CAMERA_CHANNELS)
    def test_linear_multiple_keys(self, channel):
        keys = camera_track([0, 10, 25, 40])
        assert_range_matches_sample(keys, channel, 0, 40)

    @pytest.mark.parametrize("channel", CAMERA_CHANNELS)
    def test_bezier_multiple_keys(self, channel):
        keys = camera_track([0, 10, 25, 40], {1: CURVE_A, 2: CURVE_B, 3: CURVE_C})
        assert_range_matches_sample(keys, channel, 0, 40)

    @pytest.mark.parametrize("channel", CAMERA_CHANNELS)
    def test_many_keys_with_mixed_curves(self, channel):
        frames = many_frames()
        curves = {i: MIXED[i % 4] for i in range(len(frames))}
        keys = camera_track(frames, curves)
        assert_range_matches_sample(keys, channel, frames[0], frames[-1])


class TestBoneChannels:
    @pytest.mark.parametrize("channel", BONE_CHANNELS)
    def test_linear_multiple_keys(self, channel):
        keys = bone_track([0, 10, 25, 40])
        assert_range_matches_sample(keys, channel, 0, 40)

    @pytest.mark.parametrize("channel", BONE_CHANNELS)
    def test_bezier_multiple_keys(self, channel):
        keys = bone_track([0, 10, 25, 40], {1: CURVE_A, 2: CURVE_B, 3: CURVE_C})
        assert_range_matches_sample(keys, channel, 0, 40)

    @pytest.mark.parametrize("channel", BONE_CHANNELS)
    def test_many_keys_with_mixed_curves(self, channel):
        frames = many_frames()
        curves = {i: MIXED[i % 4] for i in range(len(frames))}
        keys = bone_track(frames, curves)
        assert_range_matches_sample(keys, channel, frames[0], frames[-1])


@pytest.fixture(params=["camera", "bone"])
def kind(request):
    return request.param


def channels_of(kind):
    return CAMERA_CHANNELS if kind == "camera" else BONE_CHANNELS


def track_of(kind, frames, curves=None):
    return camera_track(frames, curves) if kind == "camera" else bone_track(frames, curves)


class TestRangePlacement:
    def test_range_extends_before_first_and_after_last(self, kind):
        keys = track_of(kind, [10, 20, 35], {1: CURVE_A, 2: CURVE_B})
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 0, 50)

    def test_range_entirely_before_first_key(self, kind):
        keys = track_of(kind, [10, 20, 35], {1: CURVE_A})
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 0, 5)

    def test_range_ending_exactly_on_first_key(self, kind):
        keys = track_of(kind, [10, 20, 35], {1: CURVE_A})
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 0, 10)

    def test_range_entirely_after_last_key(self, kind):
        keys = track_of(kind, [10, 20, 35], {1: CURVE_A, 2: CURVE_B})
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 40, 60)

    def test_range_starting_exactly_on_last_key(self, kind):
        keys = track_of(kind, [10, 20, 35], {2: CURVE_B})
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 35, 45)

    def test_partial_range_between_middle_frames(self, kind):
        keys = track_of(
            kind, [0, 12, 30, 50, 80], {1: CURVE_A, 2: CURVE_B, 3: CURVE_C, 4: CURVE_A}
        )
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 17, 61)

    def test_partial_range_within_one_segment(self, kind):
        keys = track_of(kind, [0, 30, 60], {1: CURVE_A, 2: CURVE_B})
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 35, 50)

    def test_single_frame_range_at_segment_interior(self, kind):
        keys = track_of(kind, [0, 30, 60], {1: CURVE_A, 2: CURVE_B})
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 42, 42)

    def test_single_frame_range_on_key_frame(self, kind):
        keys = track_of(kind, [0, 30, 60], {1: CURVE_A, 2: CURVE_B})
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 30, 30)

    def test_range_starting_and_ending_on_key_frames(self, kind):
        keys = track_of(kind, [0, 30, 60, 90], {1: CURVE_A, 2: CURVE_B, 3: CURVE_C})
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 30, 60)


class TestDegenerateKeys:
    def test_duplicate_frame_keys(self, kind):
        keys = track_of(kind, [0, 10, 10, 20], {1: CURVE_A, 2: CURVE_B, 3: CURVE_C})
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 0, 30)

    def test_duplicate_frame_keys_at_start(self, kind):
        keys = track_of(kind, [5, 5, 15, 30], {2: CURVE_A, 3: CURVE_B})
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 0, 35)

    def test_duplicate_frame_keys_at_end(self, kind):
        keys = track_of(kind, [0, 15, 30, 30], {1: CURVE_A, 2: CURVE_B, 3: CURVE_C})
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 0, 35)

    def test_triple_duplicate_frame_keys(self, kind):
        keys = track_of(kind, [0, 10, 10, 10, 20], {1: CURVE_A, 4: CURVE_B})
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 0, 25)

    def test_single_key(self, kind):
        keys = track_of(kind, [12])
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 0, 30)

    def test_single_key_with_range_on_key_frame(self, kind):
        keys = track_of(kind, [12])
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 12, 12)

    def test_one_frame_spacing(self, kind):
        keys = track_of(
            kind, [3, 4, 5, 6, 7], {1: CURVE_A, 2: CURVE_B, 3: CURVE_C, 4: CURVE_A}
        )
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 0, 10)

    def test_one_frame_spacing_mixed_with_long_segments(self, kind):
        keys = track_of(kind, [0, 1, 2, 40, 41, 80], {1: CURVE_A, 3: CURVE_B, 5: CURVE_C})
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 0, 85)


class TestEmptyRange:
    def test_end_before_start(self, kind):
        keys = track_of(kind, [0, 10, 20], {1: CURVE_A})
        for channel in channels_of(kind):
            assert len(interp.sample_range(keys, channel, 5, 4)) == 0

    def test_end_far_before_start(self, kind):
        keys = track_of(kind, [0, 10, 20], {1: CURVE_A})
        for channel in channels_of(kind):
            assert len(interp.sample_range(keys, channel, 15, 3)) == 0

    def test_empty_range_outside_key_span(self, kind):
        keys = track_of(kind, [10, 20])
        for channel in channels_of(kind):
            assert len(interp.sample_range(keys, channel, 100, 99)) == 0


CHANNEL_POOL = [
    (5, 122, 122, 5),
    (64, 0, 64, 127),
    (0, 64, 127, 64),
    (100, 10, 20, 120),
    (30, 90, 90, 30),
    (10, 0, 117, 127),
    (127, 0, 0, 127),
    (45, 80, 70, 15),
]
PER_CHANNEL_FRAMES = [0, 6, 13, 21, 30, 40]


def channel_curves(key_index, count):
    return [CHANNEL_POOL[(key_index * 3 + c) % len(CHANNEL_POOL)] for c in range(count)]


def per_channel_track(kind, curves_of_key=channel_curves):
    keys = []
    for i, f in enumerate(PER_CHANNEL_FRAMES):
        if kind == "camera":
            raw = camera_interp_bytes(*curves_of_key(i, 6))
            keys.append(cam_key(f, 300 + i, raw=raw))
        else:
            raw = bone_interp_bytes(*curves_of_key(i, 4))
            keys.append(bone_key(f, 400 + i, raw=raw))
    return keys


def uniform_track(kind, curve):
    return per_channel_track(kind, lambda i, n: [curve] * n)


def distance(a, b):
    if isinstance(a, tuple) or hasattr(a, "__len__"):
        return max(abs(x - y) for x, y in zip(a, b, strict=True))
    return abs(a - b)


class TestPerChannelCurves:
    def test_curves_differ_within_each_key(self, kind):
        count = len(channels_of(kind))
        assert len(PER_CHANNEL_FRAMES) >= 5
        for i in range(len(PER_CHANNEL_FRAMES)):
            assert len(set(channel_curves(i, count))) == count
        assert len({tuple(channel_curves(i, count)) for i in range(len(PER_CHANNEL_FRAMES))}) == len(
            PER_CHANNEL_FRAMES
        )

    def test_range_matches_sample_for_each_channel(self, kind):
        keys = per_channel_track(kind)
        for channel in channels_of(kind):
            assert_range_matches_sample(keys, channel, 0, 40)
            assert_range_matches_sample(keys, channel, -3, 45)

    def test_sample_uses_own_channel_curve(self, kind):
        keys = per_channel_track(kind)
        channels = channels_of(kind)
        for ci, channel in enumerate(channels):
            for i in range(1, len(keys)):
                k0, k1 = keys[i - 1], keys[i]
                own = uniform_track(kind, channel_curves(i, len(channels))[ci])
                for f in range(k0.frame + 1, k1.frame):
                    assert distance(
                        interp.sample(keys, channel, f), interp.sample(own, channel, f)
                    ) == pytest.approx(0.0, abs=1e-9), f"{channel} frame {f}"

    def test_channels_give_different_results_in_each_segment(self, kind):
        keys = per_channel_track(kind)
        channels = channels_of(kind)
        for i in range(1, len(keys)):
            k0, k1 = keys[i - 1], keys[i]
            curves = channel_curves(i, len(channels))
            for ci, channel in enumerate(channels):
                mixed = [interp.sample(keys, channel, f) for f in range(k0.frame + 1, k1.frame)]
                for cj, other_curve in enumerate(curves):
                    if cj == ci:
                        continue
                    other = uniform_track(kind, other_curve)
                    gap = max(
                        distance(m, interp.sample(other, channel, f))
                        for m, f in zip(mixed, range(k0.frame + 1, k1.frame), strict=True)
                    )
                    assert gap > 1e-3, f"segment {i} {channel} vs curve of channel {cj}"
