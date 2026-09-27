import pytest

from mocapvmd import footik


@pytest.mark.parametrize("name,side", [
    ("右足ＩＫ", "right"),
    ("左足ＩＫ", "left"),
    ("右つま先ＩＫ", "right"),
])
def test_detect_side_japanese(name, side):
    assert footik.detect_side(name) == side


@pytest.mark.parametrize("name,side", [
    ("left foot IK", "left"),
    ("right toe IK", "right"),
])
def test_detect_side_english_word(name, side):
    assert footik.detect_side(name) == side


@pytest.mark.parametrize("name,side", [
    ("foot_R_IK", "right"),
    ("foot_L_IK", "left"),
])
def test_detect_side_delimited_letter(name, side):
    assert footik.detect_side(name) == side


def test_detect_side_none_when_no_marker():
    assert footik.detect_side("足ＩＫ") is None


@pytest.mark.parametrize("name", ["leg IK", "toe IK"])
def test_detect_side_ignores_letters_inside_words(name):
    assert footik.detect_side(name) is None


@pytest.mark.parametrize("name", ["leftover foot IK", "bright toe IK"])
def test_detect_side_word_inside_larger_word_is_ignored(name):
    assert footik.detect_side(name) is None


def test_detect_side_japanese_precedes_word():
    assert footik.detect_side("右 left foot IK") == "right"


def test_detect_side_japanese_precedes_letter():
    assert footik.detect_side("右foot_L_IK") == "right"


def test_detect_side_word_precedes_letter():
    assert footik.detect_side("left foot_R_IK") == "left"


def _pairset(result):
    return {(p.side, p.foot, p.toe) for p in result.pairs}


def test_pair_japanese_left_and_right_into_two_pairs():
    tracks = [
        ("右足ＩＫ", "foot_ik"), ("右つま先ＩＫ", "toe_ik"),
        ("左足ＩＫ", "foot_ik"), ("左つま先ＩＫ", "toe_ik"),
    ]
    result = footik.pair_ik_tracks(tracks)
    assert _pairset(result) == {
        ("right", "右足ＩＫ", "右つま先ＩＫ"),
        ("left", "左足ＩＫ", "左つま先ＩＫ"),
    }
    assert result.unpaired == ()
    assert result.ambiguous == ()


def test_pair_english_single_pair():
    tracks = [("left foot IK", "foot_ik"), ("left toe IK", "toe_ik")]
    result = footik.pair_ik_tracks(tracks)
    assert _pairset(result) == {("left", "left foot IK", "left toe IK")}
    assert result.unpaired == ()
    assert result.ambiguous == ()


def test_pair_foot_without_toe_is_unpaired():
    result = footik.pair_ik_tracks([("右足ＩＫ", "foot_ik")])
    assert result.pairs == ()
    assert set(result.unpaired) == {"右足ＩＫ"}
    assert result.ambiguous == ()


def test_pair_duplicate_foot_is_ambiguous_toe_unpaired():
    tracks = [
        ("右足ＩＫ", "foot_ik"), ("右足ＩＫ先", "foot_ik"),
        ("右つま先ＩＫ", "toe_ik"),
    ]
    result = footik.pair_ik_tracks(tracks)
    assert result.pairs == ()
    assert set(result.ambiguous) == {"右足ＩＫ", "右足ＩＫ先"}
    assert set(result.unpaired) == {"右つま先ＩＫ"}


def test_pair_duplicate_toe_is_ambiguous_foot_unpaired():
    tracks = [
        ("右足ＩＫ", "foot_ik"),
        ("右つま先ＩＫ", "toe_ik"), ("右つま先ＩＫ先", "toe_ik"),
    ]
    result = footik.pair_ik_tracks(tracks)
    assert result.pairs == ()
    assert set(result.ambiguous) == {"右つま先ＩＫ", "右つま先ＩＫ先"}
    assert set(result.unpaired) == {"右足ＩＫ"}


def test_pair_unknown_side_is_ambiguous():
    tracks = [("足ＩＫ", "foot_ik"), ("つま先ＩＫ", "toe_ik")]
    result = footik.pair_ik_tracks(tracks)
    assert result.pairs == ()
    assert set(result.ambiguous) == {"足ＩＫ", "つま先ＩＫ"}
    assert result.unpaired == ()


def _still(n, pos):
    return [tuple(pos) for _ in range(n)]


def test_relative_no_rejection_when_offset_constant():
    foot = _still(6, (0.0, 0.0, 0.0))
    toe = _still(6, (1.0, 0.0, 0.0))
    assert footik.relative_rejected_frames(foot, toe) == frozenset()


def test_relative_rejects_sudden_change_frame():
    foot = _still(6, (0.0, 0.0, 0.0))
    toe = [(1.0, 0.0, 0.0)] * 3 + [(1.2, 0.0, 0.0)] * 3
    assert footik.relative_rejected_frames(foot, toe) == frozenset({3})


@pytest.mark.parametrize("jump,rejected", [
    pytest.param(0.079, set(), id="just_below_threshold_kept"),
    pytest.param(0.081, {3}, id="just_above_threshold_rejected"),
])
def test_relative_threshold_boundary(jump, rejected):
    foot = _still(6, (0.0, 0.0, 0.0))
    toe = [(0.0, 0.0, 0.0)] * 3 + [(jump, 0.0, 0.0)] * 3
    assert footik.relative_rejected_frames(foot, toe) == frozenset(rejected)


def test_relative_skips_missing_paired_key():
    foot = _still(6, (0.0, 0.0, 0.0))
    toe = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), None, (1.2, 0.0, 0.0),
           (1.2, 0.0, 0.0), (1.2, 0.0, 0.0)]
    assert footik.relative_rejected_frames(foot, toe) == frozenset()


def test_relative_uses_difference_not_toe_alone():
    foot = [(round(0.2 * i, 6), 0.0, 0.0) for i in range(6)]
    toe = [(round(1.0 + 0.2 * i, 6), 0.0, 0.0) for i in range(6)]
    assert footik.relative_rejected_frames(foot, toe) == frozenset()


@pytest.mark.parametrize("v,rejected", [
    pytest.param(0.06, {3}, id="xz_each_below_threshold_but_norm_above_rejected"),
    pytest.param(0.05, set(), id="xz_norm_below_threshold_kept"),
])
def test_relative_change_is_euclidean(v, rejected):
    foot = _still(6, (0.0, 0.0, 0.0))
    toe = [(1.0, 0.0, 0.0)] * 3 + [(round(1.0 + v, 6), 0.0, round(v, 6))] * 3
    assert footik.relative_rejected_frames(foot, toe) == frozenset(rejected)


def test_relative_change_includes_y_axis():
    foot = _still(6, (0.0, 0.0, 0.0))
    toe = [(1.0, 0.0, 0.0)] * 3 + [(1.0, 0.09, 0.0)] * 3
    assert footik.relative_rejected_frames(foot, toe) == frozenset({3})


def test_detect_paired_none_matches_single_track():
    foot = _still(8, (0.0, 0.0, 0.0))
    with_none = footik.detect_grounding_segments(foot, paired_positions=None)
    single = footik.detect_grounding_segments(foot)
    assert with_none.candidate_frames == single.candidate_frames
    assert with_none.segments == single.segments
    assert with_none.relative_rejected_frames == frozenset()


def test_detect_relative_excludes_candidate_frames():
    foot = _still(12, (0.0, 0.0, 0.0))
    toe = [(1.0, 0.0, 0.0)] * 12
    toe[6] = (1.2, 0.0, 0.0)
    det = footik.detect_grounding_segments(foot, paired_positions=toe)
    assert det.relative_rejected_frames == frozenset({6, 7})
    assert 6 not in det.candidate_frames and 7 not in det.candidate_frames
    assert [(s.start, s.end) for s in det.segments] == [(0, 5), (8, 11)]
