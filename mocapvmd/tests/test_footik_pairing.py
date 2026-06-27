"""足IK・つま先IKの左右ペアリングと相対位置急変条件のテスト(mocapvmd.md §4.3)。

- 左右判定 detect_side: 名前中の 左/右、独立語 left/right、区切りに囲まれた L/R の優先順で側を決める。
- ペアリング pair_ik_tracks: 同側で foot_ik と toe_ik が各1本のときだけペアにする。曖昧(同側同種が複数・
  側不明)はペアにせず、対応相手が無い単独は未ペアにする。
- 相対位置急変 relative_rejected_frames: 足IKとつま先IKのVMD位置差分ベクトルの1フレーム変化が
  確定閾値 0.08 を超えるフレームを返す(接地候補から除外する。mocapvmd.md §4.3)。差分の絶対距離でなく
  変化量のみを使う。対応キーを欠くフレームは判定をスキップする(欠損は急変ではない)。
- detect_grounding_segments は paired_positions を受け取ると相対条件を併用し、相対急変フレームを
  接地候補から除外し relative_rejected_frames に集計する。paired_positions が None なら単独検出に一致する。
"""

import pytest

from mocapvmd import footik


# --- detect_side ------------------------------------------------------------


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
    # 単独でない l/r(leg の l 等)を側マーカーと誤認しない。区切りに囲まれた L/R だけを採る。
    assert footik.detect_side(name) is None


@pytest.mark.parametrize("name", ["leftover foot IK", "bright toe IK"])
def test_detect_side_word_inside_larger_word_is_ignored(name):
    # 部分文字列の left(leftover)/right(bright)を独立語と誤認しない。
    assert footik.detect_side(name) is None


def test_detect_side_japanese_precedes_word():
    # 名前中の 左/右 を、独立語 left/right より優先する(両方あるとき)。
    assert footik.detect_side("右 left foot IK") == "right"


def test_detect_side_japanese_precedes_letter():
    # 名前中の 左/右 を、区切りの L/R より優先する(両方あるとき)。
    assert footik.detect_side("右foot_L_IK") == "right"


def test_detect_side_word_precedes_letter():
    # 独立語 left を、区切りの L/R より優先する(両方あるとき)。
    assert footik.detect_side("left foot_R_IK") == "left"


# --- pair_ik_tracks ---------------------------------------------------------


def _pairset(result):
    return {(p.side, p.foot, p.toe) for p in result.pairs}


def test_pair_standard_two_pairs():
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
    # 同側の足IKが2本 → どちらも曖昧。相方の一意なつま先IKはペアを組めず未ペア。
    tracks = [
        ("右足ＩＫ", "foot_ik"), ("右足ＩＫ先", "foot_ik"),
        ("右つま先ＩＫ", "toe_ik"),
    ]
    result = footik.pair_ik_tracks(tracks)
    assert result.pairs == ()
    assert set(result.ambiguous) == {"右足ＩＫ", "右足ＩＫ先"}
    assert set(result.unpaired) == {"右つま先ＩＫ"}


def test_pair_duplicate_toe_is_ambiguous_foot_unpaired():
    # 対称ケース: 同側のつま先IKが2本 → どちらも曖昧。相方の一意な足IKは未ペア。
    tracks = [
        ("右足ＩＫ", "foot_ik"),
        ("右つま先ＩＫ", "toe_ik"), ("右つま先ＩＫ先", "toe_ik"),
    ]
    result = footik.pair_ik_tracks(tracks)
    assert result.pairs == ()
    assert set(result.ambiguous) == {"右つま先ＩＫ", "右つま先ＩＫ先"}
    assert set(result.unpaired) == {"右足ＩＫ"}


def test_pair_unknown_side_is_ambiguous():
    # 側を判定できないトラックはペアにせず曖昧扱い。
    tracks = [("足ＩＫ", "foot_ik"), ("つま先ＩＫ", "toe_ik")]
    result = footik.pair_ik_tracks(tracks)
    assert result.pairs == ()
    assert set(result.ambiguous) == {"足ＩＫ", "つま先ＩＫ"}
    assert result.unpaired == ()


# --- relative_rejected_frames ----------------------------------------------


def _still(n, pos):
    return [tuple(pos) for _ in range(n)]


def test_relative_no_rejection_when_offset_constant():
    # 足IK・つま先IKがともに静止し差分が一定なら、相対急変フレームは無い。
    foot = _still(6, (0.0, 0.0, 0.0))
    toe = _still(6, (1.0, 0.0, 0.0))
    assert footik.relative_rejected_frames(foot, toe) == frozenset()


def test_relative_rejects_sudden_change_frame():
    # つま先IKが frame3 で 0.2 動く。差分の1フレーム変化 0.2 > 0.08 で frame3 を急変として返す。
    foot = _still(6, (0.0, 0.0, 0.0))
    toe = [(1.0, 0.0, 0.0)] * 3 + [(1.2, 0.0, 0.0)] * 3
    assert footik.relative_rejected_frames(foot, toe) == frozenset({3})


@pytest.mark.parametrize("jump,rejected", [(0.079, set()), (0.081, {3})])
def test_relative_threshold_boundary(jump, rejected):
    # 閾値 0.08 の内外を 0.079(非急変)/0.081(急変)で挟む。VMD位置は float32 で読まれ厳密 0.08 は
    # 量子化・sqrt 丸めで境界が揺れるため、速度しきい値テストと同様に厳密境界そのものは固定しない。
    foot = _still(6, (0.0, 0.0, 0.0))
    toe = [(0.0, 0.0, 0.0)] * 3 + [(jump, 0.0, 0.0)] * 3
    assert footik.relative_rejected_frames(foot, toe) == frozenset(rejected)


def test_relative_skips_missing_paired_key():
    # つま先IKキーを欠くフレーム(None)は相対判定をスキップする(欠損は急変ではない)。
    foot = _still(6, (0.0, 0.0, 0.0))
    toe = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), None, (1.2, 0.0, 0.0),
           (1.2, 0.0, 0.0), (1.2, 0.0, 0.0)]
    assert footik.relative_rejected_frames(foot, toe) == frozenset()


def test_relative_uses_difference_not_toe_alone():
    # 足IKとつま先IKが同量(0.2/frame)動くと差分は一定 → 急変なし。つま先単体の移動量を使う誤実装を弾く。
    foot = [(round(0.2 * i, 6), 0.0, 0.0) for i in range(6)]
    toe = [(round(1.0 + 0.2 * i, 6), 0.0, 0.0) for i in range(6)]
    assert footik.relative_rejected_frames(foot, toe) == frozenset()


@pytest.mark.parametrize("v,rejected", [(0.06, {3}), (0.05, set())])
def test_relative_change_is_euclidean(v, rejected):
    # 差分変化を X・Z 同時に v ずつ与える。合成 sqrt(2)*v が v=0.06→約0.0849(急変)、
    # v=0.05→約0.0707(非急変)。各軸単独では 0.06 も 0.08 未満なので軸別判定の誤実装を弾く。
    foot = _still(6, (0.0, 0.0, 0.0))
    toe = [(1.0, 0.0, 0.0)] * 3 + [(round(1.0 + v, 6), 0.0, round(v, 6))] * 3
    assert footik.relative_rejected_frames(foot, toe) == frozenset(rejected)


def test_relative_change_includes_y_axis():
    # 差分ベクトルは3次元。Y のみ 0.09 動いても急変として検出する(水平ノルムだけの実装を弾く)。
    foot = _still(6, (0.0, 0.0, 0.0))
    toe = [(1.0, 0.0, 0.0)] * 3 + [(1.0, 0.09, 0.0)] * 3
    assert footik.relative_rejected_frames(foot, toe) == frozenset({3})


# --- detect_grounding_segments の paired 拡張 -------------------------------


def test_detect_paired_none_matches_single_track():
    foot = _still(8, (0.0, 0.0, 0.0))
    with_none = footik.detect_grounding_segments(foot, paired_positions=None)
    single = footik.detect_grounding_segments(foot)
    assert with_none.candidate_frames == single.candidate_frames
    assert with_none.segments == single.segments
    assert with_none.relative_rejected_frames == frozenset()


def test_detect_relative_excludes_candidate_frames():
    # 足IKは静止(全フレーム接地候補)。つま先IKが frame6 で単発スパイクし、frame6/7 が相対急変で除外される。
    foot = _still(12, (0.0, 0.0, 0.0))
    toe = [(1.0, 0.0, 0.0)] * 12
    toe[6] = (1.2, 0.0, 0.0)
    det = footik.detect_grounding_segments(foot, paired_positions=toe)
    assert det.relative_rejected_frames == frozenset({6, 7})
    assert 6 not in det.candidate_frames and 7 not in det.candidate_frames
    assert [(s.start, s.end) for s in det.segments] == [(0, 5), (8, 11)]
