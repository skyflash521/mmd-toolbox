import pytest

from mocapvmd import classify


@pytest.mark.parametrize(
    "name,expected",
    [
        ("全ての親", "root"),
        ("root", "root"),
        ("センター", "center"),
        ("グルーブ", "center"),
        ("center", "center"),
        ("groove", "center"),
        ("上半身", "torso"),
        ("上半身2", "torso"),
        ("下半身", "torso"),
        ("首", "torso"),
        ("頭", "torso"),
        ("upper body", "torso"),
        ("lower body", "torso"),
        ("neck", "torso"),
        ("head", "torso"),
        ("左肩", "arms"),
        ("右腕", "arms"),
        ("左ひじ", "arms"),
        ("右肘", "arms"),
        ("左手首", "arms"),
        ("left arm", "arms"),
        ("right shoulder", "arms"),
        ("right elbow", "arms"),
        ("left wrist", "arms"),
        ("左親指1", "fingers"),
        ("右人指2", "fingers"),
        ("left thumb", "fingers"),
        ("right index finger", "fingers"),
        ("左足", "legs"),
        ("左脚", "legs"),
        ("右ひざ", "legs"),
        ("右膝", "legs"),
        ("左足首", "legs"),
        ("右つま先", "legs"),
        ("left leg", "legs"),
        ("left knee", "legs"),
        ("right ankle", "legs"),
        ("right toe", "legs"),
    ],
)
def test_classify_categories(name, expected):
    assert classify.classify(name) == expected


@pytest.mark.parametrize("name", ["left middle", "right ring", "left pinky", "right little"])
def test_english_finger_name_without_word_finger_is_fingers(name):
    assert classify.classify(name) == "fingers"


@pytest.mark.parametrize(
    "name",
    ["右足IK", "右足ＩＫ", "左足ＩＫ", "右足IK親", "left foot IK", "right leg IK"],
)
def test_foot_ik_precedes_legs(name):
    assert classify.classify(name) == "foot_ik"


@pytest.mark.parametrize(
    "name",
    ["右つま先IK", "右つま先ＩＫ", "左つま先ＩＫ", "right toe IK"],
)
def test_toe_ik_precedes_legs(name):
    assert classify.classify(name) == "toe_ik"


@pytest.mark.parametrize(
    "name",
    ["謎ボーン", "ネクタイ", "スカート1", "xyz", ""],
)
def test_unknown_for_unclassifiable(name):
    assert classify.classify(name) == "unknown"
