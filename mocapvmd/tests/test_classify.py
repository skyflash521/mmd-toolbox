"""ボーン分類のテスト(mocapvmd.md §4.1 / implementation-plan §5.1)。

ボーン名から種別(root / center / torso / arms / fingers / legs / foot_ik / toe_ik /
unknown)を推定する。具体度の高い種別を優先し、足IK・つま先IKが legs より先に判定される。
"""

import importlib.util

import pytest

# classify モジュール(実体ファイル)が無い間はモジュールごと skip する。find_spec で存在判定し、
# 存在すれば通常 import して内部 import 失敗は表面化させる。
if importlib.util.find_spec("mocapvmd.classify") is None:
    pytest.skip("impl pending: Step 1b classify", allow_module_level=True)

from mocapvmd import classify  # noqa: E402


@pytest.mark.parametrize(
    "name,expected",
    [
        # root
        ("全ての親", "root"),
        ("root", "root"),
        # center
        ("センター", "center"),
        ("グルーブ", "center"),
        ("center", "center"),
        ("groove", "center"),
        # torso
        ("上半身", "torso"),
        ("上半身2", "torso"),
        ("下半身", "torso"),
        ("首", "torso"),
        ("頭", "torso"),
        ("upper body", "torso"),
        ("lower body", "torso"),
        ("neck", "torso"),
        ("head", "torso"),
        # arms
        ("左肩", "arms"),
        ("右腕", "arms"),
        ("左ひじ", "arms"),
        ("右肘", "arms"),
        ("左手首", "arms"),
        ("left arm", "arms"),
        ("right shoulder", "arms"),
        ("right elbow", "arms"),
        ("left wrist", "arms"),
        # fingers
        ("左親指1", "fingers"),
        ("右人指2", "fingers"),
        ("left thumb", "fingers"),
        ("right index finger", "fingers"),
        # finger 英語 glob を *finger* に吸収されない裸のキーワード名で個別に検証する。
        ("left middle", "fingers"),
        ("right ring", "fingers"),
        ("left pinky", "fingers"),
        ("right little", "fingers"),
        # legs(IKでない脚部)
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


@pytest.mark.parametrize(
    "name",
    ["右足IK", "右足ＩＫ", "左足ＩＫ", "右足IK親", "left foot IK", "right leg IK"],
)
def test_foot_ik_precedes_legs(name):
    # 足IKは legs(*足*)より先に foot_ik と判定される(半角・全角IK)。
    assert classify.classify(name) == "foot_ik"


@pytest.mark.parametrize(
    "name",
    ["右つま先IK", "右つま先ＩＫ", "左つま先ＩＫ", "right toe IK"],
)
def test_toe_ik_precedes_legs(name):
    # つま先IKは legs(*つま先*)より先に toe_ik と判定される(半角・全角IK)。
    assert classify.classify(name) == "toe_ik"


@pytest.mark.parametrize(
    "name",
    ["謎ボーン", "ネクタイ", "スカート1", "xyz", ""],
)
def test_unknown_for_unclassifiable(name):
    # 分類できない名前は unknown(除外せず後段で保守的に処理する種別)。
    assert classify.classify(name) == "unknown"
