"""モーラ境界の谷のテスト。

同一母音的口形が連続するグループの内部境界(モーラの区切り)に、境界を作る子音の ApertureClass に
応じた深さの谷を入れることを既知値で検証する。ApertureClass が NONE の境界には谷の3点自体を一切
生成しない。谷どうしが密に連続する場合は、視覚的な変位量の総和を最大化する動的計画法で間引き、
採用された谷の間には最低 mora_valley_min_gap_frames の無傷の保持フレームを残す(密集回避の互換性
判定は整数フレーム化した後の値で行う)。
"""

import pytest

import lipsync
from lipsync import ConsonantClass, GenerationParams, MouthEvent, MouthShape


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


def test_basic_valley_between_equal_weight_moras():
    # あ[0,10]op0.8・あ[10,20]op0.8、両方 ApertureClass.FIRM_CLOSURE を連結(「たた」に相当する
    # 最小ケース。同一ConsonantClass・同一開き量・同一ApertureClassでも境界に谷が入る)。
    # 両隣のモーラが長いため谷は完全飽和し(肩が両隣のmidそのものと一致)、独立した肩キーは
    # 生じない。前後の最終重みはどちらも0.6(各小区間の最終重みはその小区間自身のApertureClass
    # で決まる)。
    events = [
        MouthEvent(
            MouthShape.A, 0.0, 10.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE
        ),
        MouthEvent(
            MouthShape.A, 10.0, 20.0, 0.8, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
    ]
    env = _envelope(events)
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.6),
            (5, 0.6),
            (10, 0.45),
            (15, 0.6),
            (18, 0.6),
            (20, 0.0),
        ],
    )


def test_three_moras_all_boundaries_get_valleys_when_well_spaced():
    # 3モーラ(あ、全て op0.8・FIRM_CLOSURE で最終重み0.6均一、長さ30ずつ)を連結。内部境界
    # b1・b2とも両隣のモーラが十分長く半幅が既定値(12)まで伸びるが、両隣のmidまでは届かず
    # (=飽和しない)独立した肩キーが残る。間隔にも十分な余裕があり、mora_valley_min_gap_frames
    # の既定(4)を満たして両方の境界が共存する(密集回避で見送られない基本ケース)。
    events = [
        MouthEvent(MouthShape.A, 0.0, 30.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
        MouthEvent(MouthShape.A, 30.0, 60.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
        MouthEvent(MouthShape.A, 60.0, 90.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
    ]
    env = _envelope(events, GenerationParams(vibrato_amp=0.0))
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.6),
            (15, 0.6),
            (18, 0.6),
            (30, 0.45),
            (42, 0.6),
            (45, 0.6),
            (48, 0.6),
            (60, 0.45),
            (72, 0.6),
            (75, 0.6),
            (88, 0.6),
            (90, 0.0),
        ],
    )


def test_valley_depth_scales_with_aperture_class():
    # 前後の小区間に同じ ApertureClass(両方 FIRM_CLOSURE、または両方 SLIGHT_CLOSURE)を与え
    # 前後の最終重みを0.6で揃え、境界の ApertureClass だけを変えると谷の深さが変わることを見る。
    # SLIGHT_CLOSURE(閉じが弱い)の方が FIRM_CLOSURE(閉じが強い)より谷が浅いことを検証する。
    def _valley_center(aperture_class, aperture_scale):
        # 最終重み(=open_amount*aperture_scale)が両クラスとも0.6になるよう open_amount を
        # 正規化する(谷係数だけを比較するため)。
        open_amount = 0.6 / aperture_scale
        events = [
            MouthEvent(MouthShape.A, 0.0, 10.0, open_amount, ConsonantClass.NONE, aperture_class),
            MouthEvent(MouthShape.A, 10.0, 20.0, open_amount, ConsonantClass.NONE, aperture_class),
        ]
        env = _envelope(events)
        by_frame = dict(env["あ"])
        return by_frame[10]

    firm = _valley_center(lipsync.ApertureClass.FIRM_CLOSURE, 0.75)
    slight = _valley_center(lipsync.ApertureClass.SLIGHT_CLOSURE, 0.92)
    assert firm == pytest.approx(0.45)
    assert slight == pytest.approx(0.552)
    assert firm < slight  # 閉じが強い FIRM_CLOSURE の方が谷が深い


def test_none_aperture_produces_no_valley_points():
    # 境界(b=10)の ApertureClass が NONE なら、前後の最終重みが等しくても異なっても谷の3点は
    # 一切生成されず、強弱節点間の直線補間のみになる(現行実装(谷機構未実装)でも成立する回帰)。
    events = [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(MouthShape.A, 10.0, 20.0, 0.3, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
    ]
    env = _envelope(events)
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [(0, 0.0), (2, 0.8), (5, 0.8), (15, 0.3), (18, 0.3), (20, 0.0)],
    )


def test_valley_omitted_when_half_width_floors_to_zero():
    # あ[0,10]・あ[10,11](中間モーラが1フレームと極短)・あ[11,20]。境界(b1=10)は中間モーラが
    # 極短のため半幅が0になり、e2 の ApertureClass が FIRM_CLOSURE(NONE でない)でも谷を挿入
    # しない(強弱節点までの余裕が無い)。境界(b2=11)は e3 の ApertureClass が NONE なので
    # そもそも谷は生成しない。強弱節点だけが残る。
    events = [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(
            MouthShape.A, 10.0, 11.0, 0.6, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
        MouthEvent(MouthShape.A, 11.0, 20.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
    ]
    env = _envelope(events)
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.5),
            (5, 0.5),
            (11, 0.45),  # mid2 (10.5 を四捨五入): 0.6*0.75=0.45。谷ではなく通常の強弱節点
            (16, 0.5),  # mid3 (15.5 を四捨五入)
            (18, 0.5),
            (20, 0.0),
        ],
    )


def test_valley_omitted_when_half_width_floors_to_one():
    # あ[0,10]op0.8(NONE)・あ[10,13](3フレームの短いモーラ、FIRM_CLOSURE)・あ[13,23]op0.5(NONE)。
    # 境界(b=10)の半幅は1に floor されるが、半幅2未満の境界には谷を挿入しないため、
    # (hw=0の場合と同じく)谷の3点は一切出力されず、強弱節点間の直線補間のみになる。
    events = [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(
            MouthShape.A, 10.0, 13.0, 0.6, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
        MouthEvent(MouthShape.A, 13.0, 23.0, 0.5, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
    ]
    env = _envelope(events, GenerationParams(vibrato_amp=0.0))
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.8),
            (5, 0.8),
            (12, 0.45),  # mid2。谷ではなく通常の強弱節点(0.6*0.75=0.45)
            (18, 0.5),
            (21, 0.5),
            (23, 0.0),
        ],
    )


def test_valley_uses_lerp_not_raw_endpoints_for_unequal_weights():
    # あ[0,6]op0.15(NONE)・あ[6,20]op0.8(FIRM_CLOSURE、最終重み0.60)を連結。谷の3点は前後の
    # 保持値そのもの(0.15・0.60)でなく、両者を結ぶ直線補間上の値を使う(左肩は境界の余裕がmid1
    # 自身に達するため、mid1のキーと重なる)。前後の値をそのまま使う誤った実装なら中央が0.28125
    # になり、この既知値とは異なる。
    events = [
        MouthEvent(MouthShape.A, 0.0, 6.0, 0.15, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(
            MouthShape.A, 6.0, 20.0, 0.8, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
    ]
    env = _envelope(events)
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.15),
            (3, 0.15),
            (6, 0.21375),
            (9, 0.42),
            (13, 0.6),
            (18, 0.6),
            (20, 0.0),
        ],
    )


def test_density_avoidance_prioritizes_total_displacement_over_greedy():
    # 4小区間(あ、長さ[10,8,8,10])を連結。境界の ApertureClass は中央(b2、e3自身のクラス)を
    # FIRM_CLOSURE、両端(b1・b3、e2/e4のクラス)を NARROW_CHANNEL にし、各小区間の open_amount は
    # 最終重みが0.6で揃うよう正規化する。中央単体の変位は両端それぞれの変位より大きいため、
    # 変位降順の逐次貪欲法(中央を先に採用し、間隔条件で衝突する両端を捨てる)なら中央だけが残る。
    # しかしb1とb2、b2とb3はそれぞれ間隔条件(mora_valley_min_gap_frames=4)を満たさず共存できない
    # 一方、b1とb3は間隔条件を満たして共存でき、両端の変位合計が中央単体を上回る。動的計画法は
    # 総変位を最大化するb1とb3を採用し中央(b2)を見送る。
    params = GenerationParams(vibrato_amp=0.0, mora_valley_min_gap_frames=4)
    events = [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.6, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(
            MouthShape.A, 10.0, 18.0, 0.6 / 0.85, ConsonantClass.NONE,
            lipsync.ApertureClass.NARROW_CHANNEL,
        ),
        MouthEvent(
            MouthShape.A, 18.0, 26.0, 0.6 / 0.75, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
        MouthEvent(
            MouthShape.A, 26.0, 36.0, 0.6 / 0.85, ConsonantClass.NONE,
            lipsync.ApertureClass.NARROW_CHANNEL,
        ),
    ]
    env = _envelope(events, params)
    by_frame = dict(env["あ"])
    assert by_frame[10] == pytest.approx(0.51)  # b1(端。NARROW)の谷は採用される
    assert by_frame[26] == pytest.approx(0.51)  # b3(端。NARROW)の谷は採用される
    assert 18 not in by_frame  # b2(中央。FIRM。単体では変位最大)は見送られ、谷・肩キーは出力されない
    assert by_frame[14] == pytest.approx(0.6) and by_frame[22] == pytest.approx(0.6)  # mid2・mid3 は平坦


def test_density_avoidance_uses_multi_morph_sum_not_single_morph_max():
    # え[0,10]op0.2(NONE consonant, NONE aperture)→え[10,18]op0.2(NONE consonant,
    # FIRM_CLOSURE aperture。b1を作る。主モーフ「え」だけが動く単一モーフ境界)→
    # え[18,22]op0.7(ROUNDED consonant、SLIGHT_CLOSURE aperture。b2を作る。ROUNDEDの
    # 補助「う」も加わる複モーフ境界)→え[22,30]op0.3(NONE、NONE。閉じ側の埋め)。
    # b1とb2は間隔条件(既定gap=4)を満たさず共存できない。b1(単一モーフ「え」)の変位単体は
    # b2の最大成分(「え」のみ)よりわずかに大きいが、b2の全モーフ合計(「え」+「う」)は
    # b1を上回る。最大成分だけで優先度を決める誤った実装ならb1を優先しb2を見送るが、
    # 全モーフの合計で優先度を決める仕様どおりの実装ではb2(合計が大きい)を優先しb1を見送る。
    params = GenerationParams(vibrato_amp=0.0, mora_valley_min_gap_frames=4)
    events = [
        MouthEvent(MouthShape.E, 0.0, 10.0, 0.2, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(
            MouthShape.E, 10.0, 18.0, 0.2, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
        MouthEvent(
            MouthShape.E, 18.0, 22.0, 0.7, ConsonantClass.ROUNDED,
            lipsync.ApertureClass.SLIGHT_CLOSURE,
        ),
        MouthEvent(MouthShape.E, 22.0, 30.0, 0.3, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
    ]
    env = _envelope(events, params)
    by_frame_e = dict(env["え"])
    by_frame_u = dict(env["う"])
    assert 10 not in by_frame_e  # b1(単一モーフ、変位単体は大きい)は見送られる
    assert by_frame_e[18] == pytest.approx(0.3932410256410257)  # b2の「え」は谷central(採用される)
    assert by_frame_u[18] == pytest.approx(0.10417230769230772)  # b2の「う」も谷central


def test_density_avoidance_uses_quantized_frames_for_gap_check():
    # 3小区間(あ、長さ[8, 12.02, 8]、後の2区間はop0.8・FIRM_CLOSURE)を連結する。両境界とも
    # 半幅4で、境界どうしの実数上の間隔は4.02(mora_valley_min_gap_frames=4を満たすように見える)
    # だが、両端点を整数フレーム化すると間隔は3に減り足りなくなる。実数の間隔だけで互換性を
    # 判定する誤った実装は両方採用するが、整数フレーム化した後の間隔で判定する仕様どおりの
    # 実装では一方(b2側)だけが採用される。
    params = GenerationParams(vibrato_amp=0.0, mora_valley_min_gap_frames=4)
    events = [
        MouthEvent(MouthShape.A, 0.0, 8.0, 0.6, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(MouthShape.A, 8.0, 20.02, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
        MouthEvent(MouthShape.A, 20.02, 28.02, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
    ]
    env = _envelope(events, params)
    by_frame = dict(env["あ"])
    assert by_frame[20] == pytest.approx(0.45)  # b2の谷は採用される
    assert 8 not in by_frame  # b1の谷は量子化後の間隔不足で見送られる


def test_density_avoidance_prioritizes_actual_displacement_over_aperture_class_alone():
    # 3小区間(あ、長さ[8,4,8])を SLIGHT_CLOSURE→FIRM_CLOSURE の順の境界で連結し、両境界とも
    # 間隔条件(mora_valley_min_gap_frames=4)を満たさない配置にする。先頭区間は開き量0.8、
    # 残り2区間は開き量0.02とごく小さくする。SLIGHT境界(b1)は前後の最終重み差が大きく実際に
    # 目立つ谷になる一方、FIRM境界(b2)は前後とも最終重みがほぼ0でほぼ無内容。ApertureClassの
    # 強さ(閉じの強さ)だけで優先度を決める誤った実装なら閉じの強いFIRM側(b2)を優先し
    # SLIGHT側(b1)を見送るが、実際の変位量で優先度を決める仕様どおりの実装ではSLIGHT側(b1、
    # 変位が大きい)が優先されFIRM側(b2)が見送られる。
    params = GenerationParams(vibrato_amp=0.0, mora_valley_min_gap_frames=4)
    events = [
        MouthEvent(MouthShape.A, 0.0, 8.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(MouthShape.A, 8.0, 12.0, 0.02, ConsonantClass.NONE, lipsync.ApertureClass.SLIGHT_CLOSURE),
        MouthEvent(MouthShape.A, 12.0, 20.0, 0.02, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
    ]
    env = _envelope(events, params)
    by_frame = dict(env["あ"])
    assert by_frame[8] == pytest.approx(0.2566186666666667)  # SLIGHT境界(b1)の谷は採用される
    assert 12 not in by_frame  # FIRM境界(b2)は見送られ、その谷・肩キーは一切出力されない
