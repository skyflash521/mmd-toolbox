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
    # mid1=5, mid2=15, b=10, hw=min(1,5,5)=1。前後の最終重みはどちらも0.8*0.75=0.6(各小区間の
    # 最終重みはその小区間自身のApertureClassで決まる)。谷係数d=0.75/2=0.375、
    # 谷central=0.375*(0.6+0.6)=0.45。
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
            (9, 0.6),
            (10, 0.45),
            (11, 0.6),
            (15, 0.6),
            (18, 0.6),
            (20, 0.0),
        ],
    )


def test_three_moras_all_boundaries_get_valleys_when_well_spaced():
    # 3モーラ(あ、全て op0.8・FIRM_CLOSURE で最終重み0.6均一、長さ10ずつ)を連結。内部境界
    # b1=10, b2=20(いずれも hw=1、間隔十分)は mora_valley_min_gap_frames の既定(4)を満たして
    # 両方共存できる(密集回避で見送られない基本ケース)。谷central=0.375*(0.6+0.6)=0.45。
    events = [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
        MouthEvent(MouthShape.A, 10.0, 20.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
        MouthEvent(MouthShape.A, 20.0, 30.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
    ]
    env = _envelope(events, GenerationParams(vibrato_amp=0.0))
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.6),
            (5, 0.6),
            (9, 0.6),
            (10, 0.45),
            (11, 0.6),
            (15, 0.6),
            (19, 0.6),
            (20, 0.45),
            (21, 0.6),
            (25, 0.6),
            (28, 0.6),
            (30, 0.0),
        ],
    )


def test_valley_depth_scales_with_aperture_class():
    # 前後の小区間に同じ ApertureClass(両方 FIRM_CLOSURE、または両方 SLIGHT_CLOSURE)を与え
    # 前後の最終重みを0.6で揃え、境界の ApertureClass だけを変えると谷の深さが変わることを見る。
    # FIRM_CLOSURE(aperture_scale=0.75): d=0.375, valley=0.375*1.2=0.45(深い)。
    # SLIGHT_CLOSURE(aperture_scale=0.92): d=0.46, valley=0.46*1.2=0.552(浅い)。
    # SLIGHT_CLOSURE の方が FIRM_CLOSURE より谷が浅い(0.552 > 0.45)ことを既知値で検証する。
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
    # あ[0,10]・あ[10,11](中間モーラが1フレームと極短)・あ[11,20]。境界(b1=10)の半幅は
    # hw1=floor(min(1, 10-5, 10.5-10))=floor(0.5)=0 となり、e2 の ApertureClass が
    # FIRM_CLOSURE(NONE でない)でも谷を挿入しない(強弱節点までの余裕が無い)。境界(b2=11)は
    # e3 の ApertureClass が NONE なのでそもそも谷は生成しない。強弱節点(mid1=5, mid2=10.5→11,
    # mid3=15.5→16)だけが残る。
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


def test_valley_uses_lerp_not_raw_endpoints_for_unequal_weights():
    # あ[0,6]op0.15(NONE)・あ[6,20]op0.8/0.75=... 最終重み0.60(FIRM_CLOSURE) を連結。
    # mid1=3(w=0.15), mid2=13(w=0.60), b=6, hw=min(1,3,7)=1。lerp は mid1→mid2 を結ぶ直線
    # (傾き0.045)。谷の3点は前後の保持値そのもの(0.15, 0.60)でなく、lerp(5)=0.24・
    # lerp(7)=0.33 を使う。谷central=0.375*(0.24+0.33)=0.21375。前後の値をそのまま使う
    # 誤った実装なら 0.375*(0.15+0.60)=0.28125 になり、この既知値とは異なる。
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
            (5, 0.24),
            (6, 0.21375),
            (7, 0.33),
            (13, 0.6),
            (18, 0.6),
            (20, 0.0),
        ],
    )


def test_density_avoidance_prioritizes_total_displacement_over_greedy():
    # 4小区間(あ、長さ[10,4,4,10])を連結。境界の ApertureClass は中央(b2=14、e3自身の
    # クラス)を FIRM_CLOSURE(aperture_scale=0.75)、両端(b1=10, b3=18、e2/e4のクラス)を
    # NARROW_CHANNEL(aperture_scale=0.85)にし、各小区間の open_amount は最終重みが0.6で
    # 揃うよう正規化する(NARROW側は0.6/0.85、FIRM側は0.6/0.75)。lerpが0.6均一なので
    # 変位量は disp_narrow=(1-0.85)*0.6=0.09(両端)、disp_firm=(1-0.75)*0.6=0.15(中央)。
    # 中央単体の変位(0.15)は両端それぞれ(0.09)より大きいため、変位降順の逐次貪欲法(中央を
    # 先に採用し、間隔条件で衝突する両端を捨てる)なら中央だけが残る。しかし
    # mora_valley_min_gap_frames=4 の下では、隣接候補どうし(b1-b2、b2-b3)は間隔条件を満たさず
    # 共存できないが、b1とb3は間隔条件を満たし共存できる(両端の変位合計0.18が中央単体の0.15を
    # 上回る)。動的計画法は総変位を最大化するb1とb3を採用し中央(b2)を見送る。
    # 谷central: NARROW側=0.425*1.2=0.51、FIRM側(不採用)=0.375*1.2=0.45。
    params = GenerationParams(vibrato_amp=0.0, mora_valley_min_gap_frames=4)
    events = [
        MouthEvent(MouthShape.A, 0.0, 10.0, 0.6, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(
            MouthShape.A, 10.0, 14.0, 0.6 / 0.85, ConsonantClass.NONE,
            lipsync.ApertureClass.NARROW_CHANNEL,
        ),
        MouthEvent(
            MouthShape.A, 14.0, 18.0, 0.6 / 0.75, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
        MouthEvent(
            MouthShape.A, 18.0, 28.0, 0.6 / 0.85, ConsonantClass.NONE,
            lipsync.ApertureClass.NARROW_CHANNEL,
        ),
    ]
    env = _envelope(events, params)
    by_frame = dict(env["あ"])
    assert by_frame[10] == pytest.approx(0.51)  # b1(端。NARROW)の谷は採用される
    assert by_frame[18] == pytest.approx(0.51)  # b3(端。NARROW)の谷は採用される
    assert 14 not in by_frame  # b2(中央。FIRM。単体では変位最大)は見送られ、谷・肩キーは出力されない
    assert by_frame[12] == pytest.approx(0.6) and by_frame[16] == pytest.approx(0.6)  # mid2・mid3 は平坦


def test_density_avoidance_uses_multi_morph_sum_not_single_morph_max():
    # え[0,10]op0.2(NONE consonant, NONE aperture)→え[10,16]op0.2(NONE consonant,
    # FIRM_CLOSURE aperture。b1=10 を作る。主モーフ「え」だけが動く単一モーフ境界)→
    # え[16,18]op0.7(ROUNDED consonant、SLIGHT_CLOSURE aperture。b2=16 を作る。ROUNDEDの
    # 補助「う」も加わる複モーフ境界)→え[18,26]op0.3(NONE、NONE。閉じ側の埋め)。
    # b1(FIRM_CLOSURE)の変位は主モーフ「え」だけの lerp(10)=0.16875 から disp_P=0.25*0.16875
    # =0.0421875。b2(SLIGHT_CLOSURE)は「え」lerp(16)=0.462115・「う」lerp(16)=0.127385から
    # 最大成分だけなら disp_Q_max=0.08*0.462115=0.036969 (disp_Pより小さい)だが、
    # 合計なら disp_Q_sum=0.08*(0.462115+0.127385)=0.04716 (disp_Pより大きい)。
    # b1とb2は間隔条件(既定gap=4)を満たさず共存できない(round(15)-round(11)-1=3<4)。
    # 最大成分だけで優先度を決める誤った実装ならb1(単一モーフ)を優先しb2を見送るが、
    # 全モーフの合計で優先度を決める仕様どおりの実装ではb2(複モーフ、合計が大きい)を
    # 優先しb1を見送る。
    params = GenerationParams(vibrato_amp=0.0, mora_valley_min_gap_frames=4)
    events = [
        MouthEvent(MouthShape.E, 0.0, 10.0, 0.2, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(
            MouthShape.E, 10.0, 16.0, 0.2, ConsonantClass.NONE,
            lipsync.ApertureClass.FIRM_CLOSURE,
        ),
        MouthEvent(
            MouthShape.E, 16.0, 18.0, 0.7, ConsonantClass.ROUNDED,
            lipsync.ApertureClass.SLIGHT_CLOSURE,
        ),
        MouthEvent(MouthShape.E, 18.0, 26.0, 0.3, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
    ]
    env = _envelope(events, params)
    by_frame_e = dict(env["え"])
    by_frame_u = dict(env["う"])
    assert 10 not in by_frame_e  # b1(単一モーフ、変位単体は大きい)は見送られる
    assert by_frame_e[16] == pytest.approx(0.4251461538461539)  # b2の「え」は谷central(採用される)
    assert by_frame_u[16] == pytest.approx(0.11719384615384618)  # b2の「う」も谷central


def test_density_avoidance_uses_quantized_frames_for_gap_check():
    # 3小区間(あ、op0.6均一、FIRM_CLOSURE)を、境界の実数上の左端・右端が互いに逆方向へ丸まる
    # 位置([0,9.51],[9.51,15.52],[15.52,25.52])で連結する。境界1(b1=9.51)の右端は10.51
    # (四捨五入で11)、境界2(b2=15.52)の左端は14.52(四捨五入で15)。実数の差(14.52-10.51=4.01)
    # だけで互換性を判定する誤った実装なら mora_valley_min_gap_frames=4 を上回り両方採用するが、
    # 整数フレーム化した後の差(15-11-1=3)で判定する仕様どおりの実装では
    # mora_valley_min_gap_frames=4 を満たさず一方(総変位が同値なので処理順の都合でb2側)だけが
    # 採用される。
    params = GenerationParams(vibrato_amp=0.0, mora_valley_min_gap_frames=4)
    events = [
        MouthEvent(MouthShape.A, 0.0, 9.51, 0.6, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(MouthShape.A, 9.51, 15.52, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
        MouthEvent(MouthShape.A, 15.52, 25.52, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
    ]
    env = _envelope(events, params)
    by_frame = dict(env["あ"])
    assert by_frame[16] == pytest.approx(0.45)  # b2(round=16)の谷は採用される
    assert 10 not in by_frame  # b1(round=10)の谷は量子化後の間隔不足で見送られる


def test_density_avoidance_prioritizes_actual_displacement_over_aperture_class_alone():
    # 3小区間(あ、長さ[4,2,4])を SLIGHT_CLOSURE→FIRM_CLOSURE の順の境界で連結する。最終重みは
    # w1=0.8(NONE)・w2=0.02*0.92=0.0184(SLIGHT_CLOSURE)・w3=0.02*0.75=0.015(FIRM_CLOSURE)。
    # SLIGHT境界(b1=4)は前後の最終重み差が大きく lerp(4)=0.278933 で実際に目立つ谷になる一方、
    # FIRM境界(b2=6)は前後とも最終重みがほぼ0で lerp(6)=0.017267 とほぼ無内容。
    # disp1=(1-0.92)*0.278933=0.022315、disp2=(1-0.75)*0.017267=0.004317。ApertureClassの強さ
    # (aperture_scaleの小ささ)だけで優先度を決める誤った実装なら閉じの強いFIRM側(b2)を優先し
    # SLIGHT側(b1)を見送るが、実際の変位量で優先度を決める仕様どおりの実装ではSLIGHT側(b1、
    # 変位が大きい)が優先されFIRM側(b2)が見送られる。谷central(b1)=0.46*(0.539467+0.0184)
    # =0.256619。
    params = GenerationParams(vibrato_amp=0.0, mora_valley_min_gap_frames=4)
    events = [
        MouthEvent(MouthShape.A, 0.0, 4.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(MouthShape.A, 4.0, 6.0, 0.02, ConsonantClass.NONE, lipsync.ApertureClass.SLIGHT_CLOSURE),
        MouthEvent(MouthShape.A, 6.0, 10.0, 0.02, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE),
    ]
    env = _envelope(events, params)
    by_frame = dict(env["あ"])
    assert by_frame[4] == pytest.approx(0.2566186666666667)  # SLIGHT境界(b1)の谷は採用される
    assert 6 not in by_frame  # FIRM境界(b2)は見送られ、その谷・肩キーは一切出力されない
