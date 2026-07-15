"""一般ノイズ軽減の適用層のテスト。

apply_denoise は密サンプル(位置・回転)とクリーニングパラメータ(窓・強度)から、平滑化済みの
密サンプルを返す。位置は中央値フィルタ後 Savitzky-Golay を位置のブレンド率で合成、回転は窓内正規化平均を
回転のブレンド率で合成。スパイクは抑制し、アクセント・カット境界・範囲端は保護(変更しない)。1フレーム
あたりの元値からの変更は 位置 各軸 0.3 / 回転 5度 にクランプする。

平滑化値は係数依存なので、性質(分散減・スパイク減・アクセント保持・保護不変・クランプ・strength0で恒等)で
検証する。
"""

import math

import pytest

from mocapvmd import denoise

IDENT = (0.0, 0.0, 0.0, 1.0)


def quat_y(deg):
    h = math.radians(deg) / 2.0
    return (0.0, math.sin(h), 0.0, math.cos(h))


def apply(positions, rotations, pos_window=5, rot_window=5, pos_strength=0.5, rot_strength=0.5):
    return denoise.apply_denoise(
        positions,
        rotations,
        pos_window=pos_window,
        rot_window=rot_window,
        pos_strength=pos_strength,
        rot_strength=rot_strength,
    )


def _norm(q):
    return math.sqrt(sum(c * c for c in q))


def _unit(q):
    n = _norm(q)
    return tuple(c / n for c in q)


def _quat_angle_deg(a, b):
    # 入力を正規化してから角度を測る(非単位入力で見かけ0度にしない)。
    a, b = _unit(a), _unit(b)
    d = abs(sum(x * y for x, y in zip(a, b)))
    d = min(1.0, d)
    return math.degrees(2.0 * math.acos(d))


def _assert_unit_quaternions(quats):
    for q in quats:
        assert _norm(q) == pytest.approx(1.0, abs=1e-6)


def _x_variation(positions):
    xs = [p[0] for p in positions]
    return sum(abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1))


# --- 恒等(強度0) ---------------------------------------------------------


def test_zero_strength_is_identity():
    xs = [0.0, 0.1, -0.1, 0.05, -0.05, 0.1, -0.1, 0.0, 0.05, -0.05, 0.0]
    pos = [(x, 0.0, 0.0) for x in xs]
    rots = [quat_y(2.0 * ((-1) ** i)) for i in range(11)]
    out_pos, out_rot = apply(pos, rots, pos_strength=0.0, rot_strength=0.0)
    for o, i in zip(out_pos, pos):
        assert o == pytest.approx(i)
    for o, i in zip(out_rot, rots):
        assert _quat_angle_deg(o, i) == pytest.approx(0.0, abs=1e-6)
    _assert_unit_quaternions(out_rot)


# --- 微小ノイズ軽減 ---------------------------------------------------------


def test_micro_noise_position_reduced():
    # 微小ジッタ(±0.05)を載せた静止。平滑化で軸方向の総変動が減る。
    xs = [0.0, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, 0.0]
    pos = [(x, 0.0, 0.0) for x in xs]
    out_pos, _ = apply(pos, [IDENT] * 11, pos_strength=1.0)
    assert _x_variation(out_pos) < _x_variation(pos)


def test_position_output_length_matches():
    pos = [(0.1 * i, 0.0, 0.0) for i in range(11)]
    out_pos, out_rot = apply(pos, [IDENT] * 11)
    assert len(out_pos) == 11
    assert len(out_rot) == 11


# --- スパイク抑制 -----------------------------------------------------------


def test_position_spike_suppressed():
    pos = [(0.0, 0.0, 0.0)] * 11
    pos[5] = (0.5, 0.0, 0.0)
    out_pos, _ = apply(pos, [IDENT] * 11, pos_strength=1.0)
    # スパイクが元の半分以下に抑えられる。
    assert abs(out_pos[5][0]) < 0.25


# --- アクセント・境界の保護 -------------------------------------------------


def test_accent_run_preserved():
    # 同方向継続(+0.5/frame)のアクセント区間は変更しない。
    xs = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0]
    pos = [(x, 0.0, 0.0) for x in xs]
    out_pos, _ = apply(pos, [IDENT] * 11, pos_strength=1.0)
    for f in (3, 4, 5, 6, 7):
        assert out_pos[f][0] == pytest.approx(xs[f])


def test_range_edges_preserved():
    xs = [0.0, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, 0.0]
    pos = [(x, 0.0, 0.0) for x in xs]
    out_pos, _ = apply(pos, [IDENT] * 11, pos_strength=1.0)
    assert out_pos[0][0] == pytest.approx(xs[0])
    assert out_pos[10][0] == pytest.approx(xs[10])


def test_cut_not_crossed():
    # カット境界をまたいで平滑化しない。境界の両側 5,6 は元値のまま。
    xs = [0.0] * 6 + [2.0] * 5
    pos = [(x, 0.0, 0.0) for x in xs]
    out_pos, _ = apply(pos, [IDENT] * 11, pos_strength=1.0)
    assert out_pos[5][0] == pytest.approx(0.0)
    assert out_pos[6][0] == pytest.approx(2.0)


# --- クランプ -----------------------------------------------------------


def test_position_change_clamped_per_axis():
    # 平滑化の目標が大きくても、1フレームの元値からの変更は各軸 0.3 以内。X と Y で別の交互列を与え、
    # 軸別にクランプされること、かつ実際に上限 0.3 に達するフレームがあること(クランプが効くこと)を確認。
    # X・Y を同振幅 0.6 で交互に振る。合成変位は sqrt(0.6^2+0.6^2)=0.85 でカット閾値1.0未満なので
    # カットにならず内部フレームは平滑化対象。SG目標(約0.41)はクランプ上限0.3を超えるので各軸0.3に張り付く。
    seq = [0.0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.0]
    pos = [(v, v, 0.0) for v in seq]
    out_pos, _ = apply(pos, [IDENT] * 11, pos_strength=1.0)
    dx = [abs(o[0] - i[0]) for o, i in zip(out_pos, pos)]
    dy = [abs(o[1] - i[1]) for o, i in zip(out_pos, pos)]
    assert max(dx) <= 0.3 + 1e-9
    assert max(dy) <= 0.3 + 1e-9
    # クランプ未実装(青天井)を排除: 各軸が独立に上限 0.3 へクランプされる。
    assert max(dx) == pytest.approx(0.3, abs=1e-6)
    assert max(dy) == pytest.approx(0.3, abs=1e-6)


def test_rotation_change_clamped():
    # 回転の1フレーム変更は 5度 以内。20度交互(カット閾値30度未満なのでカットにならない)で、
    # 平滑化目標が5度を超え、クランプが効く(上限5度に達するフレームがある)ことを確認。
    degs = [0.0, 20.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0, 0.0]
    rots = [quat_y(d) for d in degs]
    _, out_rot = apply([(0.0, 0.0, 0.0)] * 11, rots, rot_strength=1.0)
    changes = [_quat_angle_deg(o, i) for o, i in zip(out_rot, rots)]
    assert max(changes) <= 5.0 + 1e-6
    assert max(changes) == pytest.approx(5.0, abs=1e-3)
    _assert_unit_quaternions(out_rot)


# --- 回転平滑化 -------------------------------------------------------------


def test_rotation_jitter_reduced():
    # 微小回転ジッタ(±2度)を平滑化で抑える。隣接フレームの角度差の総和が減る。
    rots = [quat_y(2.0 * ((-1) ** i)) for i in range(11)]
    _, out_rot = apply([(0.0, 0.0, 0.0)] * 11, rots, rot_strength=1.0)
    before = sum(_quat_angle_deg(rots[i], rots[i + 1]) for i in range(10))
    after = sum(_quat_angle_deg(out_rot[i], out_rot[i + 1]) for i in range(10))
    assert after < before
    _assert_unit_quaternions(out_rot)


def test_rotation_accent_and_edges_preserved():
    # 回転側もアクセント run(同方向継続)とカット境界・範囲端を保護(変更しない)。
    # frames3-7 は +8度/frame の同方向 run でアクセント。frame0/10 は範囲端。
    degs = [0.0, 0.0, 0.0, 0.0, 8.0, 16.0, 24.0, 32.0, 32.0, 32.0, 32.0]
    rots = [quat_y(d) for d in degs]
    _, out_rot = apply([(0.0, 0.0, 0.0)] * 11, rots, rot_strength=1.0)
    for f in (3, 4, 5, 6, 7):
        assert _quat_angle_deg(out_rot[f], rots[f]) == pytest.approx(0.0, abs=1e-6)
    assert _quat_angle_deg(out_rot[0], rots[0]) == pytest.approx(0.0, abs=1e-6)
    assert _quat_angle_deg(out_rot[10], rots[10]) == pytest.approx(0.0, abs=1e-6)


def test_rotation_cut_not_crossed():
    # 回転のカット境界(40度ジャンプ>30度)の両側 5,6 を元姿勢のまま保持する。
    degs = [0.0] * 6 + [40.0] * 5
    rots = [quat_y(d) for d in degs]
    _, out_rot = apply([(0.0, 0.0, 0.0)] * 11, rots, rot_strength=1.0)
    assert _quat_angle_deg(out_rot[5], rots[5]) == pytest.approx(0.0, abs=1e-6)
    assert _quat_angle_deg(out_rot[6], rots[6]) == pytest.approx(0.0, abs=1e-6)


def test_rotation_zero_norm_input_raises():
    rots = [IDENT] * 11
    rots[5] = (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        apply([(0.0, 0.0, 0.0)] * 11, rots)
