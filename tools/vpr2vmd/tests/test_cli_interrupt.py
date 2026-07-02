"""vpr2vmd CLI の中断(Ctrl-C 等)のテスト(vpr2vmd.md §7.5)。

変換段の KeyboardInterrupt を機械モードでは cancelled(exit 130)の終端イベント、非機械モードでは
理由を標準エラーへ 1 行(exit 130)で畳む。書き込みは全計算後に 1 回だけなので、中断で出力ファイルは
残らない。

機械モード stdout は UTF-8 バイトでバイナリバッファへ書くため capsysbinary で捕捉する。
"""

import json

from vpr import Note, Part, TempoEvent, Track, VprProject

from vpr2vmd import cli


def _touch(path):
    path.write_bytes(b"")
    return str(path)


def _stub_read(monkeypatch):
    note = Note(start_tick=0, duration_tick=480, pitch=60, lyric="x", velocity=64, phonemes=["a"])
    project = VprProject(
        resolution=480,
        tempos=[TempoEvent(0, 120.0)],
        tracks=[Track(name="Vocal", parts=[Part(name="p", start_tick=0, notes=[note])])],
    )
    monkeypatch.setattr(cli, "read", lambda _src: (project, []))


def _raise_keyboard_interrupt(*_a, **_k):
    raise KeyboardInterrupt()


def _machine_error(capsysbinary):
    out = capsysbinary.readouterr().out
    events = [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]
    assert events and events[-1]["type"] == "error"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1  # 終端はちょうど 1 つ
    return events[-1]


def test_machine_cancelled_on_interrupt(tmp_path, capsysbinary, monkeypatch):
    # 変換段の KeyboardInterrupt → cancelled の error イベント・exit 130。出力は書かれない(原子性)。
    _stub_read(monkeypatch)
    monkeypatch.setattr(cli, "generate_morph_keys", _raise_keyboard_interrupt)
    src = _touch(tmp_path / "in.vpr")
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out), "--machine"])
    assert rc == 130
    e = _machine_error(capsysbinary)
    assert e["code"] == "cancelled" and e["exit_code"] == 130 and e["field"] is None
    assert not out.exists()


def test_non_machine_cancelled_on_interrupt(tmp_path, capsys, monkeypatch):
    # 非機械モードの中断は stdout に JSON を出さず理由を標準エラーへ 1 行、exit 130。出力は残らない。
    _stub_read(monkeypatch)
    monkeypatch.setattr(cli, "generate_morph_keys", _raise_keyboard_interrupt)
    src = _touch(tmp_path / "in.vpr")
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out)])
    assert rc == 130
    cap = capsys.readouterr()
    assert "error:" in cap.err.lower()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")
    assert not out.exists()
