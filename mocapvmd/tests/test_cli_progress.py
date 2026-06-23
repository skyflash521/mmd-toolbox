"""mocapvmd CLI の進捗表示配線のテスト(実装計画 §4・§6.4)。

進捗表示は副作用専用で出力VMDを変えない。--quiet で無効化(表示器 enabled=False)、既定は TTY 自動判定
(enabled=None)。クリーニング・足IK安定化・疎化の各段が begin_stage/end_stage で囲まれ、重い疎化は
reporter.update を progress コールバックとして受け取る。表示器は注入(差し替え)で配線を検証する。
"""

import pytest

from mmd_toolbox.vmd import io
from mocapvmd import cli

from .helpers import bone, write_vmd


def _curve_doc(path):
    # センターの曲線(疎化でキーが減る)。疎化段で progress(0,1)→progress(1,1) が呼ばれる。
    write_vmd(path, bone=[bone("センター", f, pos=(round(0.05 * f * f, 6), 0.0, 0.0)) for f in range(11)])


class _RecordingReporter:
    """ProgressReporter 差し替え用。構築引数(enabled)と段操作を記録する。実描画はしない。"""

    instances = []

    def __init__(self, stream=None, *, enabled=None, now=None, interval=0.15):
        self.enabled = enabled
        self.events = []
        _RecordingReporter.instances.append(self)

    def begin_stage(self, label):
        self.events.append(("begin", label))

    def update(self, done, total):
        self.events.append(("update", done, total))

    def end_stage(self):
        self.events.append(("end",))

    def close(self):
        self.events.append(("close",))


def test_quiet_does_not_change_output(tmp_path):
    # 進捗表示は副作用専用: --quiet あり/なしで出力VMDのボーンが完全一致する。
    src = tmp_path / "in.vmd"
    out_q = tmp_path / "q.vmd"
    out_n = tmp_path / "n.vmd"
    _curve_doc(src)
    assert cli.main([str(src), "-o", str(out_q), "--quiet"]) == 0
    assert cli.main([str(src), "-o", str(out_n)]) == 0
    assert io.read(str(out_q))[0].bone == io.read(str(out_n))[0].bone


def test_quiet_sets_enabled_false_default_auto(tmp_path, monkeypatch):
    # --quiet は表示器を enabled=False で作る(無効化)。既定は enabled=None(stream.isatty で自動判定)。
    _RecordingReporter.instances = []
    monkeypatch.setattr(cli.progress, "ProgressReporter", _RecordingReporter)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _curve_doc(src)
    assert cli.main([str(src), "-o", str(out), "--quiet"]) == 0
    assert _RecordingReporter.instances[-1].enabled is False
    assert cli.main([str(src), "-o", str(out)]) == 0
    assert _RecordingReporter.instances[-1].enabled is None


def test_stages_wired_and_reduce_gets_update(tmp_path, monkeypatch):
    # クリーニング→足IK安定化→疎化の順に begin_stage で囲まれ、疎化段で reporter.update が呼ばれる
    # (reduce_bones が progress=reporter.update を受け取る)。各段に end があり、最後に close。
    _RecordingReporter.instances = []
    monkeypatch.setattr(cli.progress, "ProgressReporter", _RecordingReporter)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _curve_doc(src)
    assert cli.main([str(src), "-o", str(out)]) == 0
    rep = _RecordingReporter.instances[-1]
    assert [e[1] for e in rep.events if e[0] == "begin"] == ["クリーニング", "足IK安定化", "疎化"]
    assert any(e[0] == "update" for e in rep.events)  # 疎化が update を呼ぶ
    assert rep.events.count(("end",)) == 3
    assert rep.events[-1] == ("close",)


def test_close_called_even_when_pipeline_raises(tmp_path, monkeypatch):
    # try/finally 保証: 段の途中(疎化)で例外が起きても、例外を伝播する前に finally で close() される
    # (ハートビートを止め行を確定する例外時経路)。
    _RecordingReporter.instances = []
    monkeypatch.setattr(cli.progress, "ProgressReporter", _RecordingReporter)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli.reduce, "reduce_bones", _boom)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _curve_doc(src)
    with pytest.raises(RuntimeError):
        cli.main([str(src), "-o", str(out)])
    assert ("close",) in _RecordingReporter.instances[-1].events
