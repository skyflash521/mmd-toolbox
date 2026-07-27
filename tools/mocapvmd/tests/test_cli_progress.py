"""mocapvmd CLI の進捗表示配線のテスト。

進捗表示は副作用専用で出力VMDを変えない。--quiet で無効化(表示器 enabled=False)、既定は TTY 自動判定
(enabled=None)。クリーニング・足IK安定化・疎化の各段を 1 本の行で stage 切り替えし、重い疎化は
reporter.update を progress コールバックとして受け取る。出力書き込み成功時のみ完了行(summary)を出す。
表示器は注入(差し替え)で配線を検証する。
"""

from mocapvmd import cli
from vmd import io

from .helpers import bone, write_vmd


def _curve_doc(path):
    # センターの曲線(疎化でキーが減る)。疎化段で progress(0,1)→progress(1,1) が呼ばれる。
    write_vmd(path, bone=[bone("センター", f, pos=(round(0.05 * f * f, 6), 0.0, 0.0)) for f in range(11)])


class _RecordingReporter:
    """ProgressReporter 差し替え用。構築引数(enabled)と段操作・完了行を記録する。実描画はしない。"""

    instances = []

    def __init__(self, stream=None, *, enabled=None, now=None, interval=0.15):
        self.enabled = enabled
        self.events = []
        _RecordingReporter.instances.append(self)

    def stage(self, label):
        self.events.append(("stage", label))

    def update(self, done, total):
        self.events.append(("update", done, total))

    def close(self):
        self.events.append(("close",))

    def summary(self, message):
        self.events.append(("summary", message))


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
    assert cli.main([str(src), "-o", str(out), "--overwrite"]) == 0
    assert _RecordingReporter.instances[-1].enabled is None


def test_stages_wired_reduce_update_and_completion(tmp_path, monkeypatch):
    # クリーニング→足IK安定化→疎化の順に 1 行で stage 切り替えし、疎化段で reporter.update が呼ばれる。
    # 最後に close(行を消す)→ 出力書き込み後に完了行 summary("完了 <出力>") を出す。
    _RecordingReporter.instances = []
    monkeypatch.setattr(cli.progress, "ProgressReporter", _RecordingReporter)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _curve_doc(src)
    assert cli.main([str(src), "-o", str(out)]) == 0
    rep = _RecordingReporter.instances[-1]
    assert [e[1] for e in rep.events if e[0] == "stage"] == ["ノイズ軽減", "足IK接地安定化", "キーフレーム圧縮"]
    assert any(e[0] == "update" for e in rep.events)  # 疎化(キーフレーム圧縮)が update を呼ぶ
    assert ("close",) in rep.events
    assert rep.events[-1] == ("summary", f"完了 {out}")  # 書き込み後に完了行を出す


def test_disabled_stages_are_skipped(tmp_path, monkeypatch):
    # 無効化された段は stage を呼ばない。クリーニング・足IK安定化を切ると疎化だけが stage される。
    # さらに疎化も切ると stage は1つも呼ばれない。配線の段スキップ分岐を固定する。
    _RecordingReporter.instances = []
    monkeypatch.setattr(cli.progress, "ProgressReporter", _RecordingReporter)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _curve_doc(src)
    assert cli.main([str(src), "-o", str(out), "--no-denoise", "--no-foot-ik-stabilize"]) == 0
    assert [e[1] for e in _RecordingReporter.instances[-1].events if e[0] == "stage"] == ["キーフレーム圧縮"]
    assert cli.main([
        str(src), "-o", str(out), "--overwrite", "--no-denoise", "--no-foot-ik-stabilize", "--no-reduce",
    ]) == 0
    assert [e[1] for e in _RecordingReporter.instances[-1].events if e[0] == "stage"] == []


def test_no_summary_on_dry_run(tmp_path, monkeypatch):
    # dry-run は出力を書かないので完了行(summary)を出さない。進捗の close は通る。
    _RecordingReporter.instances = []
    monkeypatch.setattr(cli.progress, "ProgressReporter", _RecordingReporter)
    src = tmp_path / "in.vmd"
    _curve_doc(src)
    assert cli.main([str(src), "--dry-run"]) == 0
    rep = _RecordingReporter.instances[-1]
    assert ("close",) in rep.events
    assert not any(e[0] == "summary" for e in rep.events)


def test_close_called_and_no_summary_when_pipeline_raises(tmp_path, monkeypatch):
    # try/finally 保証 + internal_error 畳み: 疎化で想定外例外が起きても、finally で close され、
    # 例外はトレースバックを漏らさず internal_error(exit 1)へ畳まれる(伝播しない)。書き込みに
    # 至らないので完了行(summary)は出さない。
    _RecordingReporter.instances = []
    monkeypatch.setattr(cli.progress, "ProgressReporter", _RecordingReporter)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli.reduce, "reduce_bones", _boom)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _curve_doc(src)
    rc = cli.main([str(src), "-o", str(out)])
    assert rc == 1  # internal_error（トレースバックを漏らさず畳む）
    rep = _RecordingReporter.instances[-1]
    assert ("close",) in rep.events
    assert not any(e[0] == "summary" for e in rep.events)
