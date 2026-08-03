"""mocapvmd CLI の進捗配線のテスト。

進捗表示は副作用専用で出力VMDを変えない。報告先の振り分け(機械モードのイベント送出と人間向け
ライブ表示の切り替え)の実体は共有の cli_progress_router が持つので、ここでは mocapvmd の配線
——渡す設定、各段を段 id で報告する順序、疎化の進行更新、close と完了行の位置——だけを検証する。
振り分けは注入(差し替え)で観測する。
"""

import sys

from mocapvmd import cli
from vmd import io

from .helpers import bone, write_vmd


def _curve_doc(path):
    # センターの曲線(疎化でキーが減る)。疎化段で progress(0,1)→progress(1,1) が呼ばれる。
    write_vmd(path, bone=[bone("センター", f, pos=(round(0.05 * f * f, 6), 0.0, 0.0)) for f in range(11)])


class _RecordingRouter:
    """振り分けの差し替え用。構築引数と段操作・完了行を記録する。実描画も送出もしない。"""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.events = []
        _RecordingRouter.instances.append(self)

    def stage(self, stage_id, **kwargs):
        self.events.append(("stage", stage_id, kwargs))

    def close(self):
        self.events.append(("close",))

    def summary(self, message):
        self.events.append(("summary", message))


def _install(monkeypatch):
    _RecordingRouter.instances = []
    monkeypatch.setattr(cli, "ProgressRouter", _RecordingRouter)


def _stage_ids(router):
    return [e[1] for e in router.events if e[0] == "stage"]


def test_quiet_does_not_change_output(tmp_path):
    # 進捗表示は副作用専用: --quiet あり/なしで出力VMDのボーンが完全一致する。
    src = tmp_path / "in.vmd"
    out_q = tmp_path / "q.vmd"
    out_n = tmp_path / "n.vmd"
    _curve_doc(src)
    assert cli.main([str(src), "-o", str(out_q), "--quiet"]) == 0
    assert cli.main([str(src), "-o", str(out_n)]) == 0
    assert io.read(str(out_q))[0].bone == io.read(str(out_n))[0].bone


def test_router_receives_mode_quiet_stream_and_labels(tmp_path, monkeypatch):
    _install(monkeypatch)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _curve_doc(src)

    assert cli.main([str(src), "-o", str(out), "--quiet"]) == 0
    kwargs = _RecordingRouter.instances[-1].kwargs
    assert kwargs["machine"] is False and kwargs["quiet"] is True
    assert kwargs["emitter"] is None and kwargs["stream"] is sys.stderr
    assert kwargs["labels"] == {
        "denoise": "ノイズ軽減", "foot_ik": "足IK接地安定化", "reduce": "キーフレーム圧縮"}

    assert cli.main([str(src), "-o", str(out), "--overwrite"]) == 0
    assert _RecordingRouter.instances[-1].kwargs["quiet"] is False

    assert cli.main([str(src), "-o", str(out), "--overwrite", "--machine"]) == 0
    kwargs = _RecordingRouter.instances[-1].kwargs
    assert kwargs["machine"] is True and kwargs["emitter"] is not None


def test_stages_are_reported_in_order_and_reduce_updates_progress(tmp_path, monkeypatch):
    # クリーニング→足IK安定化→疎化の順に段を報告し、疎化は同じ段 id で進行を更新する。
    # 最後に close(行を消す)→ 出力書き込み後に完了行 summary("完了 <出力>") を出す。
    _install(monkeypatch)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _curve_doc(src)
    assert cli.main([str(src), "-o", str(out)]) == 0
    router = _RecordingRouter.instances[-1]
    assert _stage_ids(router)[:3] == ["denoise", "foot_ik", "reduce"]
    updates = [e for e in router.events if e[0] == "stage" and e[1] == "reduce" and e[2]]
    assert updates  # 疎化は done/total を伴って同じ段を更新する
    assert all("elapsed" in e[2] for e in updates)
    assert ("close",) in router.events
    assert router.events[-1] == ("summary", f"完了 {out}")  # 書き込み後に完了行を出す


def test_disabled_stages_are_skipped(tmp_path, monkeypatch):
    # 無効化された段は報告しない。クリーニング・足IK安定化を切ると疎化だけが報告される。
    # さらに疎化も切ると段は1つも報告されない。配線の段スキップ分岐を固定する。
    _install(monkeypatch)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _curve_doc(src)
    assert cli.main([str(src), "-o", str(out), "--no-denoise", "--no-foot-ik-stabilize"]) == 0
    assert set(_stage_ids(_RecordingRouter.instances[-1])) == {"reduce"}
    assert cli.main([
        str(src), "-o", str(out), "--overwrite", "--no-denoise", "--no-foot-ik-stabilize", "--no-reduce",
    ]) == 0
    assert _stage_ids(_RecordingRouter.instances[-1]) == []


def test_no_summary_on_dry_run(tmp_path, monkeypatch):
    # dry-run は出力を書かないので完了行(summary)を出さない。進捗の close は通る。
    _install(monkeypatch)
    src = tmp_path / "in.vmd"
    _curve_doc(src)
    assert cli.main([str(src), "--dry-run"]) == 0
    router = _RecordingRouter.instances[-1]
    assert ("close",) in router.events
    assert not any(e[0] == "summary" for e in router.events)


def test_close_called_and_no_summary_when_pipeline_raises(tmp_path, monkeypatch):
    # try/finally 保証 + internal_error 畳み: 疎化で想定外例外が起きても、finally で close され、
    # 例外はトレースバックを漏らさず internal_error(exit 1)へ畳まれる(伝播しない)。書き込みに
    # 至らないので完了行(summary)は出さない。
    _install(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli.reduce, "reduce_bones", _boom)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _curve_doc(src)
    rc = cli.main([str(src), "-o", str(out)])
    assert rc == 1  # internal_error（トレースバックを漏らさず畳む）
    router = _RecordingRouter.instances[-1]
    assert ("close",) in router.events
    assert not any(e[0] == "summary" for e in router.events)
