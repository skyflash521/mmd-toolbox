"""shakevmd CLI の進捗表示・バージョン配線のテスト。

進捗表示は副作用専用(出力VMD・終了コード・統計・警告を変えない)で、端末(TTY)のときだけ stderr へ
1行ライブ表示し、`--quiet` で抑制する。`--version` は `__version__` を表示して終了コード0で返る。

報告先の振り分けの実体は共有の cli_progress_router が持つので、ここでは shakevmd の配線だけを見る。
検証は、CLI が生成する振り分けを差し替え可能な spy(_SpyRouter)へ monkeypatch して段・進行・close・
summary の呼ばれ方を観測する方式と、実 stderr を TTY 扱いの stream に差し替えて実表示を有効化し
出力VMDのバイト不変を観る方式を使い分ける。
"""

import io
import sys

from shakevmd import cli
from vmd import io as vmd_io
from vmd.types import CameraKey, VmdDocument

LINEAR = bytes([20, 107, 20, 107]) * 6


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, LINEAR, fov, persp)


KEYS = [
    cam(0, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(30, dist=-25.0, center=(10.0, 5.0, 2.0), rot=(0.2, 0.1, 0.0), persp=1),
    cam(60, dist=-20.0, center=(20.0, 0.0, -3.0), rot=(-0.1, 0.3, 0.05), persp=1),
]


def write_input(path, keys=KEYS):
    vmd_io.write_file(VmdDocument(camera=list(keys)), str(path))
    return str(path)


class _TTYStringIO(io.StringIO):
    """端末(TTY)扱いの stream。進捗表示の自動有効化(isatty()==True)を起こす。"""

    def isatty(self):
        return True


class _SpyRouter:
    """進捗の振り分けの差し替え用 spy。段の報告・close・summary の呼び出しを記録する。

    CLI の配線(どの段を出すか・reduce へ進行の中継を渡すか・異常/早期 return でも close するか・
    成功時のみ summary するか)を、実表示にも送出にも依存せず観測する。
    """

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.stages = []
        self.updates = []
        self.closed = 0
        self.summaries = []
        _SpyRouter.instances.append(self)

    def stage(self, stage_id, *, done=0, total=None, note="", elapsed=0.0):
        self.stages.append(stage_id)
        self.updates.append((stage_id, done, total, note))

    def close(self):
        self.closed += 1

    def summary(self, message):
        self.summaries.append(message)


def _spy(monkeypatch):
    _SpyRouter.instances.clear()
    monkeypatch.setattr(cli, "ProgressRouter", _SpyRouter)


# --- --version -------------------------------------------------------
def test_version_shows_version_and_exit0(capsys):
    # --version は __version__ を表示して終了コード0で返る(バージョンの番号源は __version__ 一本)。
    from shakevmd import __version__
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "shakevmd" in out
    assert __version__ in out


# --- 副作用専用・TTY 限定・no-op --------------------------------
def test_progress_noop_when_not_tty(tmp_path, capsys):
    # 非TTY(テスト捕捉)では進捗を1バイトも出さない(完全 no-op)。KEYS は警告を出さない入力なので
    # stderr は空でなければならない(マーカー不在だけでなく、消去シーケンス等の進捗由来バイトも出ない)。
    inp = write_input(tmp_path / "in.vmd")
    assert cli.main([inp, "-o", str(tmp_path / "out.vmd")]) == 0
    assert capsys.readouterr().err == ""


def test_progress_side_effect_only_bytes_identical(tmp_path, monkeypatch):
    # 進捗表示の有無で出力VMDがバイト一致(副作用専用)。--quiet(無効)と TTY 有効を突き合わせる。
    inp = write_input(tmp_path / "in.vmd")
    out_quiet = tmp_path / "quiet.vmd"
    assert cli.main([inp, "-o", str(out_quiet), "--quiet"]) == 0
    # stderr を TTY 扱いに差し替えて進捗表示を有効化(実 ProgressReporter が走る)。
    monkeypatch.setattr(sys, "stderr", _TTYStringIO())
    out_live = tmp_path / "live.vmd"
    assert cli.main([inp, "-o", str(out_live)]) == 0
    assert out_live.read_bytes() == out_quiet.read_bytes()


def test_quiet_suppresses_progress_even_on_tty(tmp_path, monkeypatch):
    # TTY でも --quiet なら進捗を出さない(完了行も含めて)。KEYS は警告なしなので、抑制が効いていれば
    # stderr は空(マーカー不在だけでなく消去シーケンス等の進捗由来バイトも出ない)。
    inp = write_input(tmp_path / "in.vmd")
    monkeypatch.setattr(sys, "stderr", _TTYStringIO())
    assert cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--quiet"]) == 0
    assert sys.stderr.getvalue() == ""


def test_quiet_does_not_change_nonprogress_output(tmp_path, capsys):
    # --quiet が抑制するのは進捗だけ。非TTYでは進捗は元々 no-op なので、--quiet の有無で警告・統計・
    # 標準出力・終了コードが完全一致しなければならない(警告/統計の値や有無を変える実装を排除)。
    dup = [cam(0), cam(30), cam(30, center=(9.0, 9.0, 9.0)), cam(60)]  # 正規化警告+統計を発生させる
    inp = write_input(tmp_path / "dup.vmd", dup)
    rc_plain = cli.main([inp, "-o", str(tmp_path / "a.vmd"), "--dry-run"])
    plain = capsys.readouterr()
    rc_quiet = cli.main([inp, "-o", str(tmp_path / "b.vmd"), "--quiet", "--dry-run"])
    quiet = capsys.readouterr()
    assert rc_plain == rc_quiet == 0
    assert (quiet.out, quiet.err) == (plain.out, plain.err)   # 警告・統計は --quiet で不変
    assert "warning:" in (plain.out + plain.err).lower()      # 実際に警告・統計が出ている前提を固定


def test_quiet_does_not_change_exit_code(tmp_path):
    # 終了コードは --quiet で変わらない(上書きガードは --quiet 有無のどちらでも exit 2)。
    inp = write_input(tmp_path / "in.vmd")
    assert cli.main([inp, "-o", inp]) == cli.main([inp, "-o", inp, "--quiet"]) == 2


# --- 段構成の配線 --------------------------------------------------------
def test_router_receives_mode_quiet_stream_and_labels(tmp_path, monkeypatch):
    # 段 id と利用者向けの工程名の対応表は shakevmd が持ち、構築時に振り分けへ渡す。
    inp = write_input(tmp_path / "in.vmd")
    _spy(monkeypatch)
    assert cli.main([inp, "-o", str(tmp_path / "a.vmd"), "--quiet", "--no-smooth"]) == 0
    kwargs = _SpyRouter.instances[-1].kwargs
    assert kwargs["machine"] is False and kwargs["quiet"] is True
    assert kwargs["emitter"] is None and kwargs["stream"] is sys.stderr
    assert kwargs["labels"] == {"bake": "ベイク", "smooth": "スムージング"}

    assert cli.main([inp, "-o", str(tmp_path / "b.vmd"), "--machine", "--no-smooth"]) == 0
    kwargs = _SpyRouter.instances[-1].kwargs
    assert kwargs["machine"] is True and kwargs["quiet"] is False and kwargs["emitter"] is not None


def test_no_smooth_skips_smoothing_stage(tmp_path, monkeypatch):
    # --no-smooth 時はスムージング段を出さない(ベイク段のみ)。
    inp = write_input(tmp_path / "in.vmd")
    _spy(monkeypatch)
    assert cli.main([inp, "-o", str(tmp_path / "a.vmd"), "--no-smooth"]) == 0
    spy = _SpyRouter.instances[-1]
    assert "bake" in spy.stages
    assert "smooth" not in spy.stages


def test_smooth_emits_smoothing_stage(tmp_path, monkeypatch):
    # 既定(--smooth)ではスムージング段を出す。
    inp = write_input(tmp_path / "in.vmd")
    _spy(monkeypatch)
    assert cli.main([inp, "-o", str(tmp_path / "a.vmd")]) == 0
    spy = _SpyRouter.instances[-1]
    assert "bake" in spy.stages
    assert "smooth" in spy.stages


def test_smooth_wires_progress_to_reduce(tmp_path, monkeypatch):
    # スムージング段は reduce へ進行の中継を接続する。reduce 呼び出しを差し替えて progress kwarg を
    # 捕捉し、それを呼ぶとスムージング段の進行として届くことを直接検証する(CLI が手動で進行を
    # 報告するだけの実装では通らない)。
    inp = write_input(tmp_path / "in.vmd")
    _spy(monkeypatch)
    captured = {}
    real_reduce = cli.reduce_camera_track

    def fake_reduce(*args, progress=None, **kwargs):
        captured["progress"] = progress
        if progress is not None:
            progress(3, 10, "")          # reduce が progress を呼ぶ様子を模す
        return real_reduce(*args, progress=None, **kwargs)

    monkeypatch.setattr(cli, "reduce_camera_track", fake_reduce)
    assert cli.main([inp, "-o", str(tmp_path / "out.vmd")]) == 0
    spy = _SpyRouter.instances[-1]
    assert captured.get("progress") is not None
    assert ("smooth", 3, 10, "") in spy.updates       # その progress を呼ぶと同じ段の進行として届く


# --- summary / close の配線 ----------------------------------------------
def test_summary_on_success(tmp_path, monkeypatch):
    # 書き込み成功後に完了行 summary("完了 <出力パス>") を1回だけ出す。
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    _spy(monkeypatch)
    assert cli.main([inp, "-o", str(out), "--no-smooth"]) == 0
    spy = _SpyRouter.instances[-1]
    assert spy.summaries == [f"完了 {out}"]


def test_no_summary_on_dry_run(tmp_path, monkeypatch):
    # --dry-run は出力を書かないため summary を出さない(進捗行は close で消す)。
    inp = write_input(tmp_path / "in.vmd")
    _spy(monkeypatch)
    assert cli.main([inp, "--dry-run"]) == 0
    spy = _SpyRouter.instances[-1]
    assert spy.summaries == []


def test_close_called_on_bake_error(tmp_path, monkeypatch):
    # stage 開始後に bake が異常(範囲重複→ValueError→exit 2)でも close が呼ばれる(try/finally)。
    inp = write_input(tmp_path / "in.vmd")
    _spy(monkeypatch)
    assert cli.main([inp, "--range", "0:60", "--range", "30:60"]) == 2
    spy = _SpyRouter.instances[-1]
    assert spy.closed >= 1


def test_close_called_on_dry_run(tmp_path, monkeypatch):
    # --dry-run の早期 return でも close が呼ばれ、heartbeat スレッドが残らない(try/finally)。
    inp = write_input(tmp_path / "in.vmd")
    _spy(monkeypatch)
    assert cli.main([inp, "--dry-run"]) == 0
    spy = _SpyRouter.instances[-1]
    assert spy.closed >= 1
