"""shakevmd CLI の進捗表示・バージョン配線のテスト(shakevmd.md §2.7.1, §8)。

進捗表示は副作用専用(出力VMD・終了コード・統計・警告を変えない)で、端末(TTY)のときだけ stderr へ
1行ライブ表示し、`--quiet` で抑制する。`--version` は `__version__` を表示して終了コード0で返る。

配線の検証は、CLI が生成する ProgressReporter を差し替え可能な spy(_SpyReporter)へ monkeypatch して
段・update・close・summary の呼ばれ方を観測する方式と、実 stderr を TTY 扱いの stream に差し替えて
実表示を有効化し出力VMDのバイト不変を観る方式を使い分ける。
"""

import io
import sys

import pytest

from shakevmd import cli, progress
from vmd import io as vmd_io
from vmd.types import CameraKey, VmdDocument

# 実装は後続フェーズで入る。未実装のうちは各テストを xfail で印を付け、緑コミットを保つ。
pytestmark = pytest.mark.xfail(reason="impl pending: Step3 cli wiring", strict=False)

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


class _SpyReporter:
    """ProgressReporter の差し替え用 spy。段・update・close・summary の呼び出しを記録する。

    CLI の配線(どの段を出すか・reduce へ progress を渡すか・異常/早期 return でも close するか・
    成功時のみ summary するか)を、実表示に依存せず観測する。常に有効として全呼び出しを記録する。
    """

    instances = []

    def __init__(self, stream=None, *, enabled=None, now=None, interval=0.15):
        self.stream = stream
        self.enabled = True
        self.stages = []
        self.updates = []
        self.closed = 0
        self.summaries = []
        _SpyReporter.instances.append(self)

    def stage(self, label):
        self.stages.append(label)

    def update(self, done, total, note=""):
        self.updates.append((done, total, note))

    def close(self):
        self.closed += 1

    def summary(self, message):
        self.summaries.append(message)


def _spy(monkeypatch):
    _SpyReporter.instances.clear()
    monkeypatch.setattr(progress, "ProgressReporter", _SpyReporter)


# --- --version(§8) -------------------------------------------------------
def test_version_shows_version_and_exit0(capsys):
    # --version は __version__ を表示して終了コード0で返る(版の番号源は __version__ 一本)。
    from shakevmd import __version__
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "shakevmd" in out
    assert __version__ in out


# --- 副作用専用・TTY 限定・no-op(§2.7.1) --------------------------------
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
def test_no_smooth_skips_smoothing_stage(tmp_path, monkeypatch):
    # --no-smooth 時は平滑化段を出さない(ベイク段のみ)。
    inp = write_input(tmp_path / "in.vmd")
    _spy(monkeypatch)
    assert cli.main([inp, "-o", str(tmp_path / "a.vmd"), "--no-smooth"]) == 0
    spy = _SpyReporter.instances[-1]
    assert "ベイク" in spy.stages
    assert "平滑化" not in spy.stages


def test_smooth_emits_smoothing_stage(tmp_path, monkeypatch):
    # 既定(--smooth)では平滑化段を出す。
    inp = write_input(tmp_path / "in.vmd")
    _spy(monkeypatch)
    assert cli.main([inp, "-o", str(tmp_path / "a.vmd")]) == 0
    spy = _SpyReporter.instances[-1]
    assert "ベイク" in spy.stages
    assert "平滑化" in spy.stages


def test_smooth_wires_progress_to_reduce(tmp_path, monkeypatch):
    # 平滑化段は reduce へ progress=reporter.update を接続する。reduce 呼び出しを差し替えて
    # progress kwarg を捕捉し、それが reporter.update そのものであること、かつ呼ぶと spy に届くことを
    # 直接検証する(CLI が手動で update を呼ぶだけの実装では通らない)。
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
    spy = _SpyReporter.instances[-1]
    # reduce へ reporter.update を接続していること。bound method は属性アクセス毎に別オブジェクトに
    # なり `is spy.update` は偽陰性になるため、束縛先(__self__)と関数(__func__)で同一性を判定する。
    bound = captured.get("progress")
    assert bound is not None
    assert bound.__self__ is spy and bound.__func__ is _SpyReporter.update
    assert (3, 10, "") in spy.updates                 # その progress を呼ぶと spy に届く


# --- summary / close の配線 ----------------------------------------------
def test_summary_on_success(tmp_path, monkeypatch):
    # 書き込み成功後に完了行 summary("完了 <出力パス>") を1回だけ出す。
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    _spy(monkeypatch)
    assert cli.main([inp, "-o", str(out), "--no-smooth"]) == 0
    spy = _SpyReporter.instances[-1]
    assert spy.summaries == [f"完了 {out}"]


def test_no_summary_on_dry_run(tmp_path, monkeypatch):
    # --dry-run は出力を書かないため summary を出さない(進捗行は close で消す)。
    inp = write_input(tmp_path / "in.vmd")
    _spy(monkeypatch)
    assert cli.main([inp, "--dry-run"]) == 0
    spy = _SpyReporter.instances[-1]
    assert spy.summaries == []


def test_close_called_on_bake_error(tmp_path, monkeypatch):
    # stage 開始後に bake が異常(範囲重複→ValueError→exit 2)でも close が呼ばれる(try/finally)。
    inp = write_input(tmp_path / "in.vmd")
    _spy(monkeypatch)
    assert cli.main([inp, "--range", "0:60", "--range", "30:60"]) == 2
    spy = _SpyReporter.instances[-1]
    assert spy.closed >= 1


def test_close_called_on_dry_run(tmp_path, monkeypatch):
    # --dry-run の早期 return でも close が呼ばれ、heartbeat スレッドが残らない(try/finally)。
    inp = write_input(tmp_path / "in.vmd")
    _spy(monkeypatch)
    assert cli.main([inp, "--dry-run"]) == 0
    spy = _SpyReporter.instances[-1]
    assert spy.closed >= 1
