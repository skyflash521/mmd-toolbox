"""カット検出・セグメント分割(shakevmd.md §5.3)。

カット = 「フレームFから新しいショットが始まる」境界。セグメント境界は F-1 と F の
間に置く。bake/noise/motion はここが返すセグメント列を入力として受け取り、
カットの波及(位相独立・速度解析の分割・settle抑制・フェード不要)を構造で保証する。

検出はフレーム差1の隣接カメラキー間で行う:
  - カメラ中心位置のジャンプ、または
  - カメラワールド位置のジャンプ(距離のみ急変するズームカット検出のため)が
    pos_threshold(MMD距離単位)を超える、または
  - 角度ジャンプが rot_threshold(度)を超える
手動指定(add/remove)は自動検出に優先する(コアAPI)。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """ベイク範囲を分割した1セグメント [start, end](両端含む、フレーム番号)。"""

    start: int
    end: int


def detect_cuts(keys, pos_threshold: float, rot_threshold: float) -> list[int]:
    """フレーム差1の隣接カメラキー間でカットを検出し、カットフレーム(後側)を昇順で返す。

    keys はフレーム昇順前提(vmd-io.md §5 正規化済み)。
    各隣接ペア (k0, k1) で k1.frame - k0.frame == 1 のとき:
      max(カメラ中心位置ジャンプ, カメラワールド位置ジャンプ) > pos_threshold、または
      角度ジャンプ(度) > rot_threshold なら k1.frame をカットとする。
    """
    raise NotImplementedError


def resolve_cuts(
    detected: list[int], add: list[int] | None = None, remove: list[int] | None = None
) -> list[int]:
    """自動検出に手動指定を反映する(手動優先)。add は強制追加、remove は打ち消し。

    戻り値は昇順・重複なしのカットフレーム列。
    """
    raise NotImplementedError


def segment_bounds(frame_start: int, frame_end: int, cuts: list[int]) -> list[Segment]:
    """[frame_start, frame_end] をカットで分割したセグメント列を返す。

    カットフレーム F は境界を F-1 と F の間に置く(F は次セグメントの先頭)。
    範囲内のカットのみ作用する。カットがなければ単一セグメント。
    """
    raise NotImplementedError
