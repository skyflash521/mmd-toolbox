"""サンプリング小ヘルパ(vmd: perspective 直近ホールド。元 sparsevmd.md §4.2)。

perspective は補間しない離散値で、各フレームの値を直近キーの値で保持する。
スカラー/回転のサンプリング本体は vmd.interp が担う。
"""


def perspective_series(keys, frame_start, frame_end):
    """各フレームの perspective を直近キー値で保持して返す(§4.2)。

    当該フレーム以前にキーが無い場合は先頭キーの値を用いる
    (vmd-interp の境界規約「最初のキー以前は端キーの値で一定」に倣う)。
    """
    if not keys:
        raise ValueError("キー列が空")
    series = []
    idx = 0
    n = len(keys)
    for f in range(frame_start, frame_end + 1):
        while idx + 1 < n and keys[idx + 1].frame <= f:
            idx += 1
        series.append(keys[idx].perspective)
    return series
