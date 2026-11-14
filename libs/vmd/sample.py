def perspective_series(keys, frame_start, frame_end):
    """keys はフレーム昇順であること。"""
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
