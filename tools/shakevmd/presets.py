PRESET_NAMES = ("handheld", "telephoto", "walking", "earthquake")

INTERNAL_PARAM_NAMES = (
    "gait_freq", "gait_amp", "still_profile", "moving_profile", "settle_time", "naive_rotation",
    "speed_ref_world", "speed_ref_angle",
)

_PRESETS = {
    "handheld": {
        "amp_rot": 0.9, "amp_pos": 0.06, "rot_weights": (1.0, 1.0, 0.35),
        "freq": 1.3, "motion_damp": 1.0, "settle": 0.4, "cut_threshold": (5.0, 20.0),
    },
    "telephoto": {
        "amp_rot": 1.8, "amp_pos": 0.015, "rot_weights": (1.0, 1.0, 0.2),
        "freq": 0.7, "motion_damp": 1.0, "settle": 0.5, "cut_threshold": (5.0, 20.0),
    },
    "walking": {
        "amp_rot": 1.0, "amp_pos": 0.18, "rot_weights": (1.3, 0.7, 0.3),
        "freq": 2.0, "motion_damp": 0.3, "settle": 0.3, "cut_threshold": (5.0, 20.0),
        "gait_freq": 1.0, "gait_amp": 0.1,
    },
    "earthquake": {
        "amp_rot": 3.5, "amp_pos": 0.5, "rot_weights": (1.0, 1.0, 0.8),
        "freq": 5.0, "motion_damp": 0.0, "settle": 0.6, "cut_threshold": (5.0, 20.0),
    },
}


def get_preset(name: str) -> dict:
    """未知の名前は KeyError。"""
    return dict(_PRESETS[name])
