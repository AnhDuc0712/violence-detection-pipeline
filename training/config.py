# training/config.py

FEATURE_COLUMNS = [
    "velocity_mean",
    "velocity_max",
    "accel_mean",
    "bbox_scale_var",
    "trajectory_curvature",
    "limb_amplitude",
    "joint_angle_var",
    "pose_motion_energy",

    # flags
    "flag_reflection",
    "flag_occlusion",
    "flag_id_switch",
    "flag_shadow_like",
    "flag_camera_motion",
]


LABEL_COLUMN = "label"
TRACKLET_ID = "tracklet_id"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_DIR = "models/classifier"
