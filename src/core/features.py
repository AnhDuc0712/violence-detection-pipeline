import numpy as np
from src.utils.geometry import bbox_center, bbox_height, euclid, get_angle

# =========================
# KEYPOINT INDEX
# =========================
KP = {
    "L_SHOULDER": 5, "R_SHOULDER": 6,
    "L_ELBOW": 7, "R_ELBOW": 8,
    "L_WRIST": 9, "R_WRIST": 10,
    "L_HIP": 11, "R_HIP": 12,
    "L_KNEE": 13, "R_KNEE": 14,
    "L_ANKLE": 15, "R_ANKLE": 16
}


# =========================
# TORSO SCALE
# =========================
def get_torso_scale(kps):
    scales = []
    for f in kps:
        w = np.linalg.norm(f[KP["L_SHOULDER"], :2] - f[KP["R_SHOULDER"], :2])
        h = np.linalg.norm(f[KP["L_SHOULDER"], :2] - f[KP["L_HIP"], :2])
        scales.append(max(w, h, 1.0))
    return np.median(scales) if scales else 1.0


# =========================
# FEATURE EXTRACTION
# =========================
def compute_features_advanced(kps, bbox_seq, interaction_dist, cached_scale, PEAK_VEL_THRESH):
    scale = cached_scale

    centers = np.array([bbox_center(b) for b in bbox_seq])
    heights = np.array([bbox_height(b) for b in bbox_seq])

    kps_hip_center = (kps[:, KP["L_HIP"], :2] + kps[:, KP["R_HIP"], :2]) / 2
    vel = np.linalg.norm(np.diff(kps_hip_center, axis=0), axis=1) / scale
    accel = np.abs(np.diff(vel)) if len(vel) > 1 else np.array([0.0])

    # =========================
    # WRIST CONF FILTER
    # =========================
    l_wrist = kps[:, KP["L_WRIST"]]
    r_wrist = kps[:, KP["R_WRIST"]]

    l_conf = l_wrist[:, 2]
    r_conf = r_wrist[:, 2]

    l_xy = l_wrist[:, :2]
    r_xy = r_wrist[:, :2]

    l_w_v = np.linalg.norm(np.diff(l_xy, axis=0), axis=1) / scale
    r_w_v = np.linalg.norm(np.diff(r_xy, axis=0), axis=1) / scale

    l_w_v[l_conf[:-1] < 0.35] = 0
    r_w_v[r_conf[:-1] < 0.35] = 0

    l_w_a = np.abs(np.diff(l_w_v)) if len(l_w_v) > 1 else [0]
    r_w_a = np.abs(np.diff(r_w_v)) if len(r_w_v) > 1 else [0]

    wrist_vel_avg = (np.mean(l_w_v) + np.mean(r_w_v)) / 2
    wrist_jerk_avg = (np.mean(l_w_a) + np.mean(r_w_a)) / 2

    elbows, knees, limb_exp, asym = [], [], [], []

    for f in kps:
        el = (get_angle(f[KP["L_SHOULDER"],:2], f[KP["L_ELBOW"],:2], f[KP["L_WRIST"],:2]) + 
              get_angle(f[KP["R_SHOULDER"],:2], f[KP["R_ELBOW"],:2], f[KP["R_WRIST"],:2])) / 2

        kn = (get_angle(f[KP["L_HIP"],:2], f[KP["L_KNEE"],:2], f[KP["L_ANKLE"],:2]) + 
              get_angle(f[KP["R_HIP"],:2], f[KP["R_KNEE"],:2], f[KP["R_ANKLE"],:2])) / 2

        bc = (f[KP["L_HIP"],:2] + f[KP["R_HIP"],:2]) / 2

        lw = euclid(f[KP["L_WRIST"],:2], bc)
        rw = euclid(f[KP["R_WRIST"],:2], bc)

        elbows.append(el)
        knees.append(kn)
        limb_exp.append(max(lw, rw) / scale)
        asym.append(abs(lw - rw) / scale)

    diffs = np.linalg.norm(np.diff(centers, axis=0), axis=1)

    flag_motion = float(np.mean(diffs) / np.mean(heights)) if len(diffs) else 0.0
    flag_jump = float(np.max(diffs) / np.mean(heights)) if len(diffs) else 0.0
    flag_occ = float(np.mean(np.sum(kps[:,:,0] < 1e-3, axis=1) / 17.0))
    flag_quality = float(max(0.0, 1.0 - flag_occ - min(flag_jump, 1.0)))

    feat = {
        "vel_mean": np.mean(vel) if len(vel) else 0,
        "vel_max": np.max(vel) if len(vel) else 0,
        "accel_max": np.max(accel) if len(accel) else 0,
        "motion_peak_count": int(np.sum(vel > PEAK_VEL_THRESH)),
        "angle_elbow_mean": np.mean(elbows),
        "angle_elbow_var": np.var(elbows),
        "angle_knee_var": np.var(knees),
        "limb_expansion_max": np.max(limb_exp),
        "hand_asymmetry": np.mean(asym),
        "flag_motion": flag_motion,
        "flag_trajectory_jump": flag_jump,
        "flag_occlusion": flag_occ,
        "flag_quality": flag_quality,
        "interaction_dist": interaction_dist,
        "interaction_valid": 1.0 if interaction_dist < 10 else 0.0
    }

    return feat, wrist_vel_avg, wrist_jerk_avg