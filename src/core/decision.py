def classify_interaction_state(feat, wrist_vel, wrist_jerk, raw_prob, COLORS):
    dist = feat['interaction_dist']
    asym = feat['hand_asymmetry']

    if dist < 1.2:
        if wrist_jerk < 0.12 and asym < 0.20:
            return -0.5, "HUGGING", COLORS["HUGGING"]
        elif wrist_jerk > 0.25 or asym > 0.4:
            return +0.25, "GRAPPLING", COLORS["GRAPPLING"]
        else:
            return -0.1, "CLOSE", COLORS["CLOSE"]
    else:
        if wrist_vel > 0.50 and wrist_jerk > 0.20:
            return +0.35, "STRIKING", COLORS["STRIKING"]
        elif feat['vel_mean'] > 0.5:
            return -0.2, "RUNNING", COLORS["RUNNING"]

    return 0.0, "NEUTRAL", COLORS["NEUTRAL"]