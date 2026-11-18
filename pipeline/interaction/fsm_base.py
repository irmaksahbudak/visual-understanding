"""Base FSM for visual interaction analysis.

Faithfully extracted from original notebook cells 61 and 62.
Input: merged SAM 2 object measurements and CoTracker3 gripper measurements.
No existing FSM label CSV is needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "frame", "time_s", "center_x_px", "center_y_px", "mask_area_px",
    "gripper_body_x_px", "gripper_body_y_px",
    "left_jaw_x_px", "left_jaw_y_px", "right_jaw_x_px", "right_jaw_y_px",
    "left_jaw_visible", "right_jaw_visible",
]


def analyze_base_interactions(interaction_data: pd.DataFrame, output_dir=None) -> pd.DataFrame:
    """Apply original FSM and hysteresis rules; optionally save resulting CSV.

    This function intentionally preserves all thresholds and transition behavior
    from the user's original notebook cells 61 and 62.
    """
    missing = [name for name in REQUIRED_COLUMNS if name not in interaction_data.columns]
    if missing:
        raise ValueError(f"Missing interaction columns: {missing}")

    df = interaction_data.copy().sort_values("frame").reset_index(drop=True)
    if df["frame"].duplicated().any():
        raise ValueError("Duplicate frame indices in input data")
    if df.empty:
        raise ValueError("Input interaction table is empty")

    # Original cell 61: five-frame centered median smoothing.
    columns = [
        "center_x_px", "center_y_px", "gripper_body_x_px", "gripper_body_y_px",
        "left_jaw_x_px", "left_jaw_y_px", "right_jaw_x_px", "right_jaw_y_px",
    ]
    for col in columns:
        df[col + "_s"] = df[col].rolling(5, center=True, min_periods=3).median()

    jaw_x = (df["left_jaw_x_px_s"] + df["right_jaw_x_px_s"]) / 2
    jaw_y = (df["left_jaw_y_px_s"] + df["right_jaw_y_px_s"]) / 2
    object_x = df["center_x_px_s"]
    object_y = df["center_y_px_s"]
    distance = np.hypot(jaw_x - object_x, jaw_y - object_y)
    jaw_width = np.hypot(
        df["left_jaw_x_px_s"] - df["right_jaw_x_px_s"],
        df["left_jaw_y_px_s"] - df["right_jaw_y_px_s"],
    )

    window = 5
    gx = jaw_x.diff(window)
    gy = jaw_y.diff(window)
    ox = object_x.diff(window)
    oy = object_y.diff(window)
    g_move = np.hypot(gx, gy)
    o_move = np.hypot(ox, oy)
    similarity = (gx * ox + gy * oy) / (g_move * o_move).replace(0, np.nan)
    relative_error = np.hypot(gx - ox, gy - oy)
    jaw_change = jaw_width.diff(10)
    distance_change = distance.diff(window)

    def visible_column(name):
        return df[name].astype(str).str.lower().eq("true")

    valid_tracking = (
        visible_column("left_jaw_visible")
        & visible_column("right_jaw_visible")
        & (df["mask_area_px"] > 0)
    )
    near_object = distance < 45
    approaching = near_object & (distance_change < -3) & valid_tracking
    closing = near_object & (jaw_change < -4) & valid_tracking
    moving_together = (
        (g_move > 4) & (o_move > 4) & (similarity > 0.85)
        & (relative_error < 12) & valid_tracking
    )
    possible_release = (
        (jaw_change > 3) & (g_move > 8) & (o_move < 3)
        & (distance_change > 8) & valid_tracking
    )

    state = "SEARCH"
    grasp_frames = 0
    release_frames = 0
    phase_labels = []

    for i in range(len(df)):
        if not valid_tracking.iloc[i]:
            phase_labels.append("UNCERTAIN")
            continue
        if state == "SEARCH":
            if approaching.iloc[i]:
                state = "APPROACH"
        elif state == "APPROACH":
            if closing.iloc[i]:
                grasp_frames += 1
            else:
                grasp_frames = 0
            if grasp_frames >= 3:
                state = "POSSIBLE_GRASP"
                grasp_frames = 0
        elif state == "POSSIBLE_GRASP":
            grasp_frames += 1
            if grasp_frames >= 5:
                state = "HOLDING"
        elif state in ("HOLDING", "TRANSPORT"):
            if possible_release.iloc[i]:
                release_frames += 1
            else:
                release_frames = 0
            if release_frames >= 3:
                state = "POSSIBLE_RELEASE"
            elif moving_together.iloc[i]:
                state = "TRANSPORT"
            else:
                state = "HOLDING"
        phase_labels.append(state)

    df["fsm_phase"] = phase_labels
    df["jaw_distance_px"] = jaw_width
    df["object_jaw_distance_px"] = distance
    df["motion_similarity_fsm"] = similarity

    # Original cell 62: transport evidence and hysteresis.
    # Deliberately recompute as the original cell did.
    jaw_x = (df["left_jaw_x_px_s"] + df["right_jaw_x_px_s"]) / 2
    jaw_y = (df["left_jaw_y_px_s"] + df["right_jaw_y_px_s"]) / 2
    object_x = df["center_x_px_s"]
    object_y = df["center_y_px_s"]
    gx = jaw_x.diff(5)
    gy = jaw_y.diff(5)
    ox = object_x.diff(5)
    oy = object_y.diff(5)
    gripper_move = np.hypot(gx, gy)
    object_move = np.hypot(ox, oy)
    similarity = (gx * ox + gy * oy) / (gripper_move * object_move).replace(0, np.nan)
    relative_error = np.hypot(gx - ox, gy - oy)
    transport_evidence = (
        (gripper_move > 4) & (object_move > 4) & (similarity > 0.85)
        & (relative_error < 12)
    )
    min_transport_frames = 5
    min_holding_frames = 10
    grasp_detected = df["fsm_phase"].isin(["POSSIBLE_GRASP", "HOLDING", "TRANSPORT"])

    transport_state = False
    positive_count = 0
    negative_count = 0
    new_phases = []

    for i in range(len(df)):
        original_phase = df.loc[i, "fsm_phase"]
        if original_phase in ["SEARCH", "APPROACH", "POSSIBLE_GRASP", "UNCERTAIN"]:
            new_phases.append(original_phase)
            continue
        if original_phase == "POSSIBLE_RELEASE":
            new_phases.append(original_phase)
            continue
        if not grasp_detected.iloc[i]:
            new_phases.append("SEARCH")
            continue
        if transport_evidence.iloc[i]:
            positive_count += 1
            negative_count = 0
        else:
            negative_count += 1
            positive_count = 0
        if not transport_state and positive_count >= min_transport_frames:
            transport_state = True
        elif transport_state and negative_count >= min_holding_frames:
            transport_state = False
        new_phases.append("TRANSPORT" if transport_state else "HOLDING")

    df["fsm_phase_improved"] = new_phases

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "fsm_base_results.csv", index=False)

    return df
