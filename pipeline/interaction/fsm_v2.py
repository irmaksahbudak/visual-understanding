"""FSM_v2: Rule-based robot interaction and release detection."""

from pathlib import Path

import numpy as np
import pandas as pd


def analyze_interactions(
    fsm_v1_data,
    output_dir,
    fps=10.0,
    motion_threshold=10.0,
    area_threshold=0.35,
    confirmation_frames=2,
):
    """
    Add automatic release detection to FSM_v1 predictions.

    Parameters
    ----------
    fsm_v1_data : pandas.DataFrame
        Tracking features and FSM_v1 phase predictions.

    output_dir : str or Path
        Directory for saving FSM_v2 results.

    Returns
    -------
    pandas.DataFrame
        Frame-level tracking features and FSM_v2 predictions.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = (
        fsm_v1_data.copy()
        .sort_values("frame")
        .reset_index(drop=True)
    )

    required = [
        "frame",
        "center_x_px_s",
        "center_y_px_s",
        "left_jaw_x_px_s",
        "left_jaw_y_px_s",
        "right_jaw_x_px_s",
        "right_jaw_y_px_s",
        "mask_area_px",
        "jaw_distance_px",
        "fsm_phase_improved",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    if df["frame"].duplicated().any():
        raise ValueError("Duplicate frame indices found.")

    if confirmation_frames < 1:
        raise ValueError("confirmation_frames must be >= 1")

    # --------------------------------------------------
    # 1. Object and gripper relative positions
    # --------------------------------------------------

    gripper_x = (
        df["left_jaw_x_px_s"] +
        df["right_jaw_x_px_s"]
    ) / 2

    gripper_y = (
        df["left_jaw_y_px_s"] +
        df["right_jaw_y_px_s"]
    ) / 2

    relative_x = df["center_x_px_s"] - gripper_x
    relative_y = df["center_y_px_s"] - gripper_y

    # --------------------------------------------------
    # 2. Relative motion
    # --------------------------------------------------

    window = 2

    relative_motion = np.hypot(
        relative_x.diff(window),
        relative_y.diff(window)
    )

    # --------------------------------------------------
    # 3. SAM 2 mask-area change
    # --------------------------------------------------

    area = df["mask_area_px"].astype(float)

    previous_area = (
        area.shift(1)
        .rolling(5, min_periods=3)
        .max()
    )

    area_drop = (
        (previous_area - area) /
        previous_area.replace(0, np.nan)
    ).clip(lower=0, upper=1)

    # --------------------------------------------------
    # 4. Jaw-point distance change
    # --------------------------------------------------

    jaw_change = df["jaw_distance_px"].diff(window)

    # --------------------------------------------------
    # 5. Automatic release candidates
    # --------------------------------------------------

    post_grasp = df["fsm_phase_improved"].isin([
        "HOLDING",
        "TRANSPORT",
        "POSSIBLE_RELEASE"
    ])

    valid = (
        post_grasp
        & np.isfinite(relative_motion)
        & np.isfinite(area_drop)
        & (area > 50)
    )

    release_candidate = (
        valid
        & (relative_motion > motion_threshold)
        & (area_drop > area_threshold)
    )

    df["relative_displacement_px"] = relative_motion
    df["mask_area_drop"] = area_drop
    df["jaw_change_px"] = jaw_change
    df["release_candidate_v2"] = release_candidate

    # --------------------------------------------------
    # 6. FSM_v2 state transition
    # --------------------------------------------------

    df["fsm_phase_v2"] = df["fsm_phase_improved"].copy()

    consecutive = 0
    release_confirmed = False

    for i in range(len(df)):

        if release_confirmed:
            df.loc[i, "fsm_phase_v2"] = "POSSIBLE_RELEASE"
            continue

        if bool(df.loc[i, "release_candidate_v2"]):
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= confirmation_frames:
            release_confirmed = True
            df.loc[i, "fsm_phase_v2"] = "POSSIBLE_RELEASE"

    # --------------------------------------------------
    # 7. Save results
    # --------------------------------------------------

    output_path = output_dir / "fsm_v2_results.csv"

    df.to_csv(output_path, index=False)

    return df
