"""Calculate object-gripper interaction features."""

from pathlib import Path

import numpy as np
import pandas as pd


def extract_interaction_features(
    object_tracking,
    gripper_tracking,
    output_dir,
    window=5,
):
    """
    Merge SAM 2 and CoTracker3 results and calculate
    interaction features.

    All coordinates are measured in image pixels.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_object = [
        "frame", "center_x_px", "center_y_px",
        "mask_area_px"
    ]

    required_gripper = [
        "frame", "time_s",
        "gripper_body_x_px", "gripper_body_y_px",
        "left_jaw_x_px", "left_jaw_y_px",
        "right_jaw_x_px", "right_jaw_y_px"
    ]

    for name in required_object:
        if name not in object_tracking:
            raise ValueError(f"Missing object column: {name}")

    for name in required_gripper:
        if name not in gripper_tracking:
            raise ValueError(f"Missing gripper column: {name}")

    if object_tracking["frame"].duplicated().any():
        raise ValueError("Duplicate object frame indices.")

    if gripper_tracking["frame"].duplicated().any():
        raise ValueError("Duplicate gripper frame indices.")

    if window < 1:
        raise ValueError("window must be at least 1.")

    # 1. Merge by the original frame index.
    objects = object_tracking.drop(
        columns=["time_s"], errors="ignore"
    )

    df = pd.merge(
        gripper_tracking,
        objects,
        on="frame",
        how="inner",
        validate="one_to_one"
    ).sort_values("frame").reset_index(drop=True)

    if df.empty:
        raise ValueError("No matching frames were found.")

    # 2. Object displacement over a fixed time window.
    object_dx = df["center_x_px"].diff(window)
    object_dy = df["center_y_px"].diff(window)

    # 3. Gripper-body displacement over the same window.
    gripper_dx = df["gripper_body_x_px"].diff(window)
    gripper_dy = df["gripper_body_y_px"].diff(window)

    object_motion = np.hypot(object_dx, object_dy)
    gripper_motion = np.hypot(gripper_dx, gripper_dy)

    # 4. Cosine similarity of displacement directions.
    dot_product = (
        object_dx * gripper_dx
        + object_dy * gripper_dy
    )

    denominator = object_motion * gripper_motion

    similarity = (
        dot_product / denominator.replace(0, np.nan)
    ).clip(-1, 1)

    # 5. Difference between object and gripper displacements.
    relative_error = np.hypot(
        object_dx - gripper_dx,
        object_dy - gripper_dy
    )

    # Only accept intervals with valid tracking and
    # meaningful motion in both the object and gripper.
    visible = (
        df["gripper_body_visible"].astype(bool)
        & df["gripper_body_visible"]
          .shift(window).fillna(False).astype(bool)
    )

    valid = (
        visible
        & (df["mask_area_px"] > 0)
        & (df["mask_area_px"].shift(window) > 0)
        & (gripper_motion > 3)
        & (object_motion > 3)
    )

    df["motion_similarity"] = similarity.where(valid)
    df["relative_motion_error_px"] = relative_error.where(valid)

    # 6. Distance between tracked jaw landmarks.
    df["jaw_point_distance_px"] = np.hypot(
        df["left_jaw_x_px"] - df["right_jaw_x_px"],
        df["left_jaw_y_px"] - df["right_jaw_y_px"]
    )

    # 7. Save results.
    output_path = output_dir / "interaction_features.csv"
    df.to_csv(output_path, index=False)

    return df
