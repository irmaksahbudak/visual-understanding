"""Generate task phases directly from object and gripper tracking."""

from pathlib import Path

import pandas as pd

from .interaction_features import extract_interaction_features
from .fsm_base import analyze_base_interactions
from .fsm_v2 import analyze_interactions


def run_interaction_pipeline(object_csv, gripper_csv, output_dir):
    """Run the validated interaction rules for tracking sampled at 10 FPS."""
    output_dir = Path(output_dir)

    features = extract_interaction_features(
        object_tracking=pd.read_csv(object_csv),
        gripper_tracking=pd.read_csv(gripper_csv),
        output_dir=output_dir,
        window=5,
    )

    base_phases = analyze_base_interactions(features, output_dir=output_dir)

    return analyze_interactions(
        fsm_v1_data=base_phases,
        output_dir=output_dir,
        fps=10.0,
        motion_threshold=10.0,
        area_threshold=0.35,
        confirmation_frames=2,
    )
