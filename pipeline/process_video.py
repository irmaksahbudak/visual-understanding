"""Process one robot demonstration with manually supplied tracking seeds."""

from pathlib import Path

import cv2

from .segmentation.sam2_tracker import segment_and_track_objects
from .tracking.cotracker import track_gripper_long_video
from .interaction.interaction_pipeline import run_interaction_pipeline
from .interaction.visualization import visualize_results


def extract_frames(video_path, frames_dir, fps):
    """Sample the source video using the original experiment's time grid."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    count = 0
    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_fps <= 0 or source_count <= 0:
            raise RuntimeError("Cannot read source video FPS or frame count.")

        duration = source_count / source_fps
        while count / fps < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * count / fps)
            ok, frame = cap.read()
            if not ok:
                break

            path = frames_dir / f"{count:05d}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Cannot save frame: {path}")
            count += 1
    finally:
        cap.release()

    if count == 0:
        raise RuntimeError("No frames were extracted.")

    if count / fps + 1 / fps < duration:
        print(f"Note: sampled {count / fps:.2f} s of {duration:.2f} s.")

    return count


def process_video(
    video_path,
    *,
    output_dir,
    checkpoint_path,
    initial_mask_path,
    object_reference_frame,
    gripper_start_frame,
    initial_points,
):
    """Run tracking, phase detection and visualization at 10 FPS."""
    fps = 10.0
    output_dir = Path(output_dir)
    checkpoint_path = Path(checkpoint_path)
    initial_mask_path = Path(initial_mask_path)

    for path in (Path(video_path), checkpoint_path, initial_mask_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    # A new run folder keeps previous results untouched.
    output_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir()

    frame_count = extract_frames(video_path, frames_dir, fps)
    for name, index in (
        ("object_reference_frame", object_reference_frame),
        ("gripper_start_frame", gripper_start_frame),
    ):
        if not 0 <= index < frame_count:
            raise ValueError(f"{name} must be between 0 and {frame_count - 1}.")

    print(f"[1/4] SAM 2 object tracking · {frame_count} frames")
    object_dir = output_dir / "object"
    segment_and_track_objects(
        frames_dir=frames_dir,
        initial_mask_path=initial_mask_path,
        reference_frame=object_reference_frame,
        checkpoint_path=checkpoint_path,
        model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
        output_dir=object_dir,
        fps=fps,
    )

    print("[2/4] CoTracker3 gripper tracking")
    gripper_dir = output_dir / "gripper"
    track_gripper_long_video(
        frames_dir=frames_dir,
        initial_points=initial_points,
        start_frame=gripper_start_frame,
        output_dir=gripper_dir,
        fps=fps,
        chunk_size=80,
    )

    print("[3/4] Interaction features and FSM v2")
    analysis_dir = output_dir / "analysis"
    results = run_interaction_pipeline(
        object_csv=object_dir / "object_tracking.csv",
        gripper_csv=gripper_dir / "gripper_tracking_long.csv",
        output_dir=analysis_dir,
    )

    print("[4/4] Result video and timeline")
    video, graph = visualize_results(
        results,
        frames_dir=frames_dir,
        masks_dir=object_dir / "masks",
        output_dir=output_dir,
        fps=fps,
    )
    return results, video, graph
