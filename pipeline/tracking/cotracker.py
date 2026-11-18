"""Track robot keypoints with pretrained CoTracker3."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


def track_gripper_points(
    frames_dir,
    initial_points,
    start_frame,
    output_dir,
    fps=10.0,
    max_frames=None,
    model=None,
):
    """
    Track selected robot points in RGB video frames.

    initial_points:
        Dictionary containing point names and (x, y) coordinates.

    start_frame:
        Original video frame index used for initialization.
    """

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this test.")

    # Use the original frame numbers from the filenames.
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    frame_paths = [
        p for p in frame_paths
        if int(p.stem) >= start_frame
    ]

    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    if not frame_paths:
        raise ValueError("No frames available for tracking.")

    if int(frame_paths[0].stem) != start_frame:
        raise ValueError("Starting frame is missing.")

    # Read video frames in RGB format.
    frames = []

    for path in frame_paths:
        bgr = cv2.imread(str(path))

        if bgr is None:
            raise ValueError(f"Cannot read frame: {path}")

        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    video_array = np.stack(frames)

    # CoTracker expects [batch, time, channels, height, width].
    video = (
        torch.from_numpy(video_array)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        .float()
        .to("cuda")
    )

    # Prepare custom tracking queries [time, x, y].
    names = list(initial_points)

    queries = torch.tensor(
        [
            [0.0, float(initial_points[n][0]),
             float(initial_points[n][1])]
            for n in names
        ],
        dtype=torch.float32,
        device="cuda"
    ).unsqueeze(0)

    # Load pretrained CoTracker3 if not provided.
    if model is None:
        model = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_offline"
        ).to("cuda").eval()

    with torch.inference_mode():
        tracks, visibility = model(video, queries=queries)

    tracks = tracks[0].cpu().numpy()
    visibility = visibility[0].cpu().numpy()

    # Convert tracking results to a reusable CSV table.
    records = []

    for t, path in enumerate(frame_paths):

        frame_idx = int(path.stem)

        row = {
            "frame": frame_idx,
            "time_s": frame_idx / fps
        }

        for j, name in enumerate(names):
            row[f"{name}_x_px"] = float(tracks[t, j, 0])
            row[f"{name}_y_px"] = float(tracks[t, j, 1])
            row[f"{name}_visible"] = bool(
                np.asarray(visibility[t, j]).item() > 0.5
            )

        records.append(row)

    df = pd.DataFrame(records)

    output_csv = output_dir / "gripper_tracking.csv"
    df.to_csv(output_csv, index=False)

    return df


def track_gripper_long_video(
    frames_dir,
    initial_points,
    start_frame,
    output_dir,
    fps=10.0,
    chunk_size=80,
):
    """
    Track gripper landmarks over a long video using chunks.

    Each chunk overlaps the previous chunk by one frame.
    The final point positions initialize the next chunk.
    """

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(
        p for p in frames_dir.glob("*.jpg")
        if int(p.stem) >= start_frame
    )

    if not frame_paths:
        raise ValueError("No frames found.")

    if chunk_size < 2:
        raise ValueError("chunk_size must be at least 2.")

    # Load the model once and reuse it for every chunk.
    model = torch.hub.load(
        "facebookresearch/co-tracker",
        "cotracker3_offline"
    ).to("cuda").eval()

    current_points = dict(initial_points)
    all_results = []

    start_index = 0

    while start_index < len(frame_paths):

        # Include one overlapping frame between chunks.
        end_index = min(
            start_index + chunk_size,
            len(frame_paths)
        )

        chunk = frame_paths[start_index:end_index]

        # Create a temporary directory with consecutive frames.
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            for path in chunk:
                shutil.copy2(
                    path,
                    temp_dir / path.name
                )

            chunk_result = track_gripper_points(
                frames_dir=temp_dir,
                initial_points=current_points,
                start_frame=int(chunk[0].stem),
                output_dir=output_dir / "temporary",
                fps=fps,
                model=model
            )

        # Avoid saving the overlapping frame twice.
        if all_results:
            chunk_result = chunk_result.iloc[1:].copy()

        if chunk_result.empty:
            break

        all_results.append(chunk_result)

        # Initialize the next chunk using last tracked positions.
        last = chunk_result.iloc[-1]

        current_points = {
            name: (
                float(last[f"{name}_x_px"]),
                float(last[f"{name}_y_px"])
            )
            for name in current_points
        }

        if end_index == len(frame_paths):
            break

        start_index = end_index - 1

    final = pd.concat(
        all_results,
        ignore_index=True
    )

    final.to_csv(
        output_dir / "gripper_tracking_long.csv",
        index=False
    )

    return final
