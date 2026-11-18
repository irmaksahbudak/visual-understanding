"""Render the tracked demonstration and the automatic phase timeline."""

from collections import deque
from pathlib import Path
import subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_results(results, frames_dir, masks_dir, output_dir, fps=10.0):
    frames_dir, masks_dir, output_dir = map(Path, (frames_dir, masks_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not frames:
        raise FileNotFoundError(f"No JPG frames in {frames_dir}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Cannot read {frames[0]}")
    h, w = first.shape[:2]

    results = results.sort_values("frame").set_index("frame", verify_integrity=True)
    raw = output_dir / "preview_raw.mp4"
    video = output_dir / "task_preview.mp4"
    graph = output_dir / "task_timeline.png"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not open the video writer.")

    # OpenCV colors use BGR order.
    colors = {
        "gripper_body": (0, 255, 255),
        "left_jaw": (0, 255, 0),
        "right_jaw": (255, 255, 0),
    }
    trails = {name: deque(maxlen=15) for name in colors}
    phases = [
        "NOT ANALYZED", "SEARCH", "APPROACH", "POSSIBLE_GRASP",
        "HOLDING", "TRANSPORT", "POSSIBLE_RELEASE", "UNCERTAIN",
    ]
    phase_index = {name: i for i, name in enumerate(phases)}
    times, labels = [], []

    try:
        for path in frames:
            idx = int(path.stem)
            frame = cv2.imread(str(path))
            if frame is None or frame.shape[:2] != (h, w):
                raise RuntimeError(f"Invalid frame: {path}")

            mask_path = masks_dir / f"{idx:05d}.png"
            if mask_path.is_file():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.shape == (h, w):
                    selected = mask > 127
                    frame[selected] = (
                        0.65 * frame[selected] + 0.35 * np.array([255, 0, 0])
                    ).astype(np.uint8)

            row = results.loc[idx] if idx in results.index else None
            phase = "NOT ANALYZED" if row is None else str(row["fsm_phase_v2"])
            time_s = idx / fps if row is None else float(row["time_s"])

            for name, color in colors.items():
                point = None
                if row is not None and str(row[f"{name}_visible"]).lower() == "true":
                    x, y = row[f"{name}_x_px"], row[f"{name}_y_px"]
                    if np.isfinite([x, y]).all():
                        point = (round(x), round(y))
                    if point is not None and not (0 <= point[0] < w and 0 <= point[1] < h):
                        point = None

                trail = trails[name]
                trail.append(point)
                if point is not None:
                    for a, b in zip(trail, list(trail)[1:]):
                        if a is not None and b is not None:
                            cv2.line(frame, a, b, color, 2)
                    cv2.circle(frame, point, 5, color, -1)

            cv2.rectangle(frame, (0, 0), (w - 1, 52), (25, 25, 25), -1)
            cv2.putText(frame, f"{time_s:.1f} s | FSM: {phase}", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            times.append(time_s)
            labels.append(phase_index.get(phase, phase_index["UNCERTAIN"]))
    finally:
        writer.release()

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(video),
    ], check=True)
    raw.unlink()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(times, labels, where="post", linewidth=2)
    ax.set_yticks(range(len(phases)), [p.replace("_", " ") for p in phases])
    ax.set(xlabel="Original video time (s)", title="Automatic FSM v2 task phases")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(graph, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return video, graph
