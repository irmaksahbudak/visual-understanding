"""SAM 2 video segmentation and object tracking."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from sam2.build_sam import build_sam2_video_predictor


def segment_and_track_objects(
    frames_dir,
    initial_mask_path,
    reference_frame,
    checkpoint_path,
    model_cfg,
    output_dir,
    fps=10.0,
    object_id=1,
):
    """Track an object using a manually or automatically created mask."""

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"

    masks_dir.mkdir(parents=True, exist_ok=True)

    # Validate input frames.
    frame_files = sorted(frames_dir.glob("*.jpg"))

    if not frame_files:
        raise ValueError("No JPEG frames found.")

    if not 0 <= reference_frame < len(frame_files):
        raise ValueError("Invalid reference frame.")

    # Load the binary initialization mask.
    mask = cv2.imread(
        str(initial_mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise FileNotFoundError(initial_mask_path)

    first_frame = cv2.imread(str(frame_files[0]))

    if mask.shape != first_frame.shape[:2]:
        raise ValueError("Mask and video frame dimensions differ.")

    # Load pretrained SAM 2.
    predictor = build_sam2_video_predictor(
        model_cfg,
        str(checkpoint_path),
        device="cuda"
    )

    records = {}

    with torch.inference_mode():

        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True
        )

        # Register the object at the reference frame.
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=reference_frame,
            obj_id=object_id,
            mask=mask > 127
        )

        def propagate(reverse):
            results = predictor.propagate_in_video(
                state,
                start_frame_idx=reference_frame,
                reverse=reverse
            )

            for frame_idx, obj_ids, logits in results:

                obj_index = list(obj_ids).index(object_id)

                binary_mask = (
                    logits[obj_index, 0] > 0
                ).cpu().numpy().astype(np.uint8)

                # Save the segmentation mask.
                cv2.imwrite(
                    str(masks_dir / f"{frame_idx:05d}.png"),
                    binary_mask * 255
                )

                # Calculate image-plane centroid.
                ys, xs = np.where(binary_mask > 0)

                records[frame_idx] = {
                    "frame": frame_idx,
                    "time_s": frame_idx / fps,
                    "object_id": object_id,
                    "center_x_px": (
                        float(xs.mean()) if len(xs) else np.nan
                    ),
                    "center_y_px": (
                        float(ys.mean()) if len(ys) else np.nan
                    ),
                    "mask_area_px": int(len(xs))
                }

        # Track forward and backward from reference frame.
        propagate(reverse=False)

        if reference_frame > 0:
            propagate(reverse=True)

    # Save tracking measurements.
    tracking_df = pd.DataFrame(
        [records[i] for i in sorted(records)]
    )

    csv_path = output_dir / "object_tracking.csv"
    tracking_df.to_csv(csv_path, index=False)

    predictor.reset_state(state)
    del predictor
    torch.cuda.empty_cache()

    return tracking_df
