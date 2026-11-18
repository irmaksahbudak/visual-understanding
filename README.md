# Visual Understanding

This project analyzes a single RGB video of a robot performing a manipulation task. It extracts 2D object and gripper trajectories and estimates the task phase for each analyzed frame.

- **SAM 2** segments the selected object and tracks its mask across video frames.
- **CoTracker3** tracks selected points on the robot gripper.
- A **rule-based finite-state machine (FSM)** uses motion measurements to estimate task phases such as approach, possible grasp, holding, transport, and possible release.
- The main pipeline in `pipeline/process_video.py` generates CSV results, an annotated video, and a task-phase timeline.

## Running the Pipeline

The main entry point is `pipeline.process_video.process_video`. It takes the input video, the SAM 2.1 Tiny checkpoint, an initial object mask and its reference frame, and the starting coordinates and frame for the gripper points.

In the current version, the initial mask and gripper points are selected manually. The prototype was tested in Google Colab with a GPU. The original video and model weights are not included in this repository.

## Validation

On the development video, the modular FSM reproduced the previous phase predictions for **277 out of 277 frames**. This checks that the refactored pipeline preserves the previous results. It does not measure task recognition accuracy on unseen videos.

## Future Work

A planned extension is to use ArUco markers and a calibrated camera to estimate 3D marker poses and combine them with the existing 2D tracking results. This feature is not included in the current version.
