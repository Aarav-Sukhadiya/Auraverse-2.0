import os
import cv2
import json
from pathlib import Path


def extract_video_frames(
    video_path: str,
    output_dir: str,
    target_fps: int = 10  # DEFAULT
):
    """
    Extract uniformly sampled frames from a video and store timestamps.

    Args:
        video_path (str): Path to input video
        output_dir (str): Directory to store frames and metadata
        target_fps (int): Sampling FPS (default: 10)

    Output Structure:
        output_dir/
            frames/
                frame_000000.jpg
                frame_000001.jpg
            metadata.json
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_native_frames / native_fps if native_fps > 0 else 0.0

    # Sampling step
    step = max(int(round(native_fps / target_fps)), 1)

    frame_idx = 0
    saved_idx = 0
    timestamps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            frame_name = f"frame_{saved_idx:06d}.jpg"
            frame_path = frames_dir / frame_name

            cv2.imwrite(str(frame_path), frame)

            time_sec = round(saved_idx / target_fps, 4)
            timestamps.append({
                "frame": frame_name,
                "time_sec": time_sec
            })

            saved_idx += 1

        frame_idx += 1

    cap.release()

    metadata = {
        "video_name": video_path.name,
        "native_fps": round(native_fps, 3),
        "target_fps": target_fps,
        "total_native_frames": total_native_frames,
        "extracted_frames": saved_idx,
        "duration_sec": round(duration_sec, 3),
        "frame_timestamps": timestamps
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
