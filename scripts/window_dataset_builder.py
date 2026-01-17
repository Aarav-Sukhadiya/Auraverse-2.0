import json
from pathlib import Path
from typing import List


def build_temporal_windows(
    video_dir: str,
    label: str,
    window_size: int = 16,
    stride: int = 8,
    target_fps: int = 10
):
    """
    Build temporal windows from face tracks.

    Args:
        video_dir (str): Directory containing face_tracks and metadata.json
        label (str): "real" or "fake"
        window_size (int): Number of frames per window
        stride (int): Step size between windows
        target_fps (int): FPS used during frame extraction

    Output:
        video_dir/windows.json
    """

    video_dir = Path(video_dir)
    tracks_dir = video_dir / "face_tracks"
    metadata_path = video_dir / "metadata.json"

    assert label in {"real", "fake"}, "Label must be 'real' or 'fake'"

    if not tracks_dir.exists():
        raise FileNotFoundError("face_tracks directory not found")

    if not metadata_path.exists():
        raise FileNotFoundError("metadata.json not found")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    frame_time_map = {
        entry["frame"]: entry["time_sec"]
        for entry in metadata["frame_timestamps"]
    }

    windows = []
    window_id = 0

    for track_path in sorted(tracks_dir.glob("track_*")):
        track_id = int(track_path.name.split("_")[-1])
        frame_files = sorted(track_path.glob("*.jpg"))

        if len(frame_files) < window_size:
            continue

        frame_names = [f.name for f in frame_files]

        for start in range(0, len(frame_names) - window_size + 1, stride):
            window_frames = frame_names[start:start + window_size]

            start_time = frame_time_map.get(window_frames[0], None)
            end_time = frame_time_map.get(window_frames[-1], None)

            if start_time is None or end_time is None:
                continue

            windows.append({
                "video": video_dir.name,
                "track_id": track_id,
                "window_id": window_id,
                "frames": window_frames,
                "label": label,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3)
            })

            window_id += 1

    output_path = video_dir / "windows.json"
    with open(output_path, "w") as f:
        json.dump(windows, f, indent=2)

    return {
        "video": video_dir.name,
        "num_windows": len(windows),
        "window_size": window_size,
        "stride": stride
    }
