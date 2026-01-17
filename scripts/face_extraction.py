import os
import json
import cv2
import numpy as np
from pathlib import Path
from retinaface import RetinaFace


def iou(boxA, boxB):
    """Compute Intersection over Union between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def extract_face_tracks(
    frames_dir: str,
    output_dir: str,
    iou_threshold: float = 0.5
):
    """
    Detect faces using RetinaFace and track them across frames.

    Args:
        frames_dir (str): Directory containing extracted frames
        output_dir (str): Directory to save face tracks
        iou_threshold (float): IoU threshold for tracking

    Output:
        output_dir/
            face_tracks/
                track_000/
                track_001/
            tracks.json
    """

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    tracks_dir = output_dir / "face_tracks"

    tracks_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_dir.glob("*.jpg"))
    active_tracks = {}
    finished_tracks = []
    track_id_counter = 0

    for frame_idx, frame_path in enumerate(frame_files):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        detections = RetinaFace.detect_faces(str(frame_path))
        current_boxes = []

        if isinstance(detections, dict):
            for face in detections.values():
                box = face["facial_area"]
                current_boxes.append(box)

        assigned_tracks = set()

        for box in current_boxes:
            best_iou = 0
            best_track_id = None

            for tid, track in active_tracks.items():
                prev_box = track["last_box"]
                score = iou(box, prev_box)

                if score > best_iou:
                    best_iou = score
                    best_track_id = tid

            if best_iou >= iou_threshold and best_track_id is not None:
                # Assign to existing track
                track = active_tracks[best_track_id]
                track["boxes"].append(box)
                track["frames"].append(frame_path.name)
                track["last_box"] = box
                assigned_tracks.add(best_track_id)

            else:
                # Create new track
                track_id = track_id_counter
                track_id_counter += 1

                active_tracks[track_id] = {
                    "track_id": track_id,
                    "boxes": [box],
                    "frames": [frame_path.name],
                    "last_box": box
                }
                assigned_tracks.add(track_id)

        # Move inactive tracks to finished
        inactive = set(active_tracks.keys()) - assigned_tracks
        for tid in inactive:
            finished_tracks.append(active_tracks.pop(tid))

    finished_tracks.extend(active_tracks.values())

    # Save cropped face tracks
    tracks_metadata = []

    for track in finished_tracks:
        tid = track["track_id"]
        track_path = tracks_dir / f"track_{tid:03d}"
        track_path.mkdir(parents=True, exist_ok=True)

        for frame_name, box in zip(track["frames"], track["boxes"]):
            frame_img = cv2.imread(str(frames_dir / frame_name))
            x1, y1, x2, y2 = box
            face_crop = frame_img[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            cv2.imwrite(str(track_path / frame_name), face_crop)

        tracks_metadata.append({
            "track_id": tid,
            "num_frames": len(track["frames"]),
            "frames": track["frames"]
        })

    with open(output_dir / "tracks.json", "w") as f:
        json.dump(tracks_metadata, f, indent=2)

    return tracks_metadata
