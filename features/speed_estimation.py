from typing import Dict, Optional

import cv2
import numpy as np


def _center_from_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def estimate_vehicle_speed(
    first_detection: Optional[Dict],
    second_detection: Optional[Dict],
    time_interval_seconds: float,
    meters_per_pixel: float,
) -> Dict:
    if first_detection is None or second_detection is None:
        return {
            "success": False,
            "reason": "Speed estimation needs one detected plate in each frame.",
        }

    if time_interval_seconds <= 0:
        return {
            "success": False,
            "reason": "Time interval must be greater than zero seconds.",
        }

    if meters_per_pixel <= 0:
        return {
            "success": False,
            "reason": "Calibration factor must be greater than zero meters per pixel.",
        }

    center_1 = _center_from_bbox(first_detection["bbox"])
    center_2 = _center_from_bbox(second_detection["bbox"])
    pixel_shift = float(np.linalg.norm(np.array(center_2) - np.array(center_1)))
    distance_meters = pixel_shift * meters_per_pixel
    speed_mps = distance_meters / time_interval_seconds
    speed_kmph = speed_mps * 3.6

    return {
        "success": True,
        "pixel_shift": round(pixel_shift, 2),
        "distance_meters": round(distance_meters, 3),
        "speed_mps": round(speed_mps, 3),
        "speed_kmph": round(speed_kmph, 2),
        "center_start": center_1,
        "center_end": center_2,
        "reason": "Estimated from plate center displacement between two frames.",
    }


def draw_speed_visualization(image, first_detection: Dict, second_detection: Dict, speed_result: Dict):
    canvas = image.copy()
    start = tuple(int(value) for value in _center_from_bbox(first_detection["bbox"]))
    end = tuple(int(value) for value in _center_from_bbox(second_detection["bbox"]))

    cv2.arrowedLine(canvas, start, end, (0, 255, 255), 3, tipLength=0.12)
    cv2.putText(
        canvas,
        f"Approx speed: {speed_result['speed_kmph']} km/h",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas
