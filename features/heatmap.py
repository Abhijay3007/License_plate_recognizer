from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


def create_confidence_heatmap(
    image: np.ndarray,
    detections: Iterable[Dict],
    output_path: Path | str | None = None,
) -> Tuple[np.ndarray, str | None]:
    """
    Create a soft heatmap centered on YOLO detections. This is not Grad-CAM,
    but it gives a clear confidence-focused visualization using only OpenCV.
    """
    image_height, image_width = image.shape[:2]
    heat = np.zeros((image_height, image_width), dtype=np.float32)

    detections = list(detections)
    if not detections:
        return image.copy(), None

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        confidence = max(float(detection["confidence"]), 0.05)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        sigma_x = max((x2 - x1) / 2.5, 8)
        sigma_y = max((y2 - y1) / 2.5, 8)

        xx, yy = np.meshgrid(np.arange(image_width), np.arange(image_height))
        gaussian = np.exp(
            -(
                ((xx - center_x) ** 2) / (2 * sigma_x**2)
                + ((yy - center_y) ** 2) / (2 * sigma_y**2)
            )
        )
        heat += gaussian * confidence

    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image, 0.62, heat_colored, 0.38, 0)

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            blended,
            f"{detection['confidence']:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    saved_path = None
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), blended)
        saved_path = str(path)

    return blended, saved_path
