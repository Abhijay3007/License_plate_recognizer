from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ai.ai_model import detect_plate_regions, load_yolov5_model
from ai.ocr_model import easyocr_model_load
from features.heatmap import create_confidence_heatmap
from features.plate_lookup import lookup_plate_record
from features.plate_validation import normalize_plate_text, validate_plate_format
from helper.general_utils import filter_text
from helper.params import Parameters


params = Parameters()


@lru_cache(maxsize=1)
def get_detection_model():
    return load_yolov5_model()


@lru_cache(maxsize=1)
def get_ocr_reader():
    return easyocr_model_load()


def preprocess_for_ocr(image: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    _, otsu = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return [
        image,
        cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR),
    ]


def resize_for_ocr(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if width == 0:
        return image

    target_width = max(params.ocr_input_width, 640)
    if width < target_width:
        resized_height = max(1, int(height * target_width / width))
        return cv2.resize(image, (target_width, resized_height), interpolation=cv2.INTER_CUBIC)
    return image


def generate_plate_views(image: np.ndarray) -> List[np.ndarray]:
    if image is None or image.size == 0:
        return []

    image = resize_for_ocr(image)
    height, width = image.shape[:2]
    views = [image]

    if width > 120:
        left_trim = int(width * 0.14)
        right_trim = int(width * 0.96)
        views.append(image[:, left_trim:right_trim].copy())

    if height > 60:
        mid = height // 2
        views.append(image[:mid, :].copy())
        views.append(image[mid:, :].copy())

    return [view for view in views if view.size]


def extract_plate_candidates(image: Optional[np.ndarray]) -> List[str]:
    if image is None or image.size == 0:
        return []

    reader = get_ocr_reader()
    candidates = []

    for view in generate_plate_views(image):
        for variant in preprocess_for_ocr(view):
            detailed_results = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
            texts = filter_text(
                max(variant.shape[0] * variant.shape[1], 1),
                detailed_results,
                params.region_threshold,
            )
            paragraph_results = reader.readtext(
                variant,
                detail=0,
                paragraph=True,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )

            for text in [*texts, *paragraph_results]:
                normalized = normalize_plate_text(text)
                if len(normalized) >= 6:
                    candidates.append(normalized)

    unique_candidates = list(dict.fromkeys(candidates))
    unique_candidates.sort(key=lambda item: (-candidates.count(item), -len(item), item))
    return unique_candidates


def analyze_image(
    frame: np.ndarray,
    image_name: str = "uploaded_image",
    watchlist_path: Path | str = Path("features") / "data" / "vehicle_watchlist.json",
    heatmap_dir: Path | str = Path("exports") / "heatmaps",
) -> Dict:
    model, labels = get_detection_model()
    annotated_frame, detections = detect_plate_regions(frame, model, labels)
    top_detection = detections[0] if detections else None
    plate_crop = top_detection["crop"] if top_detection else None

    candidates = extract_plate_candidates(plate_crop)
    if not candidates:
        # Fallback to the whole image when the crop OCR is weak.
        candidates = extract_plate_candidates(frame)

    if candidates:
        best_candidate = candidates[0]
        best_score = -1
        status_scores = {"VALID": 2, "SUSPICIOUS": 1, "INVALID": 0}
        for candidate in candidates:
            score = status_scores.get(validate_plate_format(candidate)["status"], 0)
            if score > best_score:
                best_score = score
                best_candidate = candidate
            if best_score == 2:
                break
        plate_text = best_candidate
    else:
        plate_text = "NO_PLATE_FOUND"

    validation = validate_plate_format(plate_text)
    lookup = lookup_plate_record(plate_text, watchlist_path)
    heatmap_output_path = Path(heatmap_dir) / f"{Path(image_name).stem}_heatmap.jpg"
    heatmap_image, saved_heatmap_path = create_confidence_heatmap(
        annotated_frame,
        detections,
        output_path=heatmap_output_path if detections else None,
    )

    return {
        "annotated_image": annotated_frame,
        "heatmap_image": heatmap_image,
        "heatmap_path": saved_heatmap_path,
        "detections": detections,
        "top_detection": top_detection,
        "plate_crop": plate_crop,
        "plate_text": plate_text,
        "candidates": candidates[:5],
        "validation": validation,
        "lookup": lookup,
    }
