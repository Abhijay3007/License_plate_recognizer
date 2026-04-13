import io
import logging
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError
from flask import Flask, Response, render_template, request
from werkzeug.exceptions import HTTPException

from ai.ai_model import detection, load_yolov5_model
from ai.ocr_model import easyocr_model_load
from helper.general_utils import filter_text, save_results
from helper.params import Parameters

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.logger.setLevel(logging.INFO)

LOG_PATH = Path("flask_error.log")
if not any(isinstance(handler, logging.FileHandler) for handler in app.logger.handlers):
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    app.logger.addHandler(file_handler)

params = Parameters()
model, labels = load_yolov5_model()
text_reader = easyocr_model_load()
STATIC_IMAGE_PATH = Path("static") / "image0.jpg"
INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")
ONE_LETTER_SERIES_REGEX = re.compile(r"^[A-Z]{2}\d{2}[A-Z]\d{4}$")
TWO_LETTER_SERIES_REGEX = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$")
VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB",
}
LETTER_SUBS = {
    "0": "D",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "Z",
    "8": "B",
}
DIGIT_SUBS = {
    "A": "4",
    "B": "8",
    "D": "0",
    "G": "6",
    "I": "1",
    "L": "1",
    "O": "0",
    "Q": "0",
    "S": "5",
    "Z": "7",
}
SERIES_INSERTION_MAP = {
    "P": ["B", "R", "D"],
    "R": ["B", "P", "H"],
    "B": ["P", "R", "D"],
    "D": ["B", "P", "R"],
    "M": ["N", "H"],
    "N": ["M", "H"],
    "H": ["M", "N", "R"],
}
WEBCAM_OCR_INTERVAL = 10


def normalize_plate_text(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return [
        image,
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR),
    ]


def resize_for_ocr(image):
    height, width = image.shape[:2]
    target_width = max(params.ocr_input_width, 640)

    if width > 0 and width < target_width:
        resized_height = max(1, int(height * target_width / width))
        return cv2.resize(
            image, (target_width, resized_height), interpolation=cv2.INTER_CUBIC
        )
    if width > 960:
        resized_height = max(1, int(height * params.ocr_input_width / width))
        return cv2.resize(
            image, (params.ocr_input_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
    return image


def generate_plate_views(image):
    if image is None or image.size == 0:
        return []

    image = resize_for_ocr(image)
    height, width = image.shape[:2]
    views = [image]

    if width > 100:
        left_trim = int(width * 0.18)
        views.append(image[:, left_trim:].copy())

    if height > 60:
        split = height // 2
        top_half = image[:split, :].copy()
        bottom_half = image[split:, :].copy()
        views.extend([top_half, bottom_half])
        if width > 100:
            left_trim = int(width * 0.18)
            views.extend(
                [top_half[:, left_trim:].copy(), bottom_half[:, left_trim:].copy()]
            )

    return [view for view in views if view.size]


def generate_scene_rois(image):
    if image is None or image.size == 0:
        return []

    height, width = image.shape[:2]
    rois = []
    roi_specs = [
        (0.50, 0.98, 0.30, 0.72),
        (0.56, 0.95, 0.34, 0.68),
        (0.60, 0.92, 0.36, 0.66),
        (0.45, 0.85, 0.25, 0.75),
        (0.58, 0.96, 0.08, 0.42),
        (0.62, 0.98, 0.10, 0.40),
        (0.55, 0.90, 0.12, 0.45),
        (0.52, 0.88, 0.18, 0.52),
    ]

    for y1_ratio, y2_ratio, x1_ratio, x2_ratio in roi_specs:
        y1 = max(0, int(height * y1_ratio))
        y2 = min(height, int(height * y2_ratio))
        x1 = max(0, int(width * x1_ratio))
        x2 = min(width, int(width * x2_ratio))
        roi = image[y1:y2, x1:x2].copy()
        if roi.size:
            rois.append(roi)

    return rois


def generate_dense_plate_rois(image):
    if image is None or image.size == 0:
        return []

    height, width = image.shape[:2]
    rois = []
    widths = [0.18, 0.24, 0.30]
    heights = [0.08, 0.12, 0.16]
    x_centers = [0.20, 0.26, 0.32, 0.38, 0.44, 0.50]
    y_centers = [0.62, 0.68, 0.74, 0.80, 0.86]

    for h_ratio in heights:
        for w_ratio in widths:
            box_h = int(height * h_ratio)
            box_w = int(width * w_ratio)
            for cx_ratio in x_centers:
                for cy_ratio in y_centers:
                    cx = int(width * cx_ratio)
                    cy = int(height * cy_ratio)
                    x1 = max(0, cx - box_w // 2)
                    y1 = max(0, cy - box_h // 2)
                    x2 = min(width, x1 + box_w)
                    y2 = min(height, y1 + box_h)
                    roi = image[y1:y2, x1:x2].copy()
                    if roi.size:
                        rois.append(roi)

    return rois


def score_plate_candidate(candidate):
    score = 0
    if INDIAN_PLATE_REGEX.fullmatch(candidate):
        score += 100
    if ONE_LETTER_SERIES_REGEX.fullmatch(candidate):
        score += 18
    if TWO_LETTER_SERIES_REGEX.fullmatch(candidate):
        score += 30
    if len(candidate) == 10:
        score += 35
    if len(candidate) == 9:
        score += 8
    if candidate[:2].isalpha():
        score += 10
    if candidate[-4:].isdigit():
        score += 10

    digit_count = sum(char.isdigit() for char in candidate)
    letter_count = sum(char.isalpha() for char in candidate)
    score += min(digit_count, 6) + min(letter_count, 4)
    score -= abs(len(candidate) - 10) * 3
    return score


def is_plausible_plate(candidate):
    candidate = normalize_plate_text(candidate)
    if len(candidate) < 8 or len(candidate) > 10:
        return False
    if candidate[:2] not in VALID_STATE_CODES:
        return False
    if sum(char.isdigit() for char in candidate) < 5:
        return False
    if not candidate[-4:].isdigit():
        return False
    if not candidate[:2].isalpha():
        return False
    if len(candidate) >= 4 and not candidate[2:4].isdigit():
        return False
    return True


def force_char_type(char, want_alpha):
    if want_alpha:
        if char.isalpha():
            return char
        return LETTER_SUBS.get(char, char)
    if char.isdigit():
        return char
    return DIGIT_SUBS.get(char, char)


def build_plate_variants(candidate):
    candidate = normalize_plate_text(candidate)
    variants = {candidate}

    if len(candidate) == 10:
        variants.add(
            "".join(
                [
                    force_char_type(candidate[0], True),
                    force_char_type(candidate[1], True),
                    force_char_type(candidate[2], False),
                    force_char_type(candidate[3], False),
                    force_char_type(candidate[4], True),
                    force_char_type(candidate[5], True),
                    force_char_type(candidate[6], False),
                    force_char_type(candidate[7], False),
                    force_char_type(candidate[8], False),
                    force_char_type(candidate[9], False),
                ]
            )
        )
        variants.add(
            "".join(
                [
                    force_char_type(candidate[0], True),
                    force_char_type(candidate[1], True),
                    force_char_type(candidate[2], False),
                    force_char_type(candidate[3], False),
                    force_char_type(candidate[4], True),
                    force_char_type(candidate[5], True),
                    force_char_type(candidate[6], True),
                    force_char_type(candidate[7], False),
                    force_char_type(candidate[8], False),
                    force_char_type(candidate[9], False),
                ]
            )
        )

        # OCR sometimes inserts one extra character into the series block for
        # two-line plates, e.g. KA43RZ7827 instead of KA43R7827.
        if (
            candidate[:2].isalpha()
            and candidate[2:4].isdigit()
            and candidate[-4:].isdigit()
            and candidate[4:6].isalpha()
        ):
            variants.add(candidate[:4] + candidate[5:])
            variants.add(candidate[:5] + candidate[6:])

    if (
        len(candidate) == 9
        and candidate[:2].isalpha()
        and candidate[2:4].isdigit()
        and candidate[4].isalpha()
        and candidate[-4:].isdigit()
    ):
        existing_series_char = candidate[4]
        for inserted_char in SERIES_INSERTION_MAP.get(existing_series_char, []):
            variants.add(candidate[:4] + inserted_char + candidate[4:])
            variants.add(candidate[:5] + inserted_char + candidate[5:])
        for inserted_char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            variants.add(candidate[:4] + inserted_char + candidate[4:])
            variants.add(candidate[:5] + inserted_char + candidate[5:])

    return {variant for variant in variants if len(variant) >= 6}


def collect_texts_from_results(results, variant, use_region_filter):
    if use_region_filter:
        texts = filter_text(
            max(variant.shape[0] * variant.shape[1], 1),
            results,
            params.region_threshold,
        )
    else:
        texts = [result[1] for result in results]

    normalized_texts = []
    for text in texts:
        normalized = normalize_plate_text(text)
        if len(normalized) >= 4:
            normalized_texts.append(normalized)
    return normalized_texts


def extract_plate_text(image, use_region_filter=True):
    if image is None or image.size == 0:
        return []

    candidates = []
    for view in generate_plate_views(image):
        split_candidates = []
        for variant in preprocess_for_ocr(view):
            detailed_results = text_reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
            paragraph_results = text_reader.readtext(
                variant,
                detail=0,
                paragraph=True,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )

            ordered_results = sorted(
                detailed_results,
                key=lambda result: (
                    round(min(point[1] for point in result[0]) / 20),
                    min(point[0] for point in result[0]),
                ),
            )
            combined_text = normalize_plate_text(
                "".join(result[1] for result in ordered_results)
            )
            if len(combined_text) >= 6:
                candidates.append(combined_text)

            for paragraph_text in paragraph_results:
                normalized = normalize_plate_text(paragraph_text)
                if len(normalized) >= 4:
                    candidates.append(normalized)

            candidates.extend(
                collect_texts_from_results(detailed_results, variant, use_region_filter)
            )
            split_candidates.extend(
                collect_texts_from_results(detailed_results, variant, False)
            )

        if len(split_candidates) >= 2:
            merged_split = normalize_plate_text("".join(split_candidates[:2]))
            if len(merged_split) >= 6:
                candidates.append(merged_split)

    unique_candidates = list(dict.fromkeys(candidates))
    corrected_candidates = []
    for candidate in unique_candidates:
        corrected_candidates.extend(build_plate_variants(candidate))

    ranked_candidates = [
        candidate
        for candidate in dict.fromkeys(corrected_candidates)
        if is_plausible_plate(candidate)
    ]
    ranked_candidates.sort(key=score_plate_candidate, reverse=True)
    return ranked_candidates


def detect_plate_candidates(frame):
    detected, _, plate_crop = detection(frame, model, labels)
    texts = extract_plate_text(plate_crop, use_region_filter=True)
    if not texts:
        texts = extract_plate_text(frame, use_region_filter=False)
    if not texts:
        for roi in generate_scene_rois(frame):
            texts = extract_plate_text(roi, use_region_filter=False)
            if texts:
                break
    if not texts:
        for roi in generate_dense_plate_rois(frame):
            texts = extract_plate_text(roi, use_region_filter=False)
            if texts:
                break
    return detected, texts


def decode_uploaded_image(file_storage):
    img_bytes = file_storage.read()
    if not img_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(
            "The selected file is not a supported image. Please upload JPG, PNG, BMP, or WEBP."
        ) from exc

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def encode_frame(frame):
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return None
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    )


def status_frame(message, width=960, height=720):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (18, 30, 27)
    cv2.rectangle(frame, (40, 40), (width - 40, height - 40), (27, 74, 60), 2)
    cv2.putText(
        frame,
        "ANPR Webcam",
        (70, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (220, 242, 236),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message,
        (70, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (220, 242, 236),
        2,
        cv2.LINE_AA,
    )
    return frame


def open_webcam():
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(None)

    for backend in backends:
        camera = cv2.VideoCapture(0, backend) if backend is not None else cv2.VideoCapture(0)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return camera
        camera.release()
    return None


def webcam_frames():
    camera = open_webcam()
    if camera is None:
        frame = status_frame("Camera not available. Close other apps using the webcam.")
        payload = encode_frame(frame)
        if payload:
            while True:
                yield payload

    frame_index = 0
    last_text = "Scanning..."

    try:
        while True:
            try:
                success, frame = camera.read()
                if not success or frame is None:
                    fallback = status_frame("Unable to read webcam frames.")
                    payload = encode_frame(fallback)
                    if payload:
                        yield payload
                    continue

                frame_index += 1
                detected_frame, _, _ = detection(frame, model, labels)

                if frame_index % WEBCAM_OCR_INTERVAL == 0:
                    detected_frame, texts = detect_plate_candidates(frame)
                    if texts:
                        last_text = texts[0]

                display_frame = detected_frame.copy()
                cv2.rectangle(display_frame, (16, 16), (520, 78), (14, 32, 28), -1)
                cv2.putText(
                    display_frame,
                    f"Plate: {last_text}",
                    (28, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (230, 248, 241),
                    2,
                    cv2.LINE_AA,
                )

                payload = encode_frame(display_frame)
                if payload:
                    yield payload
            except Exception:
                fallback = status_frame("Webcam stream recovered from an internal error.")
                payload = encode_frame(fallback)
                if payload:
                    yield payload
    finally:
        if camera is not None:
            camera.release()


@app.route("/", methods=["GET", "POST"])
def predict():
    image_url = None
    plate_text = None
    candidates = []
    error = None
    warning = None

    if request.method == "POST":
        file = request.files.get("file")
        if file is None or not file.filename:
            error = "Choose an image file first."
        else:
            try:
                frame = decode_uploaded_image(file)
                detected, texts = detect_plate_candidates(frame)

                candidates = texts[:5]
                if texts:
                    plate_text = texts[0]
                    try:
                        save_results(plate_text, "ocr_results.csv", "Detection_Images")
                    except PermissionError:
                        app.logger.warning(
                            "Could not write to ocr_results.csv because the file is in use."
                        )
                        warning = (
                            "Plate detected, but `ocr_results.csv` is currently in use. "
                            "Close that file in Excel or another app if you want new results saved."
                        )
                else:
                    plate_text = "No readable number plate found."

                STATIC_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)).save(
                    STATIC_IMAGE_PATH
                )
                image_url = f"/{STATIC_IMAGE_PATH.as_posix()}"
            except ValueError as exc:
                error = str(exc)
            except Exception:
                app.logger.exception("Upload inference failed")
                error = (
                    "The server could not process this image. "
                    "Check the Flask console for the traceback and try another image."
                )

    return render_template(
        "index.html",
        image_url=image_url,
        plate_text=plate_text,
        candidates=candidates,
        error=error,
        warning=warning,
    )


@app.errorhandler(413)
def request_entity_too_large(_error):
    return (
        render_template(
            "index.html",
            image_url=None,
            plate_text=None,
            candidates=[],
            error="The uploaded image is too large. Please keep it under 10 MB.",
            warning=None,
        ),
        413,
    )


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if isinstance(error, HTTPException):
        return error

    app.logger.exception("Unhandled application error")
    return (
        render_template(
            "index.html",
            image_url=None,
            plate_text=None,
            candidates=[],
            error=(
                "The server hit an unexpected error. "
                "Check flask_error.log in the project folder for the traceback."
            ),
            warning=None,
        ),
        500,
    )


@app.route("/video")
def video():
    return Response(
        webcam_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
