# Importing the necessary libraries for the program to run.
from ai.ai_model import load_yolov5_model
from ai.ai_model import detection

from helper.params import Parameters
from helper.general_utils import filter_text
from helper.general_utils import save_results

from ai.ocr_model import easyocr_model_load
from ai.ocr_model import easyocr_model_works
from utils.visual_utils import *

import cv2
from datetime import datetime

# Loading the parameters from the params.py file.
params = Parameters()
_CAN_SHOW_WINDOWS = None


def safe_imshow(window_name, image):
    global _CAN_SHOW_WINDOWS

    if _CAN_SHOW_WINDOWS is False:
        return

    try:
        cv2.imshow(window_name, image)
        _CAN_SHOW_WINDOWS = True
    except cv2.error:
        _CAN_SHOW_WINDOWS = False


def safe_wait_key(delay=1):
    if _CAN_SHOW_WINDOWS is False:
        return -1

    try:
        return cv2.waitKey(delay)
    except cv2.error:
        return -1


def safe_destroy_all_windows():
    if _CAN_SHOW_WINDOWS is False:
        return

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


if __name__ == "__main__":

    # Loading the model and labels from the ai_model.py file.
    model, labels = load_yolov5_model()
    # Capturing the video from the webcam.
    camera = cv2.VideoCapture(0)
    # Loading the model for the OCR.
    text_reader = easyocr_model_load()
    frame_index = 0

    while 1:

        # Reading the video from the webcam.
        ret, frame = camera.read()
        if ret:
            frame_index += 1

            # Detecting the text from the image.
            detected, _, plate_crop = detection(frame, model, labels)
            should_run_ocr = frame_index % params.ocr_every_n_frames == 0
            if plate_crop is not None and should_run_ocr:
                # OCR on the cropped plate is much faster and more reliable than
                # scanning the full camera frame each iteration.
                plate_height, plate_width = plate_crop.shape[:2]
                if plate_width > 0 and plate_width > params.ocr_input_width:
                    resized_height = max(
                        1, int(plate_height * params.ocr_input_width / plate_width)
                    )
                    plate_crop = cv2.resize(
                        plate_crop,
                        (params.ocr_input_width, resized_height),
                        interpolation=cv2.INTER_LINEAR,
                    )

                resulteasyocr = text_reader.readtext(
                    plate_crop, detail=1, paragraph=False
                )
                text = filter_text(
                    max(plate_crop.shape[0] * plate_crop.shape[1], 1),
                    resulteasyocr,
                    params.region_threshold,
                )
                if text:
                    # Save the most recent OCR result when a plate was actually read.
                    save_results(text[-1], "ocr_results.csv", "Detection_Images")
                    print(text)
            safe_imshow("detected", detected)

        if safe_wait_key(1) & 0xFF == 27:
            safe_destroy_all_windows()
            break
