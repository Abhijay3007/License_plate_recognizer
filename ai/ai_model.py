import numpy as np
import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, xyxy2xywh
import torch.backends.cudnn as cudnn

from cv2 import VideoCapture
from utils.params import Parameters

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


def load_yolov5_model():
    """
    It loads the model and returns the model and the names of the classes.
    :return: model, names
    """
    model = attempt_load(params.model, map_location=params.device)
    print("device", params.device)
    stride = int(model.stride.max())  # model stride
    names = (
        model.module.names if hasattr(model, "module") else model.names
    )  # get class names

    return model, names


def detection(frame, model, names):
    """
    It takes an image, runs it through the model, and returns the image with bounding boxes drawn around
    the detected objects
    
    :param frame: The frame of video or webcam feed on which we're running inference
    :param model: The model to use for detection
    :param names: a list of class names
    :return: the image with the bounding boxes and the label of the detected object.
    """
    annotated_frame, detections = detect_plate_regions(frame, model, names)
    label = detections[0]["label"] if detections else ""
    plate_crop = detections[0]["crop"] if detections else None
    return annotated_frame, label, plate_crop


def detect_plate_regions(frame, model, names):
    """
    Run YOLOv5 on the frame and return the annotated frame plus structured
    detection metadata for downstream features such as heatmaps and speed
    estimation.
    """
    out = frame.copy()

    from utils.augmentations import letterbox
    frame = letterbox(frame, new_shape=(params.pred_shape[0], params.pred_shape[1]), auto=False)[0]
    
    # YOLO expects CHW and RGB order
    frame = frame.transpose((2, 0, 1))[::-1]
    frame = np.ascontiguousarray(frame)

    cudnn.benchmark = True  # set True to speed up constant image size inference

    if params.device.type != "cpu":
        model(
            torch.zeros(1, 3, params.pred_shape[0], params.pred_shape[1])
            .to(params.device)
            .type_as(next(model.parameters()))
        )  # run once

    frame = torch.from_numpy(frame).to(params.device)
    frame = frame.float()
    frame /= 255.0
    if frame.ndimension() == 3:
        frame = frame.unsqueeze(0)

    pred = model(frame, augment=False)[0]
    pred = non_max_suppression(pred, params.conf_thres, max_det=params.max_det)

    detections = []
    # detections per image
    for i, det in enumerate(pred):

        img_shape = frame.shape[2:]
        out_shape = out.shape

        s_ = f"{i}: "
        s_ += "%gx%g " % img_shape  # print string

        if len(det):

            gain = min(
                img_shape[0] / out_shape[0], img_shape[1] / out_shape[1]
            )  # gain  = old / new

            coords = det[:, :4]

            pad = (
                (img_shape[1] - out_shape[1] * gain) / 2,
                (img_shape[0] - out_shape[0] * gain) / 2,
            )  # wh padding

            coords[:, [0, 2]] -= pad[0]  # x padding
            coords[:, [1, 3]] -= pad[1]  # y padding
            coords[:, :4] /= gain

            coords[:, 0].clamp_(0, out_shape[1])  # x1
            coords[:, 1].clamp_(0, out_shape[0])  # y1
            coords[:, 2].clamp_(0, out_shape[1])  # x2
            coords[:, 3].clamp_(0, out_shape[0])  # y2

            det[:, :4] = coords.round()

            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s_ += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            ordered_detections = sorted(det, key=lambda row: float(row[4]), reverse=True)
            for *xyxy, conf, cls in ordered_detections:

                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())

                # Add a small 10% padding around the YOLO crop
                cw = x2 - x1
                ch = y2 - y1
                px = max(2, int(cw * 0.10))
                py = max(2, int(ch * 0.10))

                cx1 = max(0, x1 - px)
                cy1 = max(0, y1 - py)
                cx2 = min(out.shape[1], x2 + px)
                cy2 = min(out.shape[0], y2 + py)

                detected_plate = out[cy1:cy2, cx1:cx2].copy()
                if detected_plate.size:
                    safe_imshow("Cropped Plate", detected_plate)

                # rect_size= (detected_plate.shape[0]*detected_plate.shape[1])
                c = int(cls)  # integer class
                label = names[c] if params.hide_conf else f"{names[c]} {conf:.2f}"

                detections.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float(conf.item()),
                        "class_id": c,
                        "class_name": names[c],
                        "label": label,
                        "crop": detected_plate if detected_plate.size else None,
                        "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    }
                )

                tl = params.rect_thickness

                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(
                    out, c1, c2, params.color, thickness=tl, lineType=cv2.LINE_AA
                )

                if label:
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[
                        0
                    ]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(out, c1, c2, params.color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(
                        out,
                        label,
                        (c1[0], c1[1] - 2),
                        0,
                        tl / 3,
                        [225, 255, 255],
                        thickness=tf,
                        lineType=cv2.LINE_AA,
                    )

    return out, detections
    # fps = 'FPS: {0:.2f}'.format(frame_rate_calc)
    # label_size, base_line = cv2.getTextSize(fps, cv2.FONT_HERSHEY_SIMPLEX, params.font_scale, params.thickness)
    # label_ymin = max(params.fps_y, label_size[1] + 10)
    # cv2.rectangle(out, (params.text_x_align, label_ymin - label_size[1] - 10),
    #               (params.text_x_align + label_size[0], label_ymin + base_line - 10), (255, 255, 255),
    #               cv2.FILLED)
    # cv2.rectangle(out, (params.text_x_align - 2, label_ymin - label_size[1] - 12),
    #               (params.text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
    # cv2.putText(out, fps, (params.text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, params.font_scale,
    #             params.color_blue,
    #             params.thickness,
    #             cv2.LINE_AA)
