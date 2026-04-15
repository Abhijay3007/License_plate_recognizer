import os

with open('ai/ai_model.py', 'r', encoding='utf-8') as f:
    text = f.read()

target1 = """    from utils.augmentations import letterbox
    frame = letterbox(frame, new_shape=(params.pred_shape[0], params.pred_shape[1]), auto=False)[0]
    frame = np.transpose(frame, (2, 1, 0))

    cudnn.benchmark = True  # set True to speed up constant image size inference

    if params.device.type != "cpu":
        model(
            torch.zeros(1, 3, params.imgsz, params.imgsz)
            .to(params.device)
            .type_as(next(model.parameters()))
        )  # run once

    frame = torch.from_numpy(frame).to(params.device)
    frame = frame.float()
    frame /= 255.0
    if frame.ndimension() == 3:
        frame = frame.unsqueeze(0)

    frame = torch.transpose(frame, 2, 3)"""

replacement1 = """    from utils.augmentations import letterbox
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
        frame = frame.unsqueeze(0)"""


target2 = """                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())

                detected_plate = out[y1:y2, x1:x2].copy()
                if detected_plate.size:
                    safe_imshow("Crooped Plate ", detected_plate)"""

replacement2 = """                x1 = int(xyxy[0].item())
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
                    safe_imshow("Cropped Plate", detected_plate)"""

text = text.replace('\r', '')

if target1 in text:
    text = text.replace(target1, replacement1)
    if target2 in text:
        text = text.replace(target2, replacement2)
        with open('ai/ai_model.py', 'w', encoding='utf-8') as f:
            f.write(text)
        print('Successfully replaced')
    else:
        print('Target2 not found. First 100 chars of target 2:', repr(target2[:100]))
else:
    print('Target1 not found. First 100 chars of target 1:', repr(target1[:100]))
