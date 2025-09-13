import cv2
import numpy as np
import argparse
from pathlib import Path

# ------------- args -------------
ap = argparse.ArgumentParser()
ap.add_argument("--source", type=str, default="1",
                help='Camera index like "0" or a video path like "clip.mp4"')
ap.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
ap.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold")
ap.add_argument("--width", type=int, default=640, help="inference width (speed/accuracy)")
ap.add_argument("--height", type=int, default=640, help="inference height (speed/accuracy)")
ap.add_argument("--cfg", type=str, default="yolov3-tiny.cfg")
ap.add_argument("--weights", type=str, default="yolov3-tiny.weights")
ap.add_argument("--names", type=str, default="coco.names")
args = ap.parse_args()

# ------------- load classes -------------
with open(args.names, "r", encoding="utf-8") as f:
    classes = [c.strip() for c in f.readlines()]
# Normalize common spelling variants
AIRPLANE_NAMES = {"airplane", "aeroplane"}

# ------------- build net -------------
net = cv2.dnn.readNetFromDarknet(args.cfg, args.weights)
# Use CPU; if you have OpenCV built with CUDA you can switch:
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ------------- video source -------------
# Allow "0" (string) to mean webcam index 0
src = 1 if args.source == "1" else args.source
cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if isinstance(src, int) else 0)
if not cap.isOpened():
    raise SystemExit(f"Could not open source: {args.source}")

# ------------- helpers -------------
def postprocess(frame, outputs, conf_th, nms_th):
    H, W = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    for out in outputs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            score = float(scores[class_id])
            if score < conf_th:
                continue
            # YOLO boxes are center-x, center-y, width, height (relative)
            cx, cy, w, h = det[0:4] * np.array([W, H, W, H])
            x = int(cx - w / 2)
            y = int(cy - h / 2)
            boxes.append([x, y, int(w), int(h)])
            confidences.append(score)
            class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, nms_th)
    if len(idxs) == 0:
        return []

    idxs = idxs.flatten()
    return [(class_ids[i], confidences[i], boxes[i]) for i in idxs]

# ------------- main loop -------------
while True:
    ok, frame = cap.read()
    if not ok:
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (args.width, args.height), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)

    detections = postprocess(frame, outputs, args.conf, args.nms)

    # Draw only airplanes
    for cid, conf, (x, y, w, h) in detections:
        label = classes[cid] if 0 <= cid < len(classes) else str(cid)
        if label.lower() not in AIRPLANE_NAMES:
            continue
        x = max(0, x); y = max(0, y)
        x2 = min(frame.shape[1]-1, x + w)
        y2 = min(frame.shape[0]-1, y + h)

        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x, max(0, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Airplane detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
