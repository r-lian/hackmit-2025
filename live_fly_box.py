import cv2
import numpy as np

# --- knobs you can tweak ---
CAM_INDEX = 1                 # try 1 or 2 if you have multiple cameras
VAR_TH    = 40                 # higher = stricter motion detection
MIN_AREA  = 10                 # smallest blob area to consider (px^2)
MAX_AREA  = 500               # largest blob area to consider
BLUR_K    = 5                  # odd blur kernel size; 1 disables
KERNEL    = (3, 3)             # morphology kernel

# On Windows, CAP_DSHOW often reduces startup lag
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera index {CAM_INDEX}")

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=VAR_TH, detectShadows=False
)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if BLUR_K > 1:
        gray = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)

    fg = fgbg.apply(gray)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / (h + 1e-6)
        if 0.3 < aspect < 3.3 and area > best_area:
            best_area = area
            best_box = (x, y, w, h)

    draw = frame.copy()
    if best_box is not None:
        x, y, w, h = best_box
        cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(draw, f"fly ~ area {int(best_area)}", (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("mask", fg)
    cv2.imshow("live", draw)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):   # quit
        break

cap.release()
cv2.destroyAllWindows()
