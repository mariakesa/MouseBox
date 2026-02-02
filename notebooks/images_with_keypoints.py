import cv2
import os
import pandas as pd
import numpy as np


VIDEO_PATH = "/home/maria/MouseBox/selected_data/2021-10-02_08-07-55_segment1_mouse80_lever_side-view copy.avi"
CSV_PATH = "/home/maria/MouseBox/selected_data/models/gpu_working/video_preds/session_view.csv"
OUT_DIR = "/home/maria/MouseBox/selected_data/video_frames_kp"

os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load pose data ----
dat = pd.read_csv(CSV_PATH)

bodyparts = dat.iloc[0, 1:].values
coords = dat.iloc[1, 1:].values
columns = [f"{bp}_{c}" for bp, c in zip(bodyparts, coords)]

X = dat.iloc[2:, 1:].astype(float)
X.columns = columns
X.reset_index(drop=True, inplace=True)

xy_cols = [c for c in X.columns if c.endswith("_x") or c.endswith("_y")]
X_xy = X[xy_cols].values.reshape(-1, 3, 2)

# ---- Video ----
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened()

T_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
T_pose = len(X_xy)

assert T_video == T_pose, "Video and pose length mismatch"

# ---- Drawing params ----
RADIUS = 6
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR for OpenCV

print("Saving annotated frames...")

for t in range(T_video):
    ok, frame = cap.read()
    if not ok:
        break

    pts = X_xy[t]

    for i, (x, y) in enumerate(pts):
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(
                frame,
                (int(x), int(y)),
                RADIUS,
                COLORS[i],
                thickness=-1,
            )

    out_path = os.path.join(OUT_DIR, f"frame_{t:06d}.jpg")
    cv2.imwrite(out_path, frame)

cap.release()
print("Done.")
