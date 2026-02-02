import cv2
import os

VIDEO_PATH = "/home/maria/MouseBox/selected_data/2021-10-02_08-07-55_segment1_mouse80_lever_side-view copy.avi"
OUT_DIR = "/home/maria/MouseBox/selected_data/video_frames"

os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened()

T_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Extracting", T_video, "frames")

for video_t in range(T_video):
    ok, frame = cap.read()
    if not ok:
        print("Failed at frame", video_t)
        break

    cv2.imwrite(
        os.path.join(OUT_DIR, f"frame_{video_t:06d}.jpg"),
        frame
    )

cap.release()
print("Done.")
