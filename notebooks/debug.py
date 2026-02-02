import cv2
import pandas as pd

CSV_PATH = "/home/maria/MouseBox/selected_data/models/gpu_working/video_preds/session_view.csv"
VIDEO_PATH = "/home/maria/MouseBox/selected_data/2021-10-02_08-07-55_segment1_mouse80_lever_side-view copy.avi"

dat = pd.read_csv(CSV_PATH)
T_pose = len(dat) - 2  # because first 2 rows are metadata

cap = cv2.VideoCapture(VIDEO_PATH)
T_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

tau, m = 3, 8
T_takens = T_pose - (m-1)*tau

print("T_video:", T_video)
print("T_pose:", T_pose)
print("T_takens:", T_takens)
print("Takens skips first frames:", (m-1)*tau)
print("Last video frame you can reach with current mapping:", T_pose-1)
