import cv2
import os

VIDEO_PATH = r"VIDEO RAG\data\videos"
FRAMES_PATH = r"VIDEO RAG\data\frames"

os.makedirs(FRAMES_PATH, exist_ok=True)

for file in os.listdir(VIDEO_PATH):
    video_path = os.path.join(VIDEO_PATH, file)
    video_name = os.path.splitext(file)[0]  # gets name without extension

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 10)
    count, saved = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            output_path = os.path.join(FRAMES_PATH, f"{video_name}_img_{saved}.jpg")
            cv2.imwrite(output_path, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"Saved {saved} frames for {video_name}")