import cv2
import os
import json
import numpy as np
from skimage.metrics import structural_similarity as ssim

VIDEO_PATH = r"VIDEO RAG\data\video"
FRAMES_PATH = r"VIDEO RAG\data\frame"

os.makedirs(FRAMES_PATH, exist_ok=True)

def frame_similarity(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.resize(gray1, (256, 256))
    gray2 = cv2.resize(gray2, (256, 256))

    score, _ = ssim(gray1, gray2, full=True)
    return score


for file in os.listdir(VIDEO_PATH):

    video_path = os.path.join(VIDEO_PATH, file)
    video_name = os.path.splitext(file)[0]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps*5)

    frame_index = 0
    saved_index = 0

    prev_frame = None
    metadata = []

    while True:

        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_interval == 0:
            timestamp = frame_index / fps
            if prev_frame is None:
                img_name = f"{video_name}_img_{saved_index}.jpg"
                img_path = os.path.join(FRAMES_PATH, img_name)
                cv2.imwrite(img_path, frame)

                metadata.append({
                    "start": timestamp,
                    "end": timestamp,
                    "path": img_path
                })
                prev_frame = frame
                saved_index += 1

            else:
                similarity = frame_similarity(prev_frame, frame)

                if similarity > 0.85:
                    metadata[-1]["end"] = timestamp
                else:
                    img_name = f"{video_name}_img_{saved_index}.jpg"
                    img_path = os.path.join(FRAMES_PATH, img_name)
                    cv2.imwrite(img_path, frame)

                    metadata.append({
                        "start": timestamp,
                        "end": timestamp,
                        "path": img_path
                    })

                    prev_frame = frame
                    saved_index += 1

        frame_index += 1

    cap.release()

    json_path = os.path.join(
        FRAMES_PATH,
        f"{video_name}_metadata.json"
    )

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved {saved_index} unique frames")