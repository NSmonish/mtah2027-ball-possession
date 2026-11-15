"""YOLOv8n ball detection test on the undergrad football clip.
Plain punctuation, no em dashes.
Uses the COCO yolov8n.pt already on the VM. Classes: 0 person, 32 sports ball.
Runs on a small sample first, 300 frames, then reports detection rates.
This proves the code path that will run on the tactical video once it arrives,
on ball missing frames only.
"""
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO = "yolo/football_match.mp4"
MODEL = "yolo/yolov8n.pt"
MAX_FRAMES = 300

def main():
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("cannot open video")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video {width}x{height} fps {fps} total frames {total}")
    t0 = time.time()
    n = 0
    ball_frames = 0
    ball_dets = 0
    person_frames = 0
    person_dets = 0
    ball_conf = []
    out_rows = []
    while n < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        results = model.predict(frame, conf=0.25, verbose=False, classes=[0, 32])
        b = 0
        p = 0
        bx = by = -1
        for result in results:
            for box in result.boxes:
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                if cid == 32:
                    b += 1
                    ball_conf.append(conf)
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    bx = (x1 + x2) // 2
                    by = (y1 + y2) // 2
                elif cid == 0:
                    p += 1
        if b:
            ball_frames += 1
            ball_dets += b
        if p:
            person_frames += 1
            person_dets += p
        out_rows.append((n, b, p, bx, by))
    cap.release()
    dt = time.time() - t0
    print(f"processed {n} frames in {dt:.1f}s, {n/dt:.1f} fps")
    print(f"ball: frames with ball {ball_frames}/{n} ({ball_frames/n:.1%}), total dets {ball_dets}, mean conf {np.mean(ball_conf):.3f}" if ball_conf else "ball: no detections")
    print(f"person: frames with person {person_frames}/{n} ({person_frames/n:.1%}), total dets {person_dets}, mean per frame {person_dets/n:.1f}")
    import csv
    with open("yolo/ball_test_300.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "n_ball", "n_person", "ball_x", "ball_y"])
        w.writerows(out_rows)
    print("wrote yolo/ball_test_300.csv")

if __name__ == "__main__":
    main()
