"""YOLOv8n ball detection on the tactical video, once it arrives.
Plain punctuation, no em dashes.
DO NOT RUN until the tactical video is complete on the VM.
Expected file: data/tactical.mp4
Expected size: [size removed] bytes. Verify size before running.
Run only on ATD ball missing frames in the public window first:
  frames 11185 to 18684, ATD ball is NaN on about 11.8 percent of them.
Usage:
  .venv/bin/python yolo/yolo_ball_tactical.py --start 11185 --end 18684 --conf 0.25
Output: yolo/tactical_ball_detections.csv with frame, ball_x, ball_y, conf.
Then compare detected ball positions to the ATD ball on frames where ATD
does have the ball, to measure false positives, before integrating anything
into the possession pipeline. At 8.9 fps CPU on 848x480, the full 199,319
frames at 1920x1080 would take many hours, so never run the full video blindly.
"""
import argparse
import csv
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO = "data/tactical.mp4"
MODEL = "yolo/yolov8n.pt"
ATD = "./atd_features.npz"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=11185)
    parser.add_argument("--end", type=int, default=18684)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default="yolo/tactical_ball_detections.csv")
    args = parser.parse_args()
    d = np.load(ATD)
    ball_missing = np.isnan(d["ball_x"])
    frames = [f for f in range(args.start, args.end) if ball_missing[f]]
    print(f"ATD ball missing in range: {len(frames)} of {args.end - args.start} frames")
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("cannot open video, is it downloaded and complete?")
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video frames {total}, size {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    t0 = time.time()
    rows = []
    for i, frame_no in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            rows.append((frame_no, -1, -1, 0.0))
            continue
        results = model.predict(frame, conf=args.conf, verbose=False, classes=[32])
        best = None
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if best is None or conf > best[0]:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    best = (conf, (x1 + x2) // 2, (y1 + y2) // 2)
        if best:
            rows.append((frame_no, best[1], best[2], best[0]))
        else:
            rows.append((frame_no, -1, -1, 0.0))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(frames)} frames, {time.time() - t0:.0f}s elapsed", flush=True)
    cap.release()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "ball_x_px", "ball_y_px", "conf"])
        w.writerows(rows)
    found = sum(1 for r in rows if r[1] >= 0)
    print(f"done, ball detected on {found}/{len(rows)} missing frames, wrote {args.out}")

if __name__ == "__main__":
    main()
