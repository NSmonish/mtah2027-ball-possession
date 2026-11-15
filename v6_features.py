"""Compute team geometry features from ATD txt, one pass.
Plain punctuation, no em dashes.
Outputs v6_features.npz with per frame:
  cnt_h2, cnt_a2, cnt_h3, cnt_a3: players within 2m / 3m of ball, per team
  dist_h, dist_a: nearest player distance per team, NaN if none / no ball
  ball_x, ball_y already in atd_features.npz, reused elsewhere.
"""
import sys
import numpy as np
sys.path.insert(0, ".")
from pipeline import team_of_each_track, PITCH, N_FRAMES

RAW = "data/tactical_atd.txt"
META = "data/tactical_atd.xml"
length, width = PITCH
team_of_track = team_of_each_track(META)
cnt_h2 = np.zeros(N_FRAMES, dtype=np.int8)
cnt_a2 = np.zeros(N_FRAMES, dtype=np.int8)
cnt_h3 = np.zeros(N_FRAMES, dtype=np.int8)
cnt_a3 = np.zeros(N_FRAMES, dtype=np.int8)
dist_h = np.full(N_FRAMES, np.nan)
dist_a = np.full(N_FRAMES, np.nan)
n_ball = 0
with open(RAW) as handle:
    for line in handle:
        parts = line.split(":")
        if len(parts) != 3:
            continue
        ball = parts[2]
        if ball[0] == "N":
            continue
        try:
            frame = int(parts[0]) - 1
            comma = ball.index(",")
            bx = float(ball[:comma]) * length
            by = float(ball[comma + 1:ball.index(",", comma + 1)]) * width
        except Exception:
            continue
        if not (0 <= frame < N_FRAMES):
            continue
        n_ball += 1
        best_h = np.inf
        best_a = np.inf
        ch2 = ch3 = ca2 = ca3 = 0
        for slot, group in enumerate(parts[1].split(";")):
            if not group or group[0] == "N":
                continue
            try:
                comma = group.index(",")
                px = float(group[:comma]) * length
                py = float(group[comma + 1:group.index(",", comma + 1)]) * width
            except Exception:
                continue
            d = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
            team = team_of_track[slot] if slot < len(team_of_track) else -1
            if team == 0:
                if d < best_h:
                    best_h = d
                if d <= 2.0:
                    ch2 += 1
                if d <= 3.0:
                    ch3 += 1
            elif team == 1:
                if d < best_a:
                    best_a = d
                if d <= 2.0:
                    ca2 += 1
                if d <= 3.0:
                    ca3 += 1
        cnt_h2[frame] = ch2
        cnt_a2[frame] = ca2
        cnt_h3[frame] = ch3
        cnt_a3[frame] = ca3
        dist_h[frame] = best_h if best_h != np.inf else np.nan
        dist_a[frame] = best_a if best_a != np.inf else np.nan
print("ball frames", n_ball)
np.savez("./v6_features.npz",
         cnt_h2=cnt_h2, cnt_a2=cnt_a2, cnt_h3=cnt_h3, cnt_a3=cnt_a3,
         dist_h=dist_h, dist_a=dist_a)
print("wrote v6_features.npz")
