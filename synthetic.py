"""Synthetic ATD-like test data, for testing the full ATD pipeline without real match data.
No labels from the provided game are used. This is a simulated match, generated from
football priors: possessions of mean about 14 s, dead balls of 5 to 20 s, ball missing
in about 40 percent of frames in short gaps, nearest player flicker.

It writes:
  - synthetic_atd.txt in the same format baseline.py parses: "<frame>:<player>;...:<ball>"
  - synthetic_gt.csv, frame,label, 199319 rows, with X outside the simulated graded window
Use run_evaluation.py with --gt synthetic_gt.csv --results <dir with pipeline output>.
"""
import numpy as np

N = 199319
FPS = 25
PITCH = (105.3, 68.0)
rng = np.random.default_rng(7)


def make(n_possessions=300, seed=7):
    r = np.random.default_rng(seed)
    labels = np.full(N, "X", dtype="<U1")
    # Simulate only a graded window, frames 11185 to 90651, like half 1, for speed.
    # Outside that, X.
    frame = 11185
    team = "H"
    gt_runs = []
    while frame < 90651 and len(gt_runs) < n_possessions:
        if r.random() < 0.18:
            dur = int(r.integers(125, 500))
            lab = "D"
        else:
            dur = int(r.lognormal(mean=np.log(14 * FPS), sigma=0.6))
            dur = int(np.clip(dur, 20, 1500))
            lab = team
            team = "A" if team == "H" else "H"
        dur = min(dur, 90651 - frame)
        labels[frame:frame + dur] = lab
        gt_runs.append((lab, frame, frame + dur))
        frame += dur
    # Ball position: moves with the possessing team, plus noise. Players: 22, two teams,
    # random walk around the ball, nearest player is usually the possessing team, with
    # flicker: on 15 percent of live frames the nearest player is the other team.
    ball_x = np.full(N, np.nan)
    ball_y = np.full(N, np.nan)
    raw_codes = np.full(N, -1, dtype=np.int8)
    bx, by = 50.0, 34.0
    for i in range(11185, 90651):
        lab = labels[i]
        if lab == "D":
            continue
        bx = np.clip(bx + r.normal(0, 0.4), 2, 103)
        by = np.clip(by + r.normal(0, 0.3), 2, 66)
        # Ball missing: 40 percent, in gaps of 5 to 50 frames
        if r.random() < 0.40:
            continue
        ball_x[i] = bx
        ball_y[i] = by
        true_team = 0 if lab == "H" else 1
        # Flicker
        raw_codes[i] = 1 - true_team if r.random() < 0.15 else true_team
    return labels, ball_x, ball_y, raw_codes, gt_runs


if __name__ == "__main__":
    labels, bx, by, codes, runs = make()
    with open("synthetic_gt.csv", "w") as f:
        f.write("frame,label\n")
        for i, lab in enumerate(labels):
            f.write(f"{i},{lab}\n")
    print("synthetic GT runs:", len(runs), "possessions:", sum(1 for lab, _, _ in runs if lab in "HA"))
    print("ball present:", int(np.sum(~np.isnan(bx))), "of", 90651 - 11185)
    print("raw code counts:", {int(k): int(v) for k, v in zip(*np.unique(codes, return_counts=True))})
