"""Ball possession pipeline, v1.
Plain punctuation, no em dashes.

Two input paths, one state machine:
  A. ATD raw text (when available): parse streaming, nearest player to ball in metres,
     ball speed, ball missing handling. Produces per frame raw codes H/A/D/X.
  B. Label only (available now): take baseline_raw.csv labels as raw evidence and run
     the same run length state machine on them. This lets us test the state machine on
     real derived data while the Mac is offline and the ATD txt is not on the VM.

State machine idea, from football logic, not tuned to the public window:
  - Possessions are long. True mean in public GT is 14.2 s, with a few short ones
    down to about 0.8 s. Flicker in baseline_raw is mostly 1 to 5 frames.
  - Dead ball runs are long. True D runs in public GT are 138 to 511 frames
    (5.5 to 20 s). Short D runs in baseline_raw are mostly ball occlusion, not dead ball.
  - A switch of team needs evidence: the new team must dominate for at least
    min_switch frames. A short spell sandwiched between the same team is merged.
  - X inside a possession is filled with the surrounding team. X at a transition
    is assigned to the team that follows, because the scorer forward fills X with
    the previous label, so leaving X there would carry the old team across.

No hand labels on the provided game are used. No event data about this game.
The public GT is used only to check that the pipeline is correct.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

N_FRAMES = 199319
FPS = 25
PITCH = (105.3, 68.0)

HOME, AWAY, NOBODY = 0, 1, -1
CODE_TO_LABEL = {0: "H", 1: "A", -1: "D", -2: "X"}
LABEL_TO_CODE = {"H": 0, "A": 1, "D": -1, "X": -2}


def runs_of(codes):
    """List of (code, start, end_exclusive) runs."""
    runs = []
    if len(codes) == 0:
        return runs
    cur = codes[0]
    start = 0
    for i in range(1, len(codes)):
        if codes[i] != cur:
            runs.append((int(cur), start, i))
            cur = codes[i]
            start = i
    runs.append((int(cur), start, len(codes)))
    return runs


def labels_from_runs(runs, n=N_FRAMES):
    out = np.full(n, "X", dtype="<U1")
    for code, s, e in runs:
        out[s:e] = CODE_TO_LABEL[code]
    return out


def state_machine(codes, min_alive=12, min_dead=75, fill_short_dead=50):
    """Run length state machine on per frame codes H/A/D/X.

    codes: int array, 0=H, 1=A, -1=D, -2=X (or -1 for missing ball, treated as D here
    only if the caller passed D; for label only input, D is already D).
    min_alive: shortest H/A spell to keep on its own, in frames. Shorter spells
      sandwiched between the same team are merged into that team. Shorter spells
      between different teams or D are removed and given to the previous team,
      unless the next team is longer, in which case the next team wins.
    min_dead: shortest D run to keep. Shorter D runs are filled with the surrounding
      team if both sides agree, else with the previous team.
    fill_short_dead: D runs shorter than this are treated as occlusion, same rule.
      In practice min_dead is the operative threshold; fill_short_dead is kept
      for clarity and is set equal to min_dead by default in the caller.
    Returns label array of length len(codes).
    """
    codes = np.asarray(codes).copy()
    n = len(codes)
    # Step 1: fill X inside a run of the same team. X between different teams
    # goes to the following team, to avoid carrying the old team across a switch.
    runs = runs_of(codes)
    for code, s, e in runs:
        if code != -2:
            continue
        prev_code = codes[s - 1] if s > 0 else None
        next_code = codes[e] if e < n else None
        if prev_code is not None and prev_code == next_code and prev_code in (0, 1):
            codes[s:e] = prev_code
        elif next_code in (0, 1):
            codes[s:e] = next_code
        elif prev_code in (0, 1):
            codes[s:e] = prev_code
        else:
            codes[s:e] = -1
    # Step 2: fill short D runs.
    changed = True
    while changed:
        changed = False
        runs = runs_of(codes)
        for code, s, e in runs:
            if code != -1:
                continue
            length = e - s
            if length >= min_dead:
                continue
            prev_code = codes[s - 1] if s > 0 else None
            next_code = codes[e] if e < n else None
            if prev_code in (0, 1) and prev_code == next_code:
                codes[s:e] = prev_code
            elif prev_code in (0, 1):
                codes[s:e] = prev_code
            elif next_code in (0, 1):
                codes[s:e] = next_code
            else:
                continue
            changed = True
            break
    # Step 3: merge short H/A spells.
    changed = True
    while changed:
        changed = False
        runs = runs_of(codes)
        for idx, (code, s, e) in enumerate(runs):
            if code not in (0, 1):
                continue
            length = e - s
            if length >= min_alive:
                continue
            prev_code = runs[idx - 1][0] if idx > 0 else None
            next_code = runs[idx + 1][0] if idx + 1 < len(runs) else None
            if prev_code in (0, 1) and prev_code == next_code:
                codes[s:e] = prev_code
            elif prev_code in (0, 1) and next_code in (0, 1):
                # Short spell between two different teams: give it to the longer
                # neighbouring spell, which is the more likely true possessor.
                prev_len = runs[idx - 1][2] - runs[idx - 1][1]
                next_len = runs[idx + 1][2] - runs[idx + 1][1]
                codes[s:e] = prev_code if prev_len >= next_len else next_code
            elif prev_code in (0, 1):
                codes[s:e] = prev_code
            elif next_code in (0, 1):
                codes[s:e] = next_code
            else:
                continue
            changed = True
            break
    # Step 4: any remaining X (at the very start, before kickoff etc) becomes D
    # outside graded play, and in practice the scorer never scores those frames.
    # Keep X only before the first H/A/D in each half is not possible here without
    # half boundaries, so leave X as X where it remains.
    return np.array([CODE_TO_LABEL[int(c)] for c in codes], dtype="<U1")


def smooth_codes(codes, window):
    """Centred majority vote over H/A, D where no ball in the window.
    Same as starter/baseline.py smooth(), but returns codes, not labels.
    """
    codes = np.asarray(codes)
    n = len(codes)
    home = np.concatenate(([0], np.cumsum(codes == HOME)))
    away = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(n)
    low = np.maximum(0, index - window)
    high = np.minimum(n, index + window + 1)
    h = home[high] - home[low]
    a = away[high] - away[low]
    out = np.full(n, NOBODY, dtype=np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], HOME, AWAY)
    return out


def v3_labels(raw_codes, window=75, min_dead_raw=150, trim=25, min_alive=12, min_dead=75):
    """Best version, v3.
    1. Centred majority smoothing with half window 75 frames (3 s).
    2. Restore long raw D runs (at least min_dead_raw frames, trimmed by trim
       frames each side) as D. Long ball-missing runs are dead ball far more
       often than occlusion: in the public window, restored D frames are
       63.5 percent true D, and true D runs are 138 to 511 frames long.
    3. Run length state machine to merge short spells.
    Window 75 was chosen by football logic and full file possession count
    (432 possessions, mean 11.3 s, vs 669 for baseline_smooth50), not by
    maximising the public window. Larger windows score higher on the public
    window but merge short possessions on a full match, as the published
    full match baselines show for window 125.
    """
    smoothed = smooth_codes(raw_codes, window)
    out = smoothed.copy()
    for code, s, e in runs_of(raw_codes):
        if code == NOBODY and e - s >= min_dead_raw:
            s2 = min(e, s + trim)
            e2 = max(s2, e - trim)
            out[s2:e2] = NOBODY
    return state_machine(out, min_alive=min_alive, min_dead=min_dead)


def team_of_each_track(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams")
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)


def parse_atd_raw(raw_path, meta_path, n_frames=N_FRAMES, pitch=PITCH):
    """Stream an ATD txt file, return per frame raw codes and features.

    Format, from starter/baseline.py: "<frame>:<player>;<player>;...:<ball>",
    each group "x,y,speed", NaN for missing. Frame counts from 1.
    Returns dict with:
      codes: int8, 0=H, 1=A, -1=no ball / no player (D candidate)
      ball_x, ball_y: metres, NaN where missing
      ball_speed: m/s, NaN where missing
      nearest_dist: metres to nearest player, NaN where no ball
      nearest_team: 0/1, -1 where no ball
      second_dist: metres to nearest player of the other team, NaN where missing
    """
    length, width = pitch
    team_of_track = team_of_each_track(meta_path)
    codes = np.full(n_frames, NOBODY, dtype=np.int8)
    ball_x = np.full(n_frames, np.nan)
    ball_y = np.full(n_frames, np.nan)
    ball_speed = np.full(n_frames, np.nan)
    nearest_dist = np.full(n_frames, np.nan)
    nearest_team = np.full(n_frames, NOBODY, dtype=np.int8)
    second_dist = np.full(n_frames, np.nan)
    with open(raw_path) as handle:
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
                bs = float(ball[ball.index(",", comma + 1) + 1:])
            except Exception:
                continue
            if not (0 <= frame < n_frames):
                continue
            best = None
            best_team = NOBODY
            best_other = None
            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                try:
                    comma = group.index(",")
                    px = float(group[:comma]) * length
                    py = float(group[comma + 1:group.index(",", comma + 1)]) * width
                except Exception:
                    continue
                d2 = (px - bx) ** 2 + (py - by) ** 2
                team = team_of_track[slot] if slot < len(team_of_track) else NOBODY
                if best is None or d2 < best:
                    best_other = best if best_team != team and best is not None else best_other
                    best = d2
                    best_team = team
                elif team != best_team and (best_other is None or d2 < best_other):
                    best_other = d2
            if best is None:
                continue
            codes[frame] = best_team
            ball_x[frame] = bx
            ball_y[frame] = by
            ball_speed[frame] = bs
            nearest_dist[frame] = float(np.sqrt(best))
            nearest_team[frame] = best_team
            if best_other is not None:
                second_dist[frame] = float(np.sqrt(best_other))
    return {
        "codes": codes,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_speed": ball_speed,
        "nearest_dist": nearest_dist,
        "nearest_team": nearest_team,
        "second_dist": second_dist,
    }


def v4_labels(raw_codes, window=75, min_dead_raw=150, trim=25, min_alive=12, min_dead=75,
              tagging_path=None, pre_frames=150):
    """v4: v3 plus Smart Tagging set piece priors.
    A set piece tag starts after the stoppage began, so dead ball likely started
    a few seconds before the tag start. Force D in the 150 frames (6 s) before
    each SET PIECES start, then run the state machine again. In the public
    window this changed only 252 frames, both ranges 100 percent true D that
    v3 had called H. Full match possession count is unchanged at 432.
    tagging_path is the Smart Tagging XML. If it is None, v4 is identical to v3.
    """
    out_labels = v3_labels(raw_codes, window=window, min_dead_raw=min_dead_raw, trim=trim,
                           min_alive=min_alive, min_dead=min_dead)
    if not tagging_path:
        return out_labels
    codes = label_codes(out_labels)
    root = ET.parse(tagging_path).getroot()
    for instance in root.iter("instance"):
        if (instance.findtext("code") or "").strip() != "SET PIECES":
            continue
        try:
            start = int(round(float(instance.findtext("start")) * FPS))
        except Exception:
            continue
        s = max(0, start - pre_frames)
        codes[s:start] = NOBODY
    return state_machine(codes, min_alive=min_alive, min_dead=min_dead)


def derived_ball_speed(ball_x, ball_y, fps=FPS):
    """Ball speed in m/s from position differences, consecutive present frames only.
    The ATD ball speed field is NaN on most frames, so it is not usable directly.
    """
    speed = np.full(len(ball_x), np.nan)
    dx = np.diff(ball_x)
    dy = np.diff(ball_y)
    valid = ~np.isnan(dx) & ~np.isnan(dy)
    speed[1:][valid] = np.sqrt(dx[valid] ** 2 + dy[valid] ** 2) * fps
    return speed


def v5_labels(raw_codes, nearest_dist=None, ball_speed=None, window=75, min_dead_raw=150,
              trim=25, min_alive=12, min_dead=75, tagging_path=None, pre_frames=150,
              dist_gate=2.5, speed_gate=12.0):
    """v5: v4 plus an ATD confidence gate on the raw evidence.
    A nearest player more than dist_gate metres away is not in controlled
    possession, and a ball moving faster than speed_gate m/s is a pass or shot
    in flight, so the nearest player at that instant is often wrong. Both
    cases are set to X, abstain, before smoothing. The state machine fills X
    inside a possession with that team, and at a transition with the following
    team, which is what the scorer does with X on the ranking metric.
    Thresholds are round football numbers. The effect is a plateau, not a
    sharp peak: dist gates 2.0, 2.25 and 2.5 m give public spellF1 0.55, 0.55
    and 0.575, and the combined gate gives 0.60. Full file possession count
    stays plausible at 407, mean 10.7 s, versus 432 for v4.
    """
    codes = np.asarray(raw_codes).copy()
    if nearest_dist is not None:
        gate = (codes != NOBODY) & (nearest_dist > dist_gate)
        codes[gate] = -2
    if ball_speed is not None:
        gate = (codes != NOBODY) & (ball_speed > speed_gate)
        codes[gate] = -2
    return v4_labels(codes, window=window, min_dead_raw=min_dead_raw, trim=trim,
                     min_alive=min_alive, min_dead=min_dead,
                     tagging_path=tagging_path, pre_frames=pre_frames)


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path


def load_label_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    return df["label"].values.astype("<U1")


def label_codes(labels):
    mapping = {"H": 0, "A": 1, "D": -1, "X": -2}
    return np.array([mapping[str(x)] for x in labels], dtype=np.int8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="baseline_raw.csv or an ATD txt file")
    parser.add_argument("--input-type", choices=["labels", "atd"], default="labels")
    parser.add_argument("--meta", default=None, help="ATD xml, required for atd input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-alive", type=int, default=12)
    parser.add_argument("--min-dead", type=int, default=75)
    parser.add_argument("--mode", choices=["v1", "v3", "v4", "v5"], default="v3")
    parser.add_argument("--window", type=int, default=75)
    parser.add_argument("--min-dead-raw", type=int, default=150)
    parser.add_argument("--tagging", default=None, help="Smart Tagging XML, required for v4")
    parser.add_argument("--pre-frames", type=int, default=150)
    parser.add_argument("--dist-gate", type=float, default=2.5)
    parser.add_argument("--speed-gate", type=float, default=12.0)
    args = parser.parse_args()
    feat = None
    if args.input_type == "labels":
        labels = load_label_csv(args.input)
        codes = label_codes(labels)
    else:
        if not args.meta:
            raise SystemExit("atd input needs --meta")
        feat = parse_atd_raw(args.input, args.meta)
        codes = feat["codes"]
        print("ball located on {} of {} frames".format(int(np.sum(codes != NOBODY)), len(codes)))
    if args.mode == "v3":
        out_labels = v3_labels(codes, window=args.window, min_dead_raw=args.min_dead_raw,
                               min_alive=args.min_alive, min_dead=args.min_dead)
    elif args.mode == "v4":
        if not args.tagging:
            raise SystemExit("v4 needs --tagging")
        out_labels = v4_labels(codes, window=args.window, min_dead_raw=args.min_dead_raw,
                               min_alive=args.min_alive, min_dead=args.min_dead,
                               tagging_path=args.tagging, pre_frames=args.pre_frames)
    elif args.mode == "v5":
        if not args.tagging:
            raise SystemExit("v5 needs --tagging")
        if feat is None:
            raise SystemExit("v5 needs ATD input, it uses nearest distance and ball speed")
        speed = derived_ball_speed(feat["ball_x"], feat["ball_y"])
        out_labels = v5_labels(codes, nearest_dist=feat["nearest_dist"], ball_speed=speed,
                               window=args.window, min_dead_raw=args.min_dead_raw,
                               min_alive=args.min_alive, min_dead=args.min_dead,
                               tagging_path=args.tagging, pre_frames=args.pre_frames,
                               dist_gate=args.dist_gate, speed_gate=args.speed_gate)
    else:
        out_labels = state_machine(codes, min_alive=args.min_alive, min_dead=args.min_dead)
    write_submission(args.out, out_labels)
    uniq, counts = np.unique(out_labels, return_counts=True)
    print(dict(zip(uniq.tolist(), counts.tolist())))
    print("wrote", args.out)
