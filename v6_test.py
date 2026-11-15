"""v6 candidates, tested the same way as v5.
Plain punctuation, no em dashes.
Screen with bootstrap 200, finalists re-run with bootstrap 1000.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
sys.path.insert(0, "more-than-a-hack-2027/evaluation")
from pipeline import v3_labels, v4_labels, v5_labels, state_machine, runs_of, label_codes, HOME, AWAY, NOBODY
from possession_metrics import evaluate, label_codes as pm_label_codes

d = np.load("./atd_features.npz")
g = np.load("./v6_features.npz")
der = np.load("./atd_derived.npz")
codes0 = d["codes"].copy()
dist = d["nearest_dist"]
speed = der["ball_speed_derived"]
ch3, ca3 = g["cnt_h3"], g["cnt_a3"]
ch2, ca2 = g["cnt_h2"], g["cnt_a2"]
dh, da = g["dist_h"], g["dist_a"]
gt = pd.read_csv("evaluation/gt/BAR-ATH_possession_gt_public.csv")["label"].values
gt_codes = pm_label_codes(gt)
TAG = "data/tactical_smart_tagging.xml"
graded = np.zeros(199319, bool)
graded[11185:90652] = True
graded[110808:190198] = True

def full_stats(labels):
    runs = runs_of(np.array([0 if x == "H" else 1 if x == "A" else -1 if x == "D" else -2 for x in labels]))
    poss = [(e - s) for c, s, e in runs if c in (0, 1)]
    return len(poss), float(np.mean(poss)) / 25 if poss else 0, float(np.mean(np.array(labels)[graded] == "D"))

def score_labels(labels, name, boot=200):
    r = evaluate(gt_codes, pm_label_codes(np.array(labels)), bootstrap_n=boot)
    fp, fd, dshare = full_stats(np.array(labels))
    print(f"{name:38s} macro {r['macro_f1']:.4f} spell {r['spell_f1_mean']:.4f} possPub {r['n_pred_spells']} durPub {r['mean_spell_duration_s']:.1f}s Df1 {r['per_class']['D']['f1']:.3f} trans25 {r['transitions'][25]['f1']:.3f} | full poss {fp} dur {fd:.1f}s Dgraded {dshare:.3f}", flush=True)
    return r

def weighted_smooth(codes, window=75):
    n = len(codes)
    conf = np.where(dist < 1.5, 1.0, np.where(dist < 2.5, 0.5, 0.25))
    conf = np.where(np.isnan(conf), 0.0, conf)
    conf = np.where(speed > 12, 0.0, conf)
    wh = np.where(codes == HOME, conf, 0.0)
    wa = np.where(codes == AWAY, conf, 0.0)
    ch = np.concatenate(([0], np.cumsum(wh)))
    ca = np.concatenate(([0], np.cumsum(wa)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - window)
    hi = np.minimum(n, idx + window + 1)
    h = ch[hi] - ch[lo]
    a = ca[hi] - ca[lo]
    out = np.full(n, NOBODY, dtype=np.int8)
    seen = (h + a) > 2.0
    out[seen] = np.where(h[seen] >= a[seen], HOME, AWAY)
    return out

def weighted_v4(window=75):
    sm = weighted_smooth(codes0, window=window)
    out = sm.copy()
    for code, s, e in runs_of(codes0):
        if code == NOBODY and e - s >= 150:
            s2 = min(e, s + 25)
            e2 = max(s2, e - 25)
            out[s2:e2] = NOBODY
    lab = state_machine(out, min_alive=12, min_dead=75)
    codes2 = np.array([0 if x == "H" else 1 if x == "A" else -1 for x in lab], dtype=np.int8)
    import xml.etree.ElementTree as ET
    root = ET.parse(TAG).getroot()
    for inst in root.iter("instance"):
        if (inst.findtext("code") or "").strip() != "SET PIECES":
            continue
        start = int(round(float(inst.findtext("start")) * 25))
        codes2[max(0, start - 150):start] = NOBODY
    return state_machine(codes2, min_alive=12, min_dead=75)

def v5_control():
    return v5_labels(codes0, nearest_dist=dist, ball_speed=speed, tagging_path=TAG, pre_frames=150, dist_gate=2.5, speed_gate=12.0)

print("--- v6a weighted smoothing ---")
for w in [50, 75, 100]:
    score_labels(weighted_v4(window=w), f"v6a weighted w{w}")

print("--- v6b team geometry raw ---")
count3 = np.where(ch3 > ca3, HOME, np.where(ca3 > ch3, AWAY, codes0)).astype(np.int8)
count3[codes0 == NOBODY] = NOBODY
score_labels(v4_labels(count3, tagging_path=TAG, pre_frames=150), "v6b count3 raw, v4")
count2 = np.where(ch2 > ca2, HOME, np.where(ca2 > ch2, AWAY, codes0)).astype(np.int8)
count2[codes0 == NOBODY] = NOBODY
score_labels(v4_labels(count2, tagging_path=TAG, pre_frames=150), "v6b count2 raw, v4")
# geometry as tie-break only when margin < 1m
margin = d["second_dist"] - dist
geo = codes0.copy()
tight = (margin < 1.0) & (codes0 != NOBODY)
geo[tight] = np.where(ch3[tight] > ca3[tight], HOME, np.where(ca3[tight] > ch3[tight], AWAY, codes0[tight]))
score_labels(v5_labels(geo, nearest_dist=dist, ball_speed=speed, tagging_path=TAG, pre_frames=150, dist_gate=2.5, speed_gate=12.0), "v6b margin<1 geometry tiebreak, v5")

print("--- v6c dead ball slow+far ---")
for thr_d, thr_s in [(3.0, 2.0), (2.5, 2.0), (3.0, 3.0)]:
    c = codes0.copy()
    rule = (codes0 != NOBODY) & (dist > thr_d) & (speed < thr_s)
    c[rule] = NOBODY
    score_labels(v4_labels(c, tagging_path=TAG, pre_frames=150), f"v6c D dist>{thr_d} speed<{thr_s}, v4")
    score_labels(v5_labels(c, nearest_dist=dist, ball_speed=speed, tagging_path=TAG, pre_frames=150, dist_gate=2.5, speed_gate=12.0), f"v6c D dist>{thr_d} speed<{thr_s}, v5")

print("--- v6d margin gate ---")
for thr in [0.5, 1.0]:
    c = codes0.copy()
    c[(codes0 != NOBODY) & (margin < thr)] = -2
    score_labels(v5_labels(c, nearest_dist=dist, ball_speed=speed, tagging_path=TAG, pre_frames=150, dist_gate=2.5, speed_gate=12.0), f"v6d margin<{thr} to X, v5")

print("--- control v5 ---")
score_labels(v5_control(), "v5 control")
