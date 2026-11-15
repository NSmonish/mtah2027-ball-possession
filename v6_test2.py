"""v6 extra combos, bootstrap 1000 for finalists.
Plain punctuation, no em dashes.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
sys.path.insert(0, "more-than-a-hack-2027/evaluation")
from pipeline import v4_labels, v5_labels, state_machine, runs_of, HOME, AWAY, NOBODY
from possession_metrics import evaluate, label_codes as pm_label_codes

d = np.load("./atd_features.npz")
g = np.load("./v6_features.npz")
der = np.load("./atd_derived.npz")
codes0 = d["codes"].copy()
dist = d["nearest_dist"]
speed = der["ball_speed_derived"]
ch3, ca3 = g["cnt_h3"], g["cnt_a3"]
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

def score_labels(labels, name, boot=1000):
    r = evaluate(gt_codes, pm_label_codes(np.array(labels)), bootstrap_n=boot)
    fp, fd, dshare = full_stats(np.array(labels))
    print(f"{name:44s} macro {r['macro_f1']:.4f} ({r['bootstrap_low']:.3f}-{r['bootstrap_high']:.3f}) spell {r['spell_f1_mean']:.4f} ({r['spell_bootstrap_low']:.3f}-{r['spell_bootstrap_high']:.3f}) possPub {r['n_pred_spells']} durPub {r['mean_spell_duration_s']:.1f}s Df1 {r['per_class']['D']['f1']:.3f} trans25 {r['transitions'][25]['f1']:.3f} | full poss {fp} dur {fd:.1f}s Dgraded {dshare:.3f}", flush=True)
    return r

def apply_tagging_pre150(labels):
    import xml.etree.ElementTree as ET
    codes2 = np.array([0 if x == "H" else 1 if x == "A" else -1 for x in labels], dtype=np.int8)
    root = ET.parse(TAG).getroot()
    for inst in root.iter("instance"):
        if (inst.findtext("code") or "").strip() != "SET PIECES":
            continue
        start = int(round(float(inst.findtext("start")) * 25))
        codes2[max(0, start - 150):start] = NOBODY
    return state_machine(codes2, min_alive=12, min_dead=75)

def weighted_smooth(codes, window=75, geo_bonus=0.0):
    n = len(codes)
    conf = np.where(dist < 1.5, 1.0, np.where(dist < 2.5, 0.5, 0.25))
    conf = np.where(np.isnan(conf), 0.0, conf)
    conf = np.where(speed > 12, 0.0, conf)
    # geometry bonus: 0.25 per extra player within 3m for that team, capped
    if geo_bonus:
        bh = np.clip(ch3 - ca3, 0, 2) * geo_bonus
        ba = np.clip(ca3 - ch3, 0, 2) * geo_bonus
        conf_h = conf + bh
        conf_a = conf + ba
    else:
        conf_h = conf_a = conf
    wh = np.where(codes == HOME, conf_h, 0.0)
    wa = np.where(codes == AWAY, conf_a, 0.0)
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

def weighted_v4(codes_in, window=75, geo_bonus=0.0):
    sm = weighted_smooth(codes_in, window=window, geo_bonus=geo_bonus)
    out = sm.copy()
    for code, s, e in runs_of(codes0):
        if code == NOBODY and e - s >= 150:
            s2 = min(e, s + 25)
            e2 = max(s2, e - 25)
            out[s2:e2] = NOBODY
    lab = state_machine(out, min_alive=12, min_dead=75)
    return apply_tagging_pre150(lab)

# finalist re-runs, bootstrap 1000
score_labels(weighted_v4(codes0, window=100), "v6a weighted w100 boot1000")
score_labels(weighted_v4(codes0, window=75), "v6a weighted w75 boot1000")
# weighted + v5 gate: gate first, then weighted smooth on gated codes
c = codes0.copy()
c[(codes0 != NOBODY) & ((dist > 2.5) | (speed > 12))] = -2
score_labels(weighted_v4(c, window=75), "v6e weighted w75 + v5 gate")
score_labels(weighted_v4(c, window=100), "v6e weighted w100 + v5 gate")
# geometry bonus in weighted smooth
score_labels(weighted_v4(codes0, window=75, geo_bonus=0.25), "v6f weighted w75 + geo bonus 0.25")
# control v5, bootstrap 1000
score_labels(v5_labels(codes0, nearest_dist=dist, ball_speed=speed, tagging_path=TAG, pre_frames=150, dist_gate=2.5, speed_gate=12.0), "v5 control boot1000")
