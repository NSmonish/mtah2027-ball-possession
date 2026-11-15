import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
sys.path.insert(0, "more-than-a-hack-2027/evaluation")
from pipeline import v4_labels
from possession_metrics import evaluate, label_codes as pm_label_codes
d = np.load("./atd_features.npz")
codes0 = d["codes"].copy(); dist=d["nearest_dist"]
der = np.load("./atd_derived.npz"); speed=der["ball_speed_derived"]
gt = pd.read_csv("evaluation/gt/BAR-ATH_possession_gt_public.csv")["label"].values
gt_codes = pm_label_codes(gt)
TAG="data/tactical_smart_tagging.xml"
def final(name, raw):
    lab=v4_labels(raw, tagging_path=TAG, pre_frames=150)
    r=evaluate(gt_codes, pm_label_codes(np.array(lab)), bootstrap_n=1000)
    print(f"{name}: macro {r['macro_f1']:.4f} ({r['bootstrap_low']:.3f}-{r['bootstrap_high']:.3f}) spell {r['spell_f1_mean']:.4f} ({r['spell_bootstrap_low']:.3f}-{r['spell_bootstrap_high']:.3f}) poss {r['n_pred_spells']} dur {r['mean_spell_duration_s']:.1f} transF1@25 {r['transitions'][25]['f1']:.3f} frag {r['fragmentation']:.2f}", flush=True)
    return lab
lab_v4 = final("v4 control", codes0)
c=codes0.copy(); c[(codes0!=-1)&((dist>2.5)|(speed>12))]=-2
lab_v5 = final("v5 gate dist>2.5/speed>12 X", c)
# save v5 candidate
from pipeline import write_submission
write_submission("submissions/v5_gate.csv", lab_v5)
print("wrote v5_gate.csv")
