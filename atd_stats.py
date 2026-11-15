import sys
sys.path.insert(0, ".")
import numpy as np
from pipeline import parse_atd_raw, team_of_each_track

feat = parse_atd_raw("data/tactical_atd.txt",
                     "data/tactical_atd.xml")
codes = feat["codes"]
print("codes counts H/A/D:", {k:int(v) for k,v in zip(*np.unique(codes, return_counts=True))})
print("ball located frames:", int(np.sum(codes!=-1)), "of", len(codes), "=", float(np.mean(codes!=-1)))
print("nearest_dist: mean", float(np.nanmean(feat["nearest_dist"])), "median", float(np.nanmedian(feat["nearest_dist"])), "p90", float(np.nanpercentile(feat["nearest_dist"],90)), "p95", float(np.nanpercentile(feat["nearest_dist"],95)))
print("nearest_dist <1m:", float(np.nanmean(feat["nearest_dist"]<1.0)), "<1.5m:", float(np.nanmean(feat["nearest_dist"]<1.5)), "<2m:", float(np.nanmean(feat["nearest_dist"]<2.0)), "<3m:", float(np.nanmean(feat["nearest_dist"]<3.0)))
margin = feat["second_dist"]-feat["nearest_dist"]
print("margin mean", float(np.nanmean(margin)), "median", float(np.nanmedian(margin)), "margin<0.5m:", float(np.nanmean(margin<0.5)), "margin<1m:", float(np.nanmean(margin<1.0)))
print("ball_speed mean", float(np.nanmean(feat["ball_speed"])), "median", float(np.nanmedian(feat["ball_speed"])), "p90", float(np.nanpercentile(feat["ball_speed"],90)))
# graded frames: 11185-90651 and 110808-190197 (video frames, 0-based)
graded = np.zeros(len(codes), bool)
graded[11185:90652]=True
graded[110808:190198]=True
print("graded ball present:", float(np.mean(codes[graded]!=-1)), "n graded", int(graded.sum()))
print("public window 11185-18684 ball present:", float(np.mean(codes[11185:18685]!=-1)))
# ball missing run lengths
def runs(a):
    out=[]; cur=a[0]; s=0
    for i in range(1,len(a)):
        if a[i]!=cur:
            out.append((cur,s,i)); cur=a[i]; s=i
    out.append((cur,s,len(a))); return out
miss = codes==-1
mr = [e-s for c,s,e in runs(miss) if c]
print("ball missing runs:", len(mr), "mean len", float(np.mean(mr)), "median", float(np.median(mr)), "max", int(np.max(mr)), "runs >=150 frames:", int(np.sum(np.array(mr)>=150)), "runs 25-150:", int(np.sum((np.array(mr)>=25)&(np.array(mr)<150))))
# compare to baseline_raw
import pandas as pd
raw = pd.read_csv("evaluation/results/baseline_raw.csv")
print("baseline_raw counts:", raw["label"].value_counts().to_dict())
# map my codes to labels
lab = np.array(["H" if c==0 else "A" if c==1 else "D" for c in codes])
agree = np.mean(lab==raw["label"].values)
print("ATD nearest vs baseline_raw agreement all frames:", float(agree))
print("ATD nearest vs baseline_raw agreement ball-present frames:", float(np.mean(lab[codes!=-1]==raw["label"].values[codes!=-1])))
print("ATD nearest vs baseline_raw agreement public window:", float(np.mean(lab[11185:18685]==raw["label"].values[11185:18685])))
# public GT comparison for ATD nearest (raw, no smoothing)
gt = pd.read_csv("evaluation/gt/BAR-ATH_possession_gt_public.csv")
print("GT public counts:", gt["label"].value_counts().to_dict())
# GT is X outside window, graded 11185-18684
g = gt["label"].values[11185:18685]
a = lab[11185:18685]
print("ATD nearest public: H/A/D counts", {k:int(v) for k,v in zip(*np.unique(a, return_counts=True))})
print("ATD nearest public accuracy vs GT (H/A/D only):", float(np.mean(a==g)))
for true in ["H","A","D"]:
    m=g==true
    print(" true",true,"n",int(m.sum()),"pred H",float(np.mean(a[m]=="H")),"A",float(np.mean(a[m]=="A")),"D",float(np.mean(a[m]=="D")))
# ball speed in true D vs true H/A in public window
bs = feat["ball_speed"][11185:18685]
print("ball speed public: true H mean", float(np.nanmean(bs[g=="H"])), "true A", float(np.nanmean(bs[g=="A"])), "true D", float(np.nanmean(bs[g=="D"])))
print("ball present public by true label: H", float(np.mean(codes[11185:18685][g=="H"]!=-1)), "A", float(np.mean(codes[11185:18685][g=="A"]!=-1)), "D", float(np.mean(codes[11185:18685][g=="D"]!=-1)))
np.savez("./atd_features.npz", **feat)
print("saved atd_features.npz")
print("team_of_track:", team_of_each_track("data/tactical_atd.xml"))
