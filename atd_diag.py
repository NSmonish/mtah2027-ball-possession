import numpy as np, pandas as pd
d=np.load("./atd_features.npz")
bx,by=d["ball_x"],d["ball_y"]
codes=d["codes"]
gt=pd.read_csv("evaluation/gt/BAR-ATH_possession_gt_public.csv")["label"].values
s,e=11185,18685
g=gt[s:e]
# derived ball velocity from position diff, only consecutive present frames
vx=np.full(len(bx),np.nan); vy=np.full(len(bx),np.nan)
dx=np.diff(bx); dy=np.diff(by)
valid=~np.isnan(dx)&~np.isnan(dy)
vx[1:][valid]=dx[valid]*25; vy[1:][valid]=dy[valid]*25
speed=np.sqrt(vx**2+vy**2)
print("derived ball speed: non-nan", int(np.sum(~np.isnan(speed))), "mean", float(np.nanmean(speed)), "median", float(np.nanmedian(speed)))
for lab in ["H","A","D"]:
    m=g==lab
    print(lab, "n", int(m.sum()),
          "ballpresent", round(float(np.mean(codes[s:e][m]!=-1)),3),
          "nearest_dist med", round(float(np.nanmedian(d["nearest_dist"][s:e][m])),2),
          "nearest<1.5", round(float(np.nanmean(d["nearest_dist"][s:e][m]<1.5)),3),
          "nearest<2.5", round(float(np.nanmean(d["nearest_dist"][s:e][m]<2.5)),3),
          "nearest>4", round(float(np.nanmean(d["nearest_dist"][s:e][m]>4)),3),
          "margin med", round(float(np.nanmedian((d["second_dist"]-d["nearest_dist"])[s:e][m])),2),
          "margin<0.5", round(float(np.nanmean(((d["second_dist"]-d["nearest_dist"])[s:e][m])<0.5)),3),
          "ballspeed med", round(float(np.nanmedian(speed[s:e][m])),2),
          "ballspeed>8", round(float(np.nanmean(speed[s:e][m]>8)),3),
          "ballspeed<1", round(float(np.nanmean(speed[s:e][m]<1)),3))
# nearest team accuracy when nearest_dist is small vs large
a=d["nearest_team"]
for lo,hi in [(0,1),(1,1.5),(1.5,2.5),(2.5,4),(4,99)]:
    m=(d["nearest_dist"][s:e]>=lo)&(d["nearest_dist"][s:e]<hi)&(g!="D")
    if m.sum()>20:
        pred=np.where(a[s:e][m]==0,"H","A")
        print(f"dist {lo}-{hi}: n={int(m.sum())} nearest-team accuracy vs GT H/A = {float(np.mean(pred==g[m])):.3f}")
# D detection via nearest_dist: what fraction of true D has nearest>4 or ball missing?
print("true D: ball missing", round(float(np.mean(codes[s:e][g=="D"]==-1)),3), "nearest>4", round(float(np.nanmean(d["nearest_dist"][s:e][g=="D"]>4)),3))
print("true H: ball missing", round(float(np.mean(codes[s:e][g=="H"]==-1)),3), "nearest>4", round(float(np.nanmean(d["nearest_dist"][s:e][g=="H"]>4)),3))
print("true A: ball missing", round(float(np.mean(codes[s:e][g=="A"]==-1)),3), "nearest>4", round(float(np.nanmean(d["nearest_dist"][s:e][g=="A"]>4)),3))
# ball out of bounds
print("ball x out of [0,105]:", int(np.sum((bx<0)|(bx>105))), "y out of [0,68]:", int(np.sum((by<0)|(by>68))))
# ball speed derived full match
print("derived speed full: median", float(np.nanmedian(speed)), "p90", float(np.nanpercentile(speed,90)), ">10 m/s frac", float(np.nanmean(speed>10)))
np.savez("./atd_derived.npz", ball_speed_derived=speed)
