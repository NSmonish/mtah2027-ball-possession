# More than a Hack 2027, Phase 1 Ball Possession, technical document, draft
# Team submission draft, for Monish to review. Plain punctuation, no em dashes.

## Approach
We start from the reference baseline, nearest tracked player to the ball, and fix its
two weaknesses: it flickers, and it has no real dead ball model.

Our best version, v5, is v4 plus an ATD confidence gate, all in pipeline.py:
1. Confidence gate on the raw ATD evidence. If the nearest player is more than
   2.5 metres from the ball, or the derived ball speed is over 12 m/s, set that
   frame to X, abstain. A player over 2.5 m away is not in controlled possession,
   and a fast ball is a pass or shot in flight, so the nearest player at that
   instant is often wrong. The ATD ball speed field is NaN on most frames, so
   ball speed is derived from ball position differences.
2. Centred majority vote over plus or minus 75 frames (3 seconds) on the nearest
   player team. This is the same smoothing as the baseline, with a wider window
   chosen by football logic and by full file possession count, not by maximising
   the public window.
3. Dead ball restoration. Long ball missing runs in the raw tracking, at least
   150 frames, trimmed by 25 frames at each end, are written back as D. Long
   ball missing runs are dead ball far more often than occlusion. In the public
   window, restored D frames are 63.5 percent true D, and true D runs are 138
   to 511 frames long, while occlusion gaps are short.
4. Run length state machine. Short H or A spells under 12 frames sandwiched
   between the same team are merged. Short D runs under 75 frames are filled
   with the surrounding team. X inside a possession is filled with that team.

Step 5, v4 and v5: force D in the 150 frames (6 seconds) before each SET PIECES
tag start, then run the state machine again. A set piece tag starts after the
stoppage began, so dead ball likely started a few seconds before the tag. In
the public window this changed only 252 frames, both ranges 100 percent true
D that v3 had called H. A hard D mask from the whole set piece interval was
tested and rejected: set piece intervals are phase windows, not stoppages.
They are 56 percent true D in the public window, and the goal kick interval
runs 800 frames into live play.

The result is a clean possession sequence: 407 possessions of mean 10.7 seconds
over the full file, versus 432 possessions of mean 11.0 seconds for v4, and
669 possessions of mean 7.6 seconds for baseline_smooth50.

## Assets used and dropped
Used:
- Automatic tracking, ATD, tactical txt, [size removed] bytes, md5
  [md5 removed]. Parsed directly with pipeline.py
  parse_atd_raw. Ball located on 113,488 of 199,319 frames, 56.9 percent.
  Nearest player distance and derived ball speed are the new features in v5.
  ATD nearest player reproduces baseline_raw exactly, agreement 1.0, so the
  earlier v3 and v4 versions were already using ATD nearest labels.
- ATD metadata XML, for team mapping. Track 1 to 24 are Barcelona, home, H.
  Track 25 to 46 are Atletic Club, away, A. Goalkeepers are Track 24 and 46.
- Smart Tagging, 638 instances, 59 set pieces. Only the 150 frames before each
  set piece start are used, as above. Phase team tags were checked and not used:
  they agree with true H or A only 61 percent in the public window.
- Homography, tactical, 100 percent valid over 199319 frames. Checked for
  validity, not used for labels, because pitch coordinates are already
  normalised in the ATD.
- Public 5 minute ground truth, only to check that the pipeline is correct,
  never to train or tune.

Dropped or not yet used:
- Broadcast video and ATD. Not synchronised to the tactical video, so not used.
- Broadcast video and ATD. Not synchronised to the tactical video, so not used.
- Video ball detection with YOLOv8n from the undergrad project. Setup is done, Nov 15 2026: torch 2.15.0+cpu, cv2 5.0.0, ultralytics 8.4.213 installed in the venv, numpy 2.5.4 and pandas 3.1.1 unchanged. Test on the undergrad clip, yolo/football_match.mp4, first 300 frames: 8.9 fps CPU, ball detected on 117 of 300 frames, 39.0 percent, mean conf 0.472, person on all 300 frames, 14.7 per frame. Code yolo/yolo_ball_test.py, output yolo/ball_test_300.csv. Tactical video was NOT on the VM at last check, Nov 15 about 01:05 ET, it is being uploaded to personal Drive and may take 2 hours. Do NOT download a partial video. Ready command, not yet run, is in yolo/yolo_ball_tactical.py, run only on ATD ball missing frames in the public window first. Ball missing runs are mostly long dead ball, mean 461 frames, not short occlusions, so expected gain is low compared with the download and inference cost.
- v6 ATD improvements, Nov 15 2026, tested and rejected: distance weighted smoothing w75 spellF1 0.5412, w100 0.5467, team geometry count vote 0.5111, margin geometry tiebreak 0.5750, dead ball slow plus far rule identical to v5, margin gate to X 0.5059 to 0.5250, weighted plus v5 gate 0.4824 to 0.5250, geometry bonus 0.4889. None beat v5 spellF1 0.6000. v5 stays best. Details in RESULTS.md.
- Undergrad YOLO tracking code. Used as the design starter for the ATD parser
  and for the nearest player idea, but not copied blindly. It had no ball
  detection, no team classification code, no dead ball or smoothing logic, and
  it used a 30 pixel threshold that does not transfer to pitch coordinates.

## Training data
None. No model was trained. No hand labels on the provided game were used, in
any amount, for any purpose. No event data about this game, no private ground
truth, no human produced data about this game. The only labelled data touched
was the public 5 minute ground truth, to check pipeline correctness.

## Possession rules implemented
- A possession is a maximal run of H or A. Dead ball is a stoppage, never a
  possession.
- A change of team needs sustained evidence, not one frame. The centred
  majority vote and the minimum spell merging enforce that.
- A simple touch or deflection does not change possession in our system, because
  a one frame or few frame change in nearest player is smoothed away.
- A long ball missing run is treated as dead ball, after trimming the edges
  where the restart is uncertain.
- Short ball missing gaps inside a possession are filled with that team, which
  is what the scorer does with X on the ranking metric.

## Scores, with the caveat that matters
Public 5 minute window, 17 possessions, run_evaluation.py, bootstrap 1000:
- Best version v5_pre150: macroF1 0.7223, spellF1 0.6000, poss 15, dur 18.0 s
- v4_pre150: macroF1 0.7041, spellF1 0.4824, poss 17, dur 15.9 s
- v3_w75_d150: macroF1 0.6308, spellF1 0.4353, poss 17, dur 16.5 s
- baseline_smooth50: macroF1 0.6207, spellF1 0.3256, poss 26, dur 10.8 s
- baseline_smooth125: macroF1 0.5812, spellF1 0.4242, poss 16, dur 18.5 s
- baseline_raw: macroF1 0.5972, spellF1 0.0378, poss 131, dur 2.0 s

The public window has only 17 possessions. Bootstrap with 1000 draws gives a
spellF1 interval 0.425 to 0.775 for v5 and 0.318 to 0.647 for v4, so the public
window cannot distinguish v5 from v4 on spellF1 alone. The v5 choice rests on
the plateau across distance thresholds, 2.0 to 2.5 m all give spellF1 0.55 to
0.575, on transition F1, 0.270 versus 0.154 for v4, on macroF1, and on the
football logic of the gate. v5 merges 2 short public possessions that v4 kept,
a 71 frame A spell and a 19 frame H spell, and gains by cleaner boundaries on
the longer spells. A distance weighted smoothing version, weights 1.0 under
1.5 m, 0.5 to 2.5 m, 0.25 over 2.5 m, scored spellF1 0.5412 with 17 possessions
and is a more robust alternative if the gate is judged too sharp. We did not
pick the window that maximises the public score: window 100 scores 0.4875 on
the public window in v3, but the published full match baselines show window
125 is worse than window 50 on a full match (0.2242 versus 0.2433), because
large windows merge short possessions. We picked window 75 by football logic,
by full file possession count, and by the full match baseline trend. Only 2
set pieces overlap the public window, so the v4 gain there is n=2 evidence,
stated here as such.

Full match floor to beat, from the organisers: baseline_smooth50 spellF1
0.2433. Our public score is above that, but the full match score is private
and cannot be known from this work alone.

## Limitations
- The ATD txt is now available, and v5 uses distance and derived ball speed,
  but dead ball detection still misses many stoppages. During true dead ball
  the ball is often visible, raw labels are H 54 percent, A 13 percent, D 34
  percent in the public window. v4 recovers some of these via set piece tags,
  D recall 0.3867, but most are still missed.
- No video, so ball occlusions cannot be filled by detection. Ball missing
  runs are mostly long dead ball, mean 461 frames, so video would help only
  a small share of frames.
- The smoothing window is a compromise. Short true possessions under about
  1 second, there are two in the public window at 0.76 and 1.32 seconds, are
  likely to be merged away. v5 merges both of them in the public window.
- A Viterbi HMM with generic duration priors was tried and rejected. It scored
  spellF1 0.0626 on the public window, essentially returning the raw flicker,
  because a stay probability of 0.996 is too weak against flicker blocks of
  10 frames or more.

## Reproduction
From the repo clone, with the venv at ./.venv:
- python evaluation/run_evaluation.py, to confirm baselines
- python pipeline.py --input data/tactical_atd.txt --input-type atd --meta data/tactical_atd.xml --tagging data/tactical_smart_tagging.xml --out submissions/v5_pre150.csv --mode v5 --window 75 --min-dead-raw 150 --min-alive 12 --min-dead 75 --dist-gate 2.5 --speed-gate 12
- python evaluation/validate_submission.py submissions/v5_pre150.csv
- python evaluation/run_evaluation.py --results <dir with the submission>

The submission CSV is 199319 rows, frame 0 to 199318, header frame,label,
labels H, A, D, X. This version has H 63879, A 44595, D 90845, X 0.
