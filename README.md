# More than a Hack 2027, Phase 1: Ball Possession

Ball possession for one full LaLiga match, FC Barcelona vs Athletic Club.
For every frame of the tactical video, say who has the ball: H (home),
A (away), D (dead ball), or X (no answer).

Output is a CSV with 199,319 rows, one per tactical video frame, header
`frame,label`.

This repo is the submission code for that task. It is not the official
challenge repo. The official repo is:

https://github.com/metrica-sports/more-than-a-hack-2027

## Ranking

Ranking is on spellF1, how many real possessions are recovered as whole
possessions, matched by temporal IoU. macroF1 is the first tie break.
Per frame accuracy is not the ranking metric, and a system tuned for it
can score well per frame while producing far too many possessions.

The published floor to beat is baseline_smooth50, spellF1 0.2433 on the
full match. The public ground truth covers only the first 5 minutes of
play, 17 possessions, mean 14.2 seconds. Use it only to check that a
pipeline is correct, never to train or tune. Bootstrap intervals on 17
possessions are wide, about 0.31 for spellF1, so small public gains are
noise.

## Data, gated and NOT included

No match data is in this repo. The match assets are gated, released only
to enrolled participants who accepted the terms of use. Enrol at the
official challenge landing page, read and sign the terms, receive a
download link, and unpack the bundle into `data/`, or point
`MTAH_DATA_DIR` at it, following the official repo README and DATA.md.

Expected layout, from the official repo:

```
data/
  tactical/
    tactical.mp4
    tactical_atd.xml
    tactical_atd.txt
    tactical_homography.csv
    tactical_smart_tagging.xml
  broadcast/
    broadcast.mp4
    broadcast_atd.xml
    broadcast_atd.txt
    broadcast_homography.csv
```

Do not commit match data, derived frame level tracking exports, or
clipped video to a public repo. The `.gitignore` in this repo excludes
`data/`, videos, and intermediate feature files.

## Approach

Start from the reference baseline, nearest tracked player to the ball,
and fix its two weaknesses: it flickers, and it has no real dead ball
model. All versions live in `pipeline.py`.

- v3: centred majority smoothing, window 75 frames (3 seconds), restore
  long raw dead ball runs of at least 150 frames, trimmed 25 frames at
  each end, then a run length state machine to merge short spells.
- v4: v3 plus Smart Tagging set piece priors. Force D in the 150 frames
  (6 seconds) before each SET PIECES tag start, then run the state
  machine again. A hard D mask from the whole set piece interval was
  tested and rejected, set piece intervals are phase windows, not
  stoppages, and are only 56 percent true D in the public window.
- v5, best: v4 plus an ATD confidence gate. If the nearest player is
  more than 2.5 metres from the ball, or the derived ball speed is over
  12 m/s, set that frame to X, abstain. A player over 2.5 m away is not
  in controlled possession, and a fast ball is a pass or shot in flight.
  The ATD ball speed field is NaN on most frames, so ball speed is
  derived from ball position differences. The state machine fills X
  inside a possession with that team, and at a transition with the
  following team.
- v6, tested and rejected: distance weighted smoothing, team geometry
  (count of players within 2 to 3 m per team, not just nearest 1),
  margin gates, and dead ball rules using slow ball plus far nearest
  player. None beat v5. See RESULTS.md.

No model was trained. No hand labels on the provided game were used,
in any amount, for any purpose. No event data about this game, no
private ground truth, no human produced data about this game.

## Scores, public 5 minute window only

Official `run_evaluation.py`, bootstrap 1000 where noted.

| version | macroF1 | spellF1 | possessions | mean duration |
| --- | --- | --- | --- | --- |
| baseline_raw | 0.5972 | 0.0378 | 131 | 2.0 s |
| baseline_smooth25 | 0.6200 | 0.2868 | 36 | 7.6 s |
| baseline_smooth50 | 0.6207 | 0.3256 | 26 | 10.8 s |
| baseline_smooth125 | 0.5812 | 0.4242 | 16 | 18.5 s |
| v3_w75_d150 | 0.6308 | 0.4353 | 17 | 16.5 s |
| v4_pre150 | 0.7041 | 0.4824 | 17 | 15.9 s |
| v5_pre150, best | 0.7223 | 0.6000 | 15 | 18.0 s |
| v6 weighted w75 | 0.7126 | 0.5412 | 17 | - |
| v6 weighted w100 | 0.7264 | 0.5467 | 13 | 12.8 s |
| v6 team geometry | 0.7011 | 0.5111 | 19 | - |

Full file, v5: 407 possessions, mean 10.7 s, H 63879, A 44595,
D 90845, X 0. v4: 432 possessions, mean 11.0 s. Baseline_smooth50
full file: 669 possessions, mean 7.6 s.

Caveat: v5 merges 2 short public possessions that v4 kept, a 71 frame
A spell and a 19 frame H spell, and gains by cleaner boundaries on
longer spells. The choice rests on the plateau across distance
thresholds, on transition F1 (0.270 for v5 versus 0.154 for v4), and on
football logic, not on the public spellF1 number alone.

## Reproduction

Clone the official challenge repo as well as this one, and put the
gated data in `data/` as above. This repo needs numpy and pandas only
for the pipeline path. The evaluation needs numpy and pandas only.

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027.git
python -m venv .venv
.venv/bin/pip install numpy pandas
.venv/bin/python evaluation/run_evaluation.py
```

Reproduce the best submission, v5, from the ATD txt:

```bash
python pipeline.py --input data/tactical_atd.txt --input-type atd \
  --meta data/tactical_atd.xml \
  --tagging data/tactical_smart_tagging.xml \
  --out submissions/v5_pre150.csv --mode v5 --window 75 \
  --min-dead-raw 150 --min-alive 12 --min-dead 75 \
  --dist-gate 2.5 --speed-gate 12
python evaluation/validate_submission.py submissions/v5_pre150.csv
```

The label only path reproduces v3 and v4 from the shipped baseline,
without the ATD txt:

```bash
python pipeline.py --input evaluation/results/baseline_raw.csv \
  --input-type labels --out submissions/v3_w75_d150.csv --mode v3 --window 75
```

Check the file before submitting it, with the official validator in
the official repo. The submission CSV is 199,319 rows, frame 0 to
199318, header `frame,label`.

Experiment scripts in this repo, `v5_final.py`, `v6_test.py`,
`v6_test2.py`, `atd_stats.py`, `atd_diag.py`, `v6_features.py`, and
`synthetic.py`, are the scripts used to test v5 and v6. They expect
the official repo evaluation code and the gated data in `data/`, and
some expect intermediate feature files that are not included here.
`synthetic.py` generates synthetic ATD like data for testing the
pipeline without real match data.

## YOLO ball detection, setup only

YOLOv8n ball detection was set up and proven on a small undergrad
clip, not yet run on the tactical video. The model weight
`yolo/yolov8n.pt` is included, it is a public COCO model weight.

Install, in a venv:

```bash
.venv/bin/pip install ultralytics torch opencv-python
```

Test on a small clip first:

```bash
.venv/bin/python yolo/yolo_ball_test.py
```

That test, on the first 300 frames of a small clip, gave ball
detected on 117 of 300 frames, 39.0 percent, mean conf 0.472,
person on all 300 frames, 14.7 per frame, at 8.9 fps CPU.
Output format is in `yolo/ball_test_300.csv`.

For the tactical video, once it is downloaded and verified complete,
run only on ATD ball missing frames in the public window first:

```bash
.venv/bin/python yolo/yolo_ball_tactical.py --start 11185 --end 18684 --conf 0.25
```

Compare detected ball positions to the ATD ball on frames where ATD
does have the ball, to measure false positives, before integrating
anything into the possession pipeline. Ball missing runs are mostly
long dead ball, mean 461 frames, not short occlusions, so expected
gain is low compared with the download and inference cost. The full
video at 1920x1080 would take many hours at CPU speed, never run it
blindly.

## Files

- `pipeline.py`, the v3, v4, and v5 pipeline
- `RESULTS.md`, full results log, all versions tested
- `TECHNICAL_DOC.md`, short technical document
- `submissions/v5_pre150.csv`, best, `v4_pre150.csv`, `v3_w75_d150.csv`
- `yolo/yolo_ball_test.py`, `yolo/yolo_ball_tactical.py`,
  `yolo/ball_test_300.csv`, `yolo/yolov8n.pt`
- `notes/repo_exploration.md`, notes on the official repo structure
  and data formats, from the official docs and loaders

## Licence

MIT, see LICENSE.

## Citation

```bibtex
@misc{mtah2027,
  title  = {More than a Hack 2027: Phase 1 - Ball Possession},
  author = {Metrica Sports and FC Barcelona Innovation Hub},
  year   = {2027},
  url    = {https://github.com/metrica-sports/more-than-a-hack-2027}
}
```
