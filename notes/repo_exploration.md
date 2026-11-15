# Metrica Sports More than a Hack 2027 Phase 1: repo exploration

Exploration only. No tuning, no pipeline build, no hand labels, no training on the public ground truth. Written 2026-11-14 from the public repo at `./repo` (clone already present, not re-cloned). Plain punctuation throughout, no em dashes.

## 0. Environment note

- System Python is 3.12.3. Its system pandas is broken against its numpy (`ValueError: numpy.dtype size changed`, binary incompatibility), and system `pip install -r requirements.txt` is blocked by PEP 668 (needs `--break-system-packages`).
- Fix used: a venv at `./venv` with `pip install numpy pandas`, giving numpy 2.5.4 and pandas 3.1.1. All scores below were run with that venv.
- Command run: `cd ./repo && ./venv/bin/python evaluation/run_evaluation.py` (once with `--no-detail`, once full). Source: `evaluation/run_evaluation.py`, `requirements.txt`.
- Side effect of running the scorer: it rewrites `evaluation/scoreboard.csv` and `evaluation/reports/<name>.json` plus `evaluation/reports/<name>_transitions.csv` for each scored file. That is normal scorer behaviour, not a pipeline output.
- Mac is offline. No device calls were made and no Mac files were accessed. Only one real match file exists on the VM: `data/tactical_atd.xml` (59 KB). No ATD txt, no homography CSV, no smart tagging XML, no video is present on the VM, so the ATD, homography and smart tagging schemas below are taken from the repo docs and loader code, not verified against real data files, except where the XML section says verified.

## 1. Baseline scoreboard, public 5-minute ground truth

Run 2026-11-14. Scorer header: `scoreboard gt/BAR-ATH_possession_gt_public.csv 199319 frames 7500 graded 21 gt segments`. Ground truth has 17 possessions of mean 14.2 s. Ranked on spellF1. `transF1` and `lat_f` below are at tau 25 frames (1 s). `poss` is predicted possession count, `dur` is mean predicted possession duration in seconds. Source: `evaluation/run_evaluation.py` output and `evaluation/scoreboard.csv`.

| baseline | macroF1 | spellF1 | poss | dur | transF1@25 | lat_f | shareE | frag |
|---|---|---|---|---|---|---|---|---|
| baseline_smooth125 | 0.5812 | 0.4242 | 16 | 18.5 | 0.1667 | -7.7 | 0.052 | 0.81 |
| baseline_smooth50 | 0.6207 | 0.3256 | 26 | 10.8 | 0.0816 | +4.0 | 0.048 | 1.43 |
| baseline_smooth25 | 0.6200 | 0.2868 | 36 | 7.6 | 0.2034 | +8.2 | 0.056 | 1.90 |
| baseline_raw | 0.5972 | 0.0378 | 131 | 2.0 | 0.1410 | -3.1 | 0.046 | 6.52 |

Per-class F1 on the public window (from the full scorer detail output). Source: `evaluation/run_evaluation.py`.

| baseline | F1 H | F1 A | F1 D | accuracy | kappa |
|---|---|---|---|---|---|
| baseline_smooth125 | 0.7864 | 0.8108 | 0.1464 | 0.7275 | 0.5355 |
| baseline_smooth50 | 0.7781 | 0.7615 | 0.3225 | 0.7132 | 0.5202 |
| baseline_smooth25 | 0.7620 | 0.7123 | 0.3857 | 0.6912 | 0.4878 |
| baseline_raw | 0.7290 | 0.6430 | 0.4197 | 0.6512 | 0.4282 |

Spell F1 by IoU threshold, public window. Source: `evaluation/reports/baseline_*.json` detail output.

| baseline | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
|---|---|---|---|---|---|
| baseline_smooth125 | 0.6061 | 0.6061 | 0.4242 | 0.2424 | 0.2424 |
| baseline_smooth50 | 0.5116 | 0.4651 | 0.3256 | 0.2326 | 0.0930 |
| baseline_smooth25 | 0.4528 | 0.3774 | 0.3019 | 0.2264 | 0.0755 |
| baseline_raw | 0.0946 | 0.0541 | 0.0270 | 0.0135 | 0.0000 |

Label counts in the four shipped baseline files (all 199319 rows, frames 0 to 199318, zero X). Source: counted from `evaluation/results/baseline_*.csv`.

| baseline | H | A | D | X |
|---|---|---|---|---|
| baseline_raw | 63964 | 49524 | 85831 | 0 |
| baseline_smooth25 | 69446 | 51303 | 78570 | 0 |
| baseline_smooth50 | 72957 | 54842 | 71520 | 0 |
| baseline_smooth125 | 82758 | 62124 | 54437 | 0 |

What the baselines are: nearest tracked player to the ball, team of that player, D where the ball is not visible, then a centred majority vote with half-window 0, 25, 50 or 125 frames. Ties go to home. Source: `starter/baseline.py`.

## 2. Full-match floor, confirmation

The full-match floor quoted in `EVALUATION.md`, `README.md` and the `starter/baseline.py` docstring is confirmed as stated in all three places. These are full-match numbers, not the public-window numbers in section 1. Do not mix the two tables.

| baseline | macroF1 (full match) | spellF1 (full match) |
|---|---|---|
| baseline_raw | 0.6532 | 0.0486 |
| baseline_smooth25 | 0.6752 | 0.1996 |
| baseline_smooth50 | 0.6724 | 0.2433 |
| baseline_smooth125 | 0.6356 | 0.2242 |

Published floor: `baseline_smooth50` spellF1 0.2433. Beat it. Source: `EVALUATION.md` section "The four baselines", `README.md`, `starter/baseline.py` docstring.

Warning from `EVALUATION.md` and `README.md`: on the public window smooth125 looks best (spellF1 0.4242) but on the full match it is worse than smooth50 (0.2242 vs 0.2433). Tuning on 17 possessions picks the wrong window. One missed possession moves public spellF1 by roughly six points. Noise bands: macroF1 differences below about 0.01 are noise, spellF1 differences below about 0.04 are noise on the full match. The 0.04 band does not apply to the 5-minute window; use `--bootstrap 1000` there. Source: `EVALUATION.md`.

## 3. Public ground truth stats

Files: `evaluation/gt/BAR-ATH_possession_gt_public.csv` (199319 data rows plus header, 199320 lines total) and `evaluation/gt/BAR-ATH_possession_gt_public_meta.json`. Counts below were recomputed from the CSV and agree with the meta JSON.

- Graded window: video frames 11185 to 18684 inclusive, 7500 frames, first 5 minutes of play. Everything outside the window is X and is never scored. Verified: outside the window the label set is exactly {X}.
- Window label counts: H 3454, A 2577, D 1469. Whole-file counts: H 3454, A 2577, D 1469, X 191819.
- Segments in the window: 21 (17 possessions plus 4 D segments). Possessions: 17, mean duration 14.191 s (meta JSON says 14.191, scorer prints 14.2). Home share of alive time in the window: 0.5727.
- First window rows: frame 11185 label A. Last window rows: frame 18684 label H.
- Full-match graded frames (private GT): 158857. Public window is only a correctness check, not training or tuning data. Source: `DATA.md`, `evaluation/gt/BAR-ATH_possession_gt_public_meta.json`, `EVALUATION.md`, `RULES.md`.

All 21 segments in the window, in order (start frame, label, length in frames, duration in seconds):

| start | label | frames | seconds |
|---|---|---|---|
| 11185 | A | 166 | 6.64 |
| 11351 | H | 202 | 8.08 |
| 11553 | A | 71 | 2.84 |
| 11624 | H | 1041 | 41.64 |
| 12665 | A | 186 | 7.44 |
| 12851 | D | 138 | 5.52 |
| 12989 | H | 715 | 28.60 |
| 13704 | A | 81 | 3.24 |
| 13785 | H | 1061 | 42.44 |
| 14846 | A | 500 | 20.00 |
| 15346 | H | 19 | 0.76 |
| 15365 | A | 401 | 16.04 |
| 15766 | H | 33 | 1.32 |
| 15799 | A | 90 | 3.60 |
| 15889 | D | 511 | 20.44 |
| 16400 | A | 893 | 35.72 |
| 17293 | H | 87 | 3.48 |
| 17380 | D | 458 | 18.32 |
| 17838 | A | 189 | 7.56 |
| 18027 | D | 362 | 14.48 |
| 18389 | H | 296 | 11.84 |

Possession durations in the window range from 0.76 s to 42.44 s. D segments range from 5.52 s to 20.44 s.

## 4. Match, halves and frame alignment

- Match: FC Barcelona (home, H) vs Athletic Club (away, A), LaLiga EA Sports 2024-25, matchday 2. Source: `DATA.md`, `games.py`.
- Tactical video: 199319 frames, 25 fps, 1920x1080. Broadcast video: 164488 frames, own timeline, cuts and replays, not synchronised to tactical. A submission with about 164k rows indexed the broadcast video and is rejected by the row-count check. Source: `DATA.md`, `games.py`.
- Submission row i is frame i of the tactical video, i from 0 to 199318. Source: `DATA.md`, `games.py`, `SUBMISSION.md`.
- Graded halves, in video frames. Source: `DATA.md`, `games.py` function `half_video_blocks`.

| block | first frame | last frame | frames | graded |
|---|---|---|---|---|
| before the match | 0 | 11184 | 11185 | no |
| 1st half | 11185 | 90651 | 79467 | yes |
| halftime | 90652 | 110807 | 20156 | no |
| 2nd half | 110808 | 190197 | 79390 | yes |
| after the final whistle | 190198 | 199318 | 9121 | no |

- Total graded: 158857 frames. Total ungraded: 40462 frames. Whatever is written on an ungraded frame is ignored.
- `games.py` per-half offsets for the separate tracking counter: half 1 offset 1702057, half 2 offset 1702058, so `id_frame = video_frame + offset`. IdFrame spans: half 1 (1713242, 1792708), half 2 (1812866, 1892255). Second-half kickoff video frame is 113742, which is inside the second graded block, not at its start. First-half kickoff video frame is 11185. Source: `games.py`, `starter/load_atd.py`, `starter/load_smart_tagging.py`.
- `games.py` pitch: (105.3, 68.0) metres, measured from corrected tracking. ATD metadata pitch: 105 x 68 m. The 30 cm length difference matters only for metre distances. Source: `games.py`, `DATA.md`, `starter/utils.py`.

## 5. ATD schema (tracking)

Sources: `DATA.md`, `starter/load_atd.py`, `starter/baseline.py`, `starter/02_tracking_atd.ipynb`, and the verified XML in section 8.

### 5.1 Files

- Tactical: `data/tactical/tactical_atd.xml` (metadata) and `data/tactical/tactical_atd.txt` (raw tracking). Broadcast: same stem with `broadcast`, 164488 records. Paths are built by `games.asset_paths()`, never hard-coded. Source: `games.py`, `DATA.md`.
- Format: Metrica FIFA/EPTS export. `kloppy` reads the pair directly: `metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])`. Source: `DATA.md`, `starter/load_atd.py`.

### 5.2 Raw txt line format (exact)

One line per frame, 199319 lines tactical. Format, from `starter/baseline.py` and `starter/load_atd.py`:

```
<frame_id>:<player>;<player>;...:<ball>
```

- Field separator is `:` and splitting on `:` gives exactly three parts: frame id, players group, ball group.
- Player groups are separated by `;`. There is one group per track slot, in slot order (46 groups tactical, 51 broadcast).
- Inside a group, fields are separated by `,` and are `x,y,speed`. Speed is in m/s.
- A missing observation is the literal string `NaN`. In `starter/baseline.py` the test is only the first character: `MISSING = "N"`, so a group starting with `N` is treated as missing.
- The ball group is also `x,y,speed` or `NaN`. In `starter/baseline.py` only `ball_x` and `ball_y` are parsed; ball speed is ignored (parsed up to the second comma).
- `frame_id` counts from 1 to 199319. Alignment: `video_frame = frame_id - 1`, and inversely `frame_id = video_frame + 1`. That is the whole alignment for the ATD. Source: `starter/load_atd.py` functions `video_frame_of` and `frame_id_of`.
- `DataFormatSpecification` in the verified XML confirms: `startFrame 1`, `endFrame 199319`, separator `:`. Split registers: `;` then `,`. Source: verified XML, section 8.

### 5.3 kloppy table (as `dataset.to_df()` would read it)

- One row per frame: `frame_id`, `ball_x`, `ball_y`, then for each track slot `N_x`, `N_y`, `N_s`, for example `1_x`, `1_y`, `1_s` up to `46_x`, `46_y`, `46_s` tactical.
- Coordinates are normalised to 0-1, not metres. Multiply by pitch to get metres: x by length, y by width. The baseline scales both axes to metres before comparing distances, because the pitch is not square. Source: `DATA.md`, `starter/load_atd.py`, `starter/02_tracking_atd.ipynb`, `starter/baseline.py` function `nearest_player_labels`.
- Missing observations are NaN, for players and ball alike. Source: `DATA.md`.

### 5.4 Track slots and teams

- Tactical: 46 track slots, named `Track 1` to `Track 46`. Broadcast: 51 slots. There are 22 players on the pitch, so slots are tracklets, not people. Slots appear, vanish and swap identity as the tracker loses and re-acquires people. `ShirtNumber` is the slot index, not a shirt number and not a person. No player identities are provided. Analysis has to be collective, not per-player. Source: `DATA.md`, `starter/load_atd.py`, `starter/baseline.py` function `team_of_each_track`.
- Team of a slot comes from the XML `Player` element `teamId`. The first `Team` in the file is the home side (FIFA/EPTS convention, also what kloppy reports as `metadata.teams[0]`). For this match that is Barcelona. Source: `starter/baseline.py` function `team_of_each_track`.
- Verified mapping from the real XML (section 8): Track 1 to Track 24 are Barcelona (H), Track 25 to Track 46 are Atletic Club (A). Track 24 is the Barcelona goalkeeper, Track 46 is the Atletic Club goalkeeper. All other slots have `position_type Unknown`, `position_index -1`, `position_x 0.0`, `position_y 0.0` in the metadata, which carry no usable information.
- Mean tracked objects during play is about 20.5, occasionally over 22 when the tracker duplicates someone. Source: `DATA.md`.

### 5.5 Ball visibility

- Ball present on 56.9 percent of all tactical frames (113488 of 199319) and 59.3 percent of graded frames (94214 of 158857). So the ball is missing on 43 percent of all frames and 41 percent of graded frames. Broadcast ball visible on 50.5 percent of frames. Source: `DATA.md`.
- Any approach assuming a ball position on every frame will fail. The baseline returns no answer (code -1, NOBODY) where the ball is missing, and its smoother then writes D there. Source: `starter/baseline.py`.

## 6. Homography schema

Sources: `DATA.md`, `starter/load_homography.py`.

- Files: `data/tactical/tactical_homography.csv` (199319 rows) and `data/broadcast/broadcast_homography.csv` (164488 rows). One row per video frame of that recording, `frame` counting from 0. For tactical, row i is the homography for submission row i.
- Exact header: `frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid`
- Columns 1 to 9 are the 3x3 matrix in row-major order, so `row[1:10].reshape(3, 3)` gives the matrix. Loader constant: `MATRIX_COLUMNS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]`.
- Direction: the matrix maps normalised pitch coordinates (0-1) to image pixels, the same 0-1 convention as the ATD, so an ATD position goes straight in with no conversion. Do not multiply by pitch size before projecting. Inverse (`unproject`) maps pixels back to normalised pitch coordinates.
- Example row from `DATA.md`: `11185,1822.663,-511.2005,44.48168,1.698323,169.2971,255.7303,-0.01907188,-0.5269586,1,0.2910854,1`
- `valid`: check before using a matrix. Where it is 0 the nine matrix columns are NaN, because no pitch transform exists for that frame. Tactical is valid on 100 percent of frames. Broadcast is valid on 65.8 percent of frames, in 122 stretches, longest over a minute and a half (replays, close-ups, crowd shots).
- `reprojection_error`: how far the fitted matrix misses by, lower is better, no upper bound, broadcast reaches 304. Tactical median is 0.24 during play and about 1.5 across halftime, when the camera is not pointed at the pitch. `valid` says whether a matrix exists at all, `reprojection_error` says how much to trust the one you have.
- Loader return: `load()` returns `(matrices, reprojection_error, valid)` where matrices is (n, 3, 3) float64 with NaN wherever valid is False, and valid is boolean. `project(matrix, x, y)` returns pixels; NaN in gives NaN out.

## 7. Smart Tagging schema

Sources: `DATA.md`, `starter/load_smart_tagging.py`, `starter/01_smart_tagging.ipynb`. No smart tagging file exists on the VM, so counts below are as documented, not re-verified.

- File: `data/tactical/tactical_smart_tagging.xml` only (broadcast has none). SportsCode XML. 638 instances. Kloppy: `sportscode.load(asset_paths()["smart_tagging"]).to_df()`. Stdlib loader: `starter/load_smart_tagging.py` function `load()`.
- Each instance is an interval with a code and a set of tags. XML fields per `instance`: `start`, `end`, `code`, and one `label` per tag group, each label having `group` and `text`.
- `load()` returns a list of dicts in file order with keys `code`, `start_s`, `end_s`, `start_frame`, `end_frame`, `tags`. `tags` is a dict from group name to text; if a group appears more than once, last one wins.
- Time base, verified in the loader: `start` and `end` are seconds from the first frame of the tactical video, so `video_frame = round(seconds * 25)`. `end_frame` is half-open on the right (first frame after the phase). The check `check_time_base()` verifies both `Kick Off (Start)` tags contain the real kickoff frames 11185 (1st half) and 113742 (2nd half). A set piece tag marks a window of a few seconds around the restart, opening about 3.5 s early, not the restart instant.
- Kloppy columns: `timestamp`, `end_timestamp` (pandas timedeltas), `code`, and one column per tag group. Tag columns are title case: the column is `Team`, not `team`. Codes are upper case, tag values are title case. Match case-insensitively or results are silently empty.
- Code counts (from `DATA.md`):

| code | n | code | n |
|---|---|---|---|
| PLAYERS IN THE BOX | 97 | CREATING CHANCES | 30 |
| BALL IN FINAL THIRD | 93 | DEFENDING IN DEFENSIVE THIRD | 30 |
| BALL IN THE BOX | 72 | LONG BALL | 29 |
| PROGRESSION | 66 | ATTACKING TRANSITION | 24 |
| DEFENDING IN MIDDLE THIRD | 66 | DEFENSIVE TRANSITION | 24 |
| SET PIECES | 59 | BUILD UP | 22 |
| | | DEFENDING IN ATTACKING THIRD | 22 |
| | | GOALS | 4 |

- Tag groups and values (from `DATA.md`):
  - `Team`: `Barcelona` (314), `Atletic Club` (288), `N/A` (36). Every throw-in is tagged N/A.
  - `Half`: `1st Half`, `2nd Half`, strings not numbers.
  - `Type`: `Throw-in` (26), `Corner Kick` (12), `Goal Kick` (10), `Free Kick` (5), `Kick Off (After Goal)` (4), `Kick Off (Start)` (2). Only present for set pieces.
  - `Side`: `Central`, `Left`, `Right`.
  - `Direction of ball entry`: `Horizontal`, `Vertical`, `Diagonal`.
  - `Max Players in the box`: `1` to `6`, and `7+`, a string not an integer.
- Caveats: it is inferred automatically, not ground truth, so it has false positives and false negatives. Intervals are phases, not possessions. A Build Up can span one possession, stop short of one, or cover two. Do not copy an interval across as a label. Every phase carries a team tag and every set piece is a restart at the end of a dead-ball stretch, so it is a prior on both halves of the label, not an answer.

## 8. Verified real match file: tactical ATD XML

File: `data/tactical_atd.xml`, 59 KB (58K on disk listing). Parsed with `xml.etree.ElementTree`. Source for every fact in this section is that file itself.

- Root: `main`, namespace `FIFADataTransferFormatEPTSNamespace/FIFADataTransferFormatEPTS`. `FileDate 2026-08-31T06:36:55.545Z`. `FileName 2024-2025_LALIGAEASPORTS_J2_BAR-ATH_TAC_v1_FifaData`. `TrackingType Optical`. `ProviderName Metrica Sports`. `FrameRate 25.0`.
- `FieldSize`: `Width 105`, `Height 68`. That is the ATD metadata pitch, 105 x 68 m, distinct from `games.py` pitch (105.3, 68.0).
- Teams, in file order (first is home):
  - `MET_user_team_o4Hossvb8M`, Name `Barcelona`, attack direction first half `left_to_right`.
  - `MET_user_team_x7C4FFX2HF`, Name `Atletic Club`, attack direction first half `right_to_left`.
- Players: 46 `Player` elements. Each has `id` (1 to 46), `Name` (`Track 1` to `Track 46`), `ShirtNumber` equal to the id, and `teamId` equal to one of the two team ids above.
  - Track 1 to Track 24: `teamId MET_user_team_o4Hossvb8M`, Barcelona, home, H.
  - Track 25 to Track 46: `teamId MET_user_team_x7C4FFX2HF`, Atletic Club, away, A.
  - Track 24 is the Barcelona goalkeeper (`position_type Goalkeeper`, `position_index 0`, `position_x 0.0`, `position_y 0.5`). Track 46 is the Atletic Club goalkeeper (same values). All other 44 slots have `position_type Unknown`, `position_index -1`, `position_x 0.0`, `position_y 0.0`.
- Provider global parameters: `first_half_start 0`, `first_half_end 199319`, `second_half_start` empty, `second_half_end` empty, first and second extra half start/end empty, `data_type atd`, `tracking_source Metrica Sports`, `version 1`. Caveat: these XML half values span the whole file and do not split the halves; do not use them as half boundaries. Use `games.py` half blocks in section 4 instead.
- `Session id session1`, `SessionType Match`, `Start 2026-08-31T06:36:55.545Z`. `Score` has `idLocalTeam MET_user_team_o4Hossvb8M` and `idVisitingTeam MET_user_team_x7C4FFX2HF`, with empty local and visiting scores.
- `DataFormatSpecification`: `startFrame 1`, `endFrame 199319`, separator `:`. `SplitRegister` separators: `;` for groups, `,` inside groups.
- Channels: 138 `PlayerChannel` elements, 3 per player (x, y, s), ids like `player1_x`, `player1_y`, `player1_speed`, with `channelId` x, y or s and `playerId` 1 to 46. `BallChannelRef`: 3 entries, `channelId` x, y, s. `Devices`: `device1` (Device 1). `Sensors`: `position`, `speed`.
- This XML is metadata only. It contains no per-frame positions. The per-frame data lives in the txt file, which is not on the VM.

## 9. Metric logic (skim of `evaluation/possession_metrics.py`, 723 lines)

Source file is the specification. If `EVALUATION.md` and this file ever disagree, the file is right. Constants: `LABELS = ("H", "A", "D", "X")`, codes H 0, A 1, D 2, X 3. `SCORED = ("H", "A", "D")`. `TAUS = (12, 25)` frames. `SPELL_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)`, headline IoU 0.5. `FPS = 25`. `SPELL_NOISE_BAND = 0.04`. `BOOTSTRAP_N = 0` by default, `BOOTSTRAP_SEED = 0`, percentiles (5, 95).

- `macroF1`: unweighted mean of per-class F1 over H, A, D, over every graded frame. Frame-exact. A constant guess still scores something, so the usable range is roughly 0.16 to 1.0. Source: `macro_f1`, `per_class_prf`, `EVALUATION.md`.
- `spellF1`: the ranking metric. A possession is a maximal run of H or of A; D runs are stoppages, not possessions. Predicted possessions are matched one-to-one against true ones by temporal IoU, greedy in descending IoU order, same team only, and the F1 of that matching is averaged over IoU 0.3, 0.4, 0.5, 0.6 and 0.7. Range is a true 0 to 1; a constant guess scores exactly 0. Perfect scores 1.0. Perfect but 1 frame late still scores 1.0. Perfect but 1 s late scores 0.6539. Source: `spell_scores`, `match_spells`, `spell_overlaps`, `possession_spells`, `EVALUATION.md`.
- X handling, the key asymmetry:
  - `macroF1` charges every X on a graded frame as a miss, exactly like a wrong answer. The scored confusion is the 3x4 slice (GT rows H/A/D only, all four prediction columns), so a predicted X on a graded frame counts as a false negative, while a GT X row can never manufacture a false positive.
  - `spellF1` never sees X. Before any temporal metric runs, `fill_abstentions` forward-fills each X with the last non-X label in the same half (valid run), back-filling at the head of a half with the first known label. So on the ranking metric X means repeat my previous answer. Going quiet inside a possession is free on spellF1 and safer than a guess that might split the possession. At a transition it is the opposite: guess, because the fill carries the previous answer across the hand-over. A run the submission abstained on entirely is left alone. Source: `fill_abstentions`, `valid_runs`, `evaluate`, `EVALUATION.md`, `CHALLENGE.md`.
  - `validate_submission.py` function `describe` counts spells differently: it uses the raw codes, so X is a gap there and `H X X H` counts as two possessions, while the scorer fills the gap and counts one. Do not compare those two counts directly in files that contain X.
- Valid runs: `valid_runs` derives half-open spans where GT is not X, from the data, never hard-coded. For this match that is (11185, 90652) and (110808, 190198). Every temporal computation runs inside one run and never across the halftime hole. Source: `valid_runs`, `segments`, `transitions`.
- Transitions: `transitions` returns (frame, from_code, to_code) where frame is the first frame of the new label. `match_transitions` matches predicted to GT transitions one-to-one within plus or minus tol frames, matched on destination label by default (`typed="to"`). So `D -> H` and `D -> A` at the same frame are different events, but the origin label does not have to agree. Tolerances reported are 12 and 25 frames (about 0.5 s and 1.0 s). Latency is pred frame minus GT frame. `n_ambiguous` counts GT transitions with more than one feasible prediction; when it is not 0, latency figures are indicative only. Source: `match_transitions`, `transition_scores`.
- D handling: D is a scored class for macroF1 and for alive/dead F1, but D runs are never possessions. `alive_dead_f1` treats ALIVE (H or A) as the positive class. `team_accuracy` returns strict accuracy over all GT-alive frames and both-alive accuracy over frames where GT and prediction agree the ball is alive. `possession_share` is home share of alive time, each side over its own alive frames. `fragmentation` is n_pred_segments divided by n_gt_segments; a flickering submission can hold a respectable macroF1 while producing many times too many segments. On the public window n_gt_segments is 21. Source: `alive_dead_f1`, `team_accuracy`, `possession_share`, `fragmentation`, `possession_stats`.
- `evaluate(gt_codes, pred_codes, taus=TAUS, bootstrap_n=BOOTSTRAP_N, typed="to")` is the only function `run_evaluation.py` needs. It touches the full arrays twice: once for the confusion matrix, once for the run-length encoding. It returns macro_f1, spell_f1_mean, spell_f1 per IoU, n_gt_spells, n_pred_spells, spell_count_error, mean and GT mean spell duration, accuracy, kappa, alive_dead_f1, team accuracies, shares, fragmentation, n_graded, n_abstain_on_graded, bootstrap intervals, confusion, per_class, and transitions per tau.

## 10. Validation rules for a submission CSV

Sources: `SUBMISSION.md`, `evaluation/validate_submission.py`, `evaluation/possession_metrics.py` functions `read_label_csv` and `validate_submission`, `evaluation/run_evaluation.py`.

- Header is exactly `frame,label`. Header case and surrounding whitespace are tolerated; anything else is not. UTF-8, newline or CRLF, no index column.
- Exactly 199319 data rows, one per frame of the tactical video. `frame` is 0 to 199318, in order, integer dtype, contiguous, duplicate-free. One comparison in the code covers all of that.
- `label` is one of H, A, D, X. Values are stripped and upper-cased before checking, so `h` and ` H ` pass, but anything outside the four labels fails. No blank cells: an empty field is rejected as a mistake. Write X to abstain.
- A file that fails validation is skipped, not fatal, at scoring time. It appears at the bottom of the board with a one-line reason, and the scorer exit code is 1 if anything was skipped. A broken ground truth is fatal instead.
- Always run `python evaluation/validate_submission.py <file>.csv` on the exact file to be submitted, before every submission. It prints label counts, possession count and mean duration on success, and it checks format only, not correctness.
- Submission deliverables (from `SUBMISSION.md`): the possession CSV, a public code repository that regenerates the exact same file from the provided data, and a short technical document (approach, assets used and dropped, training data with source and labeller, possession rules implemented, limitations, reproduction). Deadline: November 17, 2026. Name the CSV and the document after the team. One response per team via the Google Form; the last response before the deadline is scored. There is no live leaderboard.
- Rules that bind the pipeline (from `RULES.md`): labels must be produced by an algorithm. No hand labels on the provided game, in any amount and for any purpose, including training, validation, correcting tracking, or correcting the output at transitions. Hand labels on other games are fine. No private ground truth and no human-produced data about this game from any source (event data, corrected tracking, possession data). The public 5 minutes are for checking pipeline correctness only, not training or tuning. Improving or replacing the provided data with your own algorithmic detection or tracking is legitimate. Do not redistribute the match data and do not publish derived data that would reconstruct it. Teams must have 3 to 5 members to be eligible for the Innovation Challenge and Final Pitch.

## 11. Function signatures and file paths the pipeline builder will need

### 11.1 Repo layout

- `games.py`: per-match constants, asset paths, frame alignment. Imports only the standard library.
- `evaluation/possession_metrics.py`: the metric library, the specification.
- `evaluation/run_evaluation.py`: the scorer, this exact file produces the official result.
- `evaluation/validate_submission.py`: the format checker.
- `evaluation/gt/BAR-ATH_possession_gt_public.csv` and `evaluation/gt/BAR-ATH_possession_gt_public_meta.json`: public 5-minute GT.
- `evaluation/results/baseline_raw.csv`, `baseline_smooth25.csv`, `baseline_smooth50.csv`, `baseline_smooth125.csv`: the four shipped baselines. Also `evaluation/results/README.md`, `evaluation/scoreboard.csv`, `evaluation/reports/`.
- `starter/baseline.py`: reference baseline, numpy only.
- `starter/load_atd.py`, `starter/load_smart_tagging.py`, `starter/load_homography.py`: loaders.
- `starter/utils.py`: pitch plotting for the notebooks.
- `starter/make_dummy_submission.py`: writes trivial constant submissions.
- `starter/01_smart_tagging.ipynb`, `starter/02_tracking_atd.ipynb`, `starter/03_first_submission.ipynb`: guided notebooks.
- `requirements.txt`: numpy and pandas only, the evaluation needs nothing else. `requirements-starter.txt`: adds `kloppy==3.18.0`, matplotlib, notebook. Python 3.8 or newer for evaluation, 3.9 plus recommended for notebooks.
- `data/`: unpack the bundle here, or set `MTAH_DATA_DIR`. Expected layout is `data/tactical/` and `data/broadcast/` with the stems in section 5.1. `data/` is gitignored. Source: `DATA.md`, `README.md`, `data/README.md`.

### 11.2 `games.py`

- `GAME = "BAR-ATH"`. `DATA_DIR` from env `MTAH_DATA_DIR` or `<repo>/data`. `GAMES["BAR-ATH"]` keys: `n_video_frames` 199319, `fps` 25, `offsets` {1: 1702057, 2: 1702058}, `halves` {1: (1713242, 1792708), 2: (1812866, 1892255)}, `kickoff` {"minute": 7, "second": 27, "frame": 11}, `kickoff_2nd` {"minute": 75, "second": 49, "frame": 16}, `pitch` (105.3, 68.0).
- `SOURCES`: tactical stem `tactical`, n_video_frames 199319, assets ("video", "atd_meta", "atd_raw", "homography", "smart_tagging"); broadcast stem `broadcast`, n_video_frames 164488, assets ("video", "atd_meta", "atd_raw", "homography"). `DEFAULT_SOURCE = "tactical"`. `SUFFIXES`: video `.mp4`, atd_meta `_atd.xml`, atd_raw `_atd.txt`, homography `_homography.csv`, smart_tagging `_smart_tagging.xml`. `ANCHOR_FPS = 25`.
- `asset_paths(game=GAME, data_dir=None, source=DEFAULT_SOURCE)` returns dict keyed by asset name. Paths are not checked for existence.
- `require(names, game=GAME, data_dir=None, source=DEFAULT_SOURCE)` returns the requested paths or raises a readable error naming every missing file.
- `source_frames(source=DEFAULT_SOURCE, game=GAME)` returns the frame count for a source.
- `video_path(game=GAME, data_dir=None, source=DEFAULT_SOURCE)`.
- `halves(game)` returns [(half, first_id_frame, last_id_frame)].
- `frame_offset(game, half, adjust=0)`.
- `half_video_blocks(game, adjust=0)` returns [(half, first_video_frame, last_video_frame)], for BAR-ATH [(1, 11185, 90651), (2, 110808, 190197)].
- `half_of_video_frame(game, video_frame, adjust=0)` returns a half or None.
- `video_to_id(game, video_frame, adjust=0)` and `id_to_video(game, id_frame, adjust=0)`.
- `anchor_video_frame(anchor)` computes minute, second, frame at 25 fps.

### 11.3 `starter/baseline.py`

- Constants: `VARIANTS = (("baseline_raw", 0), ("baseline_smooth25", 25), ("baseline_smooth50", 50), ("baseline_smooth125", 125))`. Half-widths are in frames; 25 frames is 1 s at 25 fps. `HOME, AWAY, NOBODY = 0, 1, -1`. `LABELS = np.array(["H", "A", "D"])`. `FIELD_SEP = ":"`, `GROUP_SEP = ";"`, `MISSING = "N"`. `RESULTS_DIR` is `<repo>/evaluation/results`.
- `team_of_each_track(meta_path)` returns an int8 array over track slots, 0 home, 1 away. Reads `Team` ids in file order and `Player` `teamId` values.
- `team_names(meta_path)` returns (home, away) club names.
- `nearest_player_labels(raw_path, team_of_track, n_frames, pitch)` returns per-frame team codes 0, 1 or -1. Distances are computed in metres after scaling normalised x by pitch length and y by pitch width. Writes `out[int(parts[0]) - 1] = best_team`.
- `smooth(codes, window)` returns label strings. Window 0 or less gives raw H/A/D with NOBODY as D. Otherwise a centred majority vote over plus or minus window frames, computed with cumulative sums; ties go to home; labels are D where no frame in the window has a ball.
- `write_submission(path, labels)` writes header `frame,label` then one `frame,label` row per frame, enumerated from 0.
- `build(game=GAME, data_dir=None)` runs `require(["atd_meta", "atd_raw"], ...)`, reads the ATD once, prints teams and ball coverage, and returns the per-frame codes.
- CLI: `python starter/baseline.py` writes the four variants; `--window` takes half-widths instead; `--out` sets the output directory (default `evaluation/results`).

### 11.4 `starter/load_atd.py`

- `atd_paths(game=GAME, data_dir=None, source=DEFAULT_SOURCE)` returns (meta_data, raw_data).
- `load(game=GAME, data_dir=None, source=DEFAULT_SOURCE)` returns a kloppy TrackingDataset via `metrica.load_tracking_epts(meta_data=meta, raw_data=raw)`. Needs kloppy; raises a readable ImportError otherwise.
- `video_frame_of(frame_id)` returns `int(frame_id) - 1`. `frame_id_of(video_frame)` returns `int(video_frame) + 1`.
- `check_alignment_raw(game=GAME, data_dir=None, source=DEFAULT_SOURCE)` streams the raw file and returns dict with `n_records`, `n_video_frames`, `frame_id_range`, `contiguous`, `first_video_frame`, `agrees`.
- `check_alignment(dataset, game=GAME, source=DEFAULT_SOURCE)` does the same check from a loaded kloppy dataset.
- Running it as a script loads the dataset, prints teams and pitch (ATD metadata vs `games.py`), prints the alignment check, and prints landmarks: video 11185 (1st half kickoff), 18684 (end of public window), 100000 (halftime), 113742 (2nd half kickoff), 190197 (last graded frame).

### 11.5 `starter/load_smart_tagging.py`

- `KICKOFF_CODE = "set pieces"`, `KICKOFF_TYPE = "kick off (start)"`, `KICKOFF_VIDEO_FRAMES = {"1st Half": 11185, "2nd Half": 113742}`.
- `pattern_path(game=GAME, data_dir=None)` returns the smart tagging path or a readable error.
- `load(game=GAME, data_dir=None, fps=None)` returns a list of dicts `{code, start_frame, end_frame, start_s, end_s, tags}` in file order. Frames are `int(round(seconds * fps))`, end frame half-open.
- `check_time_base(events)` returns `n_kickoff_tags`, per-half rows with `half`, `true_kickoff_frame`, `tagged` intervals and `contains_kickoff`, and `agrees`.
- Running it as a script prints the instance count, code counts, Team tag values, and the kickoff containment check.

### 11.6 `starter/load_homography.py`

- `MATRIX_COLUMNS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]`.
- `load(game=GAME, data_dir=None, source=DEFAULT_SOURCE)` returns `(matrices, reprojection_error, valid)`: matrices (n, 3, 3) float64, reprojection_error (n,), valid (n,) boolean. Raises if columns are missing.
- `project(matrix, x, y)` returns (pixel_x, pixel_y) from normalised pitch coordinates. Scalars or arrays.
- `unproject(matrix, pixel_x, pixel_y)` returns normalised pitch coordinates via the inverse matrix.
- `pitch_corners(matrix)` returns the four pitch corners in pixels, clockwise from (0, 0).
- `check_alignment(game=GAME, data_dir=None, source=DEFAULT_SOURCE)` returns `n_rows`, `n_video_frames`, `contiguous`, `agrees`.
- Running it as a script prints, for tactical and broadcast, row counts, valid and invalid counts and percentages, median and p99 reprojection error over valid frames, and the projected pitch centre and corners of the first valid frame.

### 11.7 `starter/utils.py` and `starter/make_dummy_submission.py`

- `utils.py`: `PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0`. `HOME_COLOUR, AWAY_COLOUR, BALL_COLOUR = "#d62728", "#1f77b4", "#ffffff"`. `draw_pitch(ax=None, length=PITCH_LENGTH, width=PITCH_WIDTH)` draws a pitch in metres. `plot_frame(dataset, frame_index, ax=None, length=..., width=...)` draws both teams and the ball for one frame; the title notes when the ball is not tracked. `animate_frames(dataset, start_frame, end_frame, length=..., width=...)` returns a matplotlib FuncAnimation. Matplotlib is imported inside the functions.
- `make_dummy_submission.py`: `DEFAULTS = (("dummy_dead", "D"), ("dummy_home", "H"), ("dummy_abstain", "X"))`. `write_constant(name, label, n_frames, results_dir)` writes a constant-label CSV. CLI: `python starter/make_dummy_submission.py`, or `--label A --name my_baseline` for one file. All three dummies score spellF1 0.0000.

### 11.8 `evaluation/possession_metrics.py`

- `read_label_csv(path)` returns the label column as an object array. Tolerates header case and whitespace; requires `frame` and `label` columns.
- `validate_submission(frame, n_expected)` returns None if usable, else a one-line reason string.
- `label_codes(frame_or_labels)` maps label strings to codes 0 to 3, accepting a DataFrame or a bare array.
- `confusion(gt_codes, pred_codes)`, `scored_confusion(cm)`, `per_class_prf(cm3, k)`, `macro_f1(cm3)`, `accuracy(cm3)`, `cohens_kappa(cm3)`, `alive_dead_f1(cm3)`, `team_accuracy(cm3)`, `possession_share(cm3)`.
- `valid_runs(gt_codes)` returns half-open [start, stop) spans where GT is not X.
- `fill_abstentions(pred_codes, runs)` returns pred codes with X forward-filled then back-filled at run heads, inside runs only.
- `segments(codes, runs)` returns [(start, stop, code)] constant-label runs inside each valid run.
- `transitions(codes, runs)` returns [(frame, from_code, to_code)].
- `match_transitions(gt_trans, pred_trans, tol, typed="to")` returns (pairs, missed, spurious). `transition_scores(gt_trans, pred_trans, tol, typed="to")` returns precision, recall, F1, counts, latencies and n_ambiguous.
- `fragmentation(gt_codes, pred_codes, runs)` returns (n_gt_segments, n_pred_segments, ratio).
- `possession_spells(codes, runs)` returns [(start, stop, code)] maximal H or A runs.
- `spell_overlaps(gt_spells, pred_spells)` returns [(iou, gt_index, pred_index)] same-team overlapping pairs, sorted by descending IoU. `matched_gt_flags(overlaps, n_gt, threshold)`, `match_spells(overlaps, n_gt, n_pred, threshold)`, `spell_scores(gt_codes, pred_codes, runs, ious=SPELL_IOUS)`.
- `possession_stats(codes, runs)` returns n_spells, mean_duration_s, home_share.
- `bootstrap_macro_f1(...)`, `bootstrap_spell_f1(...)` for block-bootstrap intervals over GT segments or possessions.
- `evaluate(gt_codes, pred_codes, taus=TAUS, bootstrap_n=BOOTSTRAP_N, typed="to")` returns the full result dict described in section 9.

### 11.9 `evaluation/run_evaluation.py` and `evaluation/validate_submission.py`

- `run_evaluation.py` CLI: `python evaluation/run_evaluation.py [--gt public|private|PATH] [--results DIR] [--out DIR] [--tau ...] [--bootstrap N] [--no-detail]`. Defaults: `--gt public`, `--results evaluation/results`, `--out evaluation/`. `GT_ALIASES`: public matches `*_possession_gt_public.csv`, private matches `*_possession_gt.csv`. `RANK_METRIC = "spell_f1_mean"`. Rank key: spellF1 desc, then macroF1 desc, then transition F1 at the loose tau desc, then mean absolute latency asc, then filename.
- `find_gt(gt_arg)`, `load_meta(gt_path)`, `score_submission(path, gt_codes, taus, bootstrap_n)`, `rank_key(entry, loose_tau)`, `print_scoreboard(entries, skipped, gt_path, meta, loose_tau)`, `scoreboard_row(rank, entry, taus)`, `write_transitions_csv(path, result, loose_tau)` (columns: status, gt_frame, pred_frame, latency, gt_from, gt_to, pred_from, pred_to).
- `validate_submission.py` CLI: `python evaluation/validate_submission.py <file>.csv [--rows 199319]`. `describe(codes, fps)` returns (n_spells, mean_duration) using raw codes, so X counts as a gap there, unlike the scorer.

### 11.10 Notebooks, in brief

- `01_smart_tagging.ipynb` (14 cells): loads via kloppy sportscode, shows code counts, Team values, converts timestamps to frames with `round(seconds * 25)`, checks both kickoff tags contain the real kickoffs, warns that Team is N/A for all throw-ins and that intervals are windows not instants.
- `02_tracking_atd.ipynb` (20 cells): loads via kloppy metrica with `limit=12000` for the demo, shows `dataset.to_df()` with `frame_id`, `ball_x`, `ball_y`, `1_x`, `1_y`, `1_s` and so on, checks alignment with `load_atd.check_alignment_raw()`, counts ball presence, prints ATD pitch vs `games.py` pitch, plots frame 11185 with `utils.plot_frame`, animates frames 11185 to 11435.
- `03_first_submission.ipynb` (22 cells): calls `baseline.build()`, counts ball coverage per minute, writes raw labels with `baseline.smooth(codes, 0)` and `baseline.write_submission`, validates, scores with `pm.evaluate`, then sweeps smoothing windows (0, 12, 25, 50, 75, 125, 200) and plots macroF1 vs spellF1. Its warning is the same as section 2: the window that wins on the public window is not the window that wins on the full match.

## 12. Gaps and things not verified

- No ATD txt file on the VM, so the raw line format, ball coverage percentages, 46 slot count in the txt, and kloppy `to_df()` column names are from docs and code, not from a real txt file. The XML side (teams, slots, goalkeepers, pitch, frame rate) is verified against the real file.
- No homography CSV and no smart tagging XML on the VM, so their row counts, code counts and valid percentages are as documented in `DATA.md` and the loaders, not re-measured.
- No video on the VM, so no frame was watched and no homography projection was tested on a real frame.
- The full-match baseline numbers in section 2 are quoted from `EVALUATION.md`, `README.md` and `starter/baseline.py`. They were not re-run, because the private full-match ground truth is not in the repo and cannot be re-run here. Only the public-window numbers in section 1 were actually executed.
- `evaluation/scoreboard.csv` and `evaluation/reports/` were regenerated by the scorer run in this task. They are scorer outputs, not new submissions.
