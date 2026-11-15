# Push checklist, Metrica More than a Hack 2027 Ball Possession

Prepared, NOT pushed. GitHub access was 401, parent agent is asking
Monish to reconnect it. Do not push until access is fixed and
Monish confirms the repo name.

Directory: `~/workspace/mtah2027-public/`
Total staged files below: 21 files, 11,729,858 bytes (about 11.7 MB).
This checklist file itself is also staged, for a total of 22 files.

## Files to push

| file | size (bytes) | md5 |
| --- | --- | --- |
| `.gitignore` | 360 | `874abdc8a711c2f709fc217bf7520ccf` |
| `LICENSE` | 1,082 | `f0d0d79102b4d71b6b727ab27e85e0f3` |
| `README.md` | 8,594 | `ce9c59d28dfaefbcd6744310540ecea9` |
| `RESULTS.md` | 16,238 | `cef0460de1b4d3adfad25fd330d790b3` |
| `TECHNICAL_DOC.md` | 10,069 | `7be50eeac7c028e626c9c4a066165fde` |
| `atd_diag.py` | 2,838 | `43cab0fbe8dd779a86ab51bd64440fdc` |
| `atd_stats.py` | 4,029 | `587b8de1ed31d8a878dfb7facc5ebbaf` |
| `notes/repo_exploration.md` | 42,227 | `bc85bf1d3e605813e09ff76e2034d9f7` |
| `pipeline.py` | 18,936 | `4a5e9d81d9368422140c135d53215f64` |
| `submissions/v3_w75_d150.csv` | 1,682,773 | `8fd72798b072e547a7b3f7a73176b9bf` |
| `submissions/v4_pre150.csv` | 1,682,773 | `df4dfd3c8308d6e86d3547c1997acc8f` |
| `submissions/v5_pre150.csv` | 1,682,773 | `fe6a8fe544ba8480cbfa7b4cf3f5e6d1` |
| `synthetic.py` | 2,939 | `12ac8acf17bd49501d736c8d7704d36d` |
| `v5_final.py` | 1,320 | `63df09dde1e56963cc1b04210cfa60b2` |
| `v6_features.py` | 2,811 | `6329ab0c808b4544004c6a0a1dab1e28` |
| `v6_test.py` | 5,268 | `c51eab713a9cd1d4e68f3d085c6bd0c3` |
| `v6_test2.py` | 4,455 | `81fb04056581a8ddcb6e0c99434d3d71` |
| `yolo/ball_test_300.csv` | 4,662 | `2cb86727172d059a50d77875f0982fe7` |
| `yolo/yolo_ball_tactical.py` | 3,240 | `b31127678d0187d00f99cd8e7a03f0b2` |
| `yolo/yolo_ball_test.py` | 2,675 | `1365df1ee95454272b6f20a6a70de91c` |
| `yolo/yolov8n.pt` | 6,549,796 | `95a2449609c73cd69a072b09daaff0cc` |

## Verification done

- `git init` in `mtah2027-public/`, `git add -A`, `git status` shows
  all 21 files above staged, plus this checklist.
- Submission md5s match the source workspace exactly:
  v3 8fd72798b072e547a7b3f7a73176b9bf, v4 df4dfd3c8308d6e86d3547c1997acc8f,
  v5 fe6a8fe544ba8480cbfa7b4cf3f5e6d1. Each CSV is 199,320 lines,
  header plus 199,319 rows.
- No file over 10 MB. Largest is `yolo/yolov8n.pt`, 6,549,796 bytes,
  a public COCO model weight, allowed.
- Clean venv test: fresh venv with numpy 2.5.4 and pandas 3.1.1 only,
  `pipeline.py --input baseline_raw.csv --input-type labels --out
  v3_test.csv --mode v3 --window 75` reproduced v3 exactly, md5
  8fd72798b072e547a7b3f7a73176b9bf. v5 needs the gated ATD txt, which
  is not in this repo, so v5 was not re-run in the clean venv.
  Pipeline needs numpy and pandas only for the label path, plus
  ultralytics, torch, and opencv only for the YOLO scripts.
- Forbidden pattern grep, exit 1, no hits, for the patterns in the
  task brief: personal identifiers, credentials, Drive IDs and links,
  personal email addresses, university email addresses, match file
  hash and size, and match data filenames. Sanitised copies only:
  local absolute paths in the source scripts and docs were replaced
  with relative paths, and the sensitive values above were removed.
  The only remaining match name string is the public ground truth
  filename shipped in the official public repo, not match data.

## Excluded, hard

- `data/` entirely: ATD txt, homography CSVs, Smart Tagging XML,
  ATD XMLs, videos.
- `.venv/`, `venv/`, `__pycache__/`, `*.npz`, `*.npy`,
  `synthetic_gt.csv`, `test_*/`, `test_results*/`,
  `atd_derived.npz`, `atd_features.npz`, `v6_features.npz`.
- `repo/`, the cloned Metrica repo, linked in README only.
- `yolo/football_match.mp4`, undergrad clip, 9.6 MB, not needed.
- `v5_gate.csv`, test copy of v5, only `v5_pre150.csv` is kept.
- `v5_test*.py`, not in the include list.

## Scores, public 5 minute window only

- v5_pre150, best: macroF1 0.7223, spellF1 0.6000
- v4_pre150: macroF1 0.7041, spellF1 0.4824
- v3_w75_d150: macroF1 0.6308, spellF1 0.4353
- v6: none beat v5, best weighted smoothing spellF1 0.5467
- Full match floor: baseline_smooth50 spellF1 0.2433

## Do NOT do yet

- Do NOT create a GitHub repo, do NOT push, do NOT call the GitHub
  API with POST. Access was 401. Parent agent is asking Monish to
  reconnect it first.
