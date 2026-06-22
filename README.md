# Traffic Sign Sparse Localization and Direction

COMP5340 project. We reproduce the localization stage of AutoTS (Han et al.
2025) and reframe traffic-sign geo-localization as **sparse outlier recovery**
from noisy Vision-GPS observations. The compressed-sensing contribution lives in
`AutoTS/sparse_localization/`.

## Where to start

| You want | Open |
|---|---|
| The team handoff (TL;DR, talking points, honesty notes) | `2026_06_23_sparse_localization_readme.md` |
| The CS module reference (methods, commands) | `AutoTS/sparse_localization/README.md` |
| Filled result tables | `AutoTS/results/cs/report_tables.md` |
| Figures for slides | `AutoTS/results/cs/figures/` |
| Full proposal with math and derivations | `2026_06_18_compressed_sensing_autots_proposal_report_18.md` |

## Setup (CS module, no GPU)

```bash
git clone https://github.com/phungdo/traffic_sign_sparse_localization_direction.git
cd traffic_sign_sparse_localization_direction
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements_cs.txt
```

## Run it

All commands run from inside `AutoTS/`.

```bash
cd AutoTS
```

Experiment 5 needs no dataset (it generates synthetic data) and is the quickest
way to confirm the install works:

```bash
python -m sparse_localization.experiments.exp5_synthetic_phase
```

Experiments 1 to 4 use the KITTI-TS sign annotations, which are in the repo
(`AutoTS/KITTI-TS/train/sign_id_GT.json` and `test/sign_id_GT.json`). They run
out of the box using the thin-lens depth model:

```bash
python -m sparse_localization.run_all          # all five experiments
python -m sparse_localization.make_report      # rebuild tables + figures
```

Outputs land in `AutoTS/results/cs/` (CSVs, `report_tables.md`, `figures/`).

## A note on the depth file

The results committed under `AutoTS/results/cs/` were produced with the AutoTS
PlaneDepth depth maps (`AutoTS/data/img_depth.npz`, 3.5 GB). That file is too
large for GitHub and is not in the repo, so a fresh clone automatically falls
back to the thin-lens depth model and prints a notice. The methods, experiments,
and figures all work either way; only the exact numbers shift. To reproduce the
committed numbers exactly, obtain `img_depth.npz` from the project's data drive
and place it at `AutoTS/data/img_depth.npz`.

## What is not in the repo

To stay under GitHub size limits, these are excluded (see `.gitignore`):

- KITTI-TS frame images and the depth maps (`*.npz`, `AutoTS/data/`)
- model checkpoints (`*.pth`, `*.pt`)
- dataset and source archives (`*.zip`)
- the published paper PDF and the vendored Detectron2 source

The orientation and detector reproduction scripts (`AutoTS/*.py` outside
`sparse_localization`) need those checkpoints plus Detectron2 and PyTorch, set up
separately. See `reproduce_readme.md`. The CS module needs none of that.

## Integrity note

Read the "Data provenance and honesty" section of
`2026_06_23_sparse_localization_readme.md` before presenting. In short:
Experiment 5 is synthetic by design, the location points rest on a fixed camera
model and assumed sign heights, the diagrams are hand-drawn illustrations, and
any table row marked "(paper)" is from Han et al. 2025, not our run.
