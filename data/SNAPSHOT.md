# Data snapshot: IFC projects, downloaded 14 November 2026

The trained category model was built from this exact copy of IFC's public data. IFC adds and
edits projects every day, so downloading again later gives slightly different data. To reproduce
our results exactly, use this snapshot instead of downloading again.

The snapshot itself is **not stored in git** (it is 65 MB and contains IFC's page texts), and it is
kept private. This file describes it, so you can check that a copy is the right one. If you only
want to *use* the model, you do not need the snapshot: the trained model is published on the
GitHub release `v0.2.0` (see the README's Quick start).

## The archive

| | |
|---|---|
| File | `ifc_snapshot_2026-11-14.tar.gz` (64,879,916 bytes) |
| SHA-256 | `4237ee50dfbec240f971d9e59ff724914373a37500761042f4be526b0cced2ce` |
| Stored at | the project owner's Google Drive (private: ask JanNikolaidovic for access) |

Contents (paths as in this repository):

| Path | What | SHA-256 |
|---|---|---|
| `data/raw/ifc/project_list.json` | the API's answer: 7,187 IFC projects, untouched | `68ec63885b33e34c7fb76603bf378ce731e7526d484d5b9e826467ddd95b4dcd` |
| `data/raw/ifc/pages/*.html.gz` | 4,015 project pages exactly as IFC's server sent them (compressed) | |
| `data/raw/ifc/download.log` | log of the download (times, 2 failed pages) | |
| `data/processed/ifc_projects.csv` | the training table: 3,994 projects (306 A, 3,168 B, 520 C) | `97f378d0708b2340947951b4d47e6a794e2a3edcdd6aed61ce550e5ce4cd8f16` |
| `data/processed/ifc_esgify_features.csv` | ESGify scores per project (saves ~1.5 h when re-running `ifc-evaluate`) | `6d8452b39b01c62f135ee9db1e07c3217665d7f3b88bbfb8c70160bcd946172b` |
| `models/category_model.joblib` + `.json` | the trained category model and its report card (also published on GitHub release `v0.2.0`) | `0827404013c922b82dec3402c2efd585c87fcd7c4a0337abd019725028570e7f` (`.joblib`) |

## Reproduce the results

```bash
# 1. set up the code and the exact package versions (see README, Quick start)
# 2. put the archive in the repository folder, check it and unpack it
sha256sum ifc_snapshot_2026-11-14.tar.gz        # must print the SHA-256 above
tar -xzf ifc_snapshot_2026-11-14.tar.gz          # recreates data/raw/ifc, data/processed and models/

# 3. optional: rebuild the training table from the raw pages; it must match the archived one
python main.py ifc-build
sha256sum data/processed/ifc_projects.csv        # must print 97f378d0...

# 4. retrain: gives the same model, with the same test results as models/category_model.json
python main.py train-category
```

Training has no randomness: retraining from `ifc_projects.csv` with the package versions in
`requirements.txt` gives the same vocabulary, the same coefficients and the same test results
(checked on 15 November 2026). The `.joblib` file itself can differ byte by byte (it is a saved
Python object), and the report card's `trained_at` date changes, so compare test results, not files.
