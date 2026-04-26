# Causal Graph Learning for Randomized-Test Recommendation

This repository contains the code and reproducibility artifacts for `final_report.pdf`, **"When does graph propagation help causal recommendation?"** The project studies whether LightGCN-style graph propagation helps ranking under randomized-test evaluation when the training graph is missing-not-at-random (MNAR).

The main comparison is controlled: all models use the same ID embeddings, BPR loss, optimizer, relevance threshold, and train/test splits. The ablation changes only the graph propagation operator:

- `zero_hop`: matched dot-product embedding baseline, equivalent to LightGCN with `K=0`.
- `lightgcn_k1`, `lightgcn_k2`, `lightgcn_k3`: vanilla LightGCN propagation depths.
- `corrected_k2`, `corrected_k2_stronger`: two degree-tempered residual propagation variants used to test whether a simple popularity penalty reduces bias without hurting ranking.

## Repository Contents

```text
final_report.pdf                         final submitted report
final_report.tex                         LaTeX source for the report
requirements.txt                         Python package list
reproduceable_workspace/
  README.md                              short original workflow note
  data_raw/
    coat/                                Coat raw train/test matrices used by this project
    yahoo_r3/kaggle/                     Yahoo! R3 Kaggle mirror files used here
  scripts/
    run_coat_lightgcn.py                 Coat training, evaluation, aggregation, figures, tables
    run_yahoo_lightgcn_smoke.py          Yahoo! R3 data audit, full/smoke training, figures, tables
    make_mechanism_artifacts.py          mechanism diagnostics and correction swap summaries
    make_report_figures.py               publication figure regeneration from cached outputs
  artifacts/
    summaries/                           cached Coat CSV summaries used for report tables/figures
    runs/coat/                           cached per-run CSV/JSON outputs, without model checkpoints
    yahoo_full/                          cached Yahoo full-run CSV/JSON summaries
  figures/                               report and diagnostic figures as PDF/PNG
  tables/                                generated LaTeX tables
  notebooks/                             executed audit notebooks
```

Large files that are not needed to reproduce the report were removed or ignored: copied external RL repositories, smoke-test output caches, LaTeX build logs, temporary rendered PDF pages, checkpoint files, and learned embedding `.npy` files. The retained run artifacts are CSV/JSON outputs so the reported tables and diagnostics can be audited without storing model binaries.

## Data

Two explicit-feedback datasets are used.

| Dataset | Training split | Test split | Local files |
| --- | --- | --- | --- |
| Coat | observational/MNAR ratings | randomized ratings | `reproduceable_workspace/data_raw/coat/train.ascii`, `test.ascii` |
| Yahoo! R3 | user-supplied ratings | randomized ratings | `reproduceable_workspace/data_raw/yahoo_r3/kaggle/user.txt`, `random.txt` |

Both scripts convert ratings to a binary relevance label with `rating >= 3`. The observational split is used for training; the randomized split is used only for evaluation.

## Environment Setup

Use Python 3. A virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For full retraining, install a CUDA-enabled PyTorch build that matches your machine. The Coat training script intentionally raises an error if CUDA is unavailable. The final report states that the reported training runs used CUDA on an NVIDIA GeForce RTX 3060.

To compile the PDF, install a LaTeX distribution that provides `pdflatex` and the packages used in `final_report.tex` (`booktabs`, `amsmath`, `graphicx`, `subcaption`, `natbib`, `times`, `titlesec`, and standard LaTeX dependencies).

## Fast Reproduction From Cached CSV Outputs

This path verifies the report artifacts without retraining models. It uses the retained CSV/JSON outputs and regenerates data caches, figures, and tables.

```bash
cd reproduceable_workspace

# rebuild Coat data cache, figures, and base LaTeX tables from raw data + cached summaries
python3 scripts/run_coat_lightgcn.py --artifacts-only

# rebuild mechanism diagnostics, swap summaries, and mechanism table
python3 scripts/make_mechanism_artifacts.py

# rebuild publication figures from cached summaries
python3 scripts/make_report_figures.py

# rebuild Yahoo full-data figures from raw Yahoo files + cached yahoo_full summaries
python3 scripts/run_yahoo_lightgcn_smoke.py \
  --source kaggle \
  --output-name yahoo_full \
  --sample-frac 1.0 \
  --seed 0 \
  --epochs 20 \
  --data-only

cd ..
pdflatex -interaction=nonstopmode final_report.tex
pdflatex -interaction=nonstopmode final_report.tex
```

The final report should compile to `final_report.pdf`. The second `pdflatex` pass resolves cross-references.

## Full Retraining Workflow

This path reruns the experiments used to produce the cached outputs. It is slower and requires CUDA for the Coat workflow.

```bash
cd reproduceable_workspace

# Coat: 6 model variants x 20 seeds
python3 scripts/run_coat_lightgcn.py --seed-count 20 --no-reuse

# Yahoo! R3: one full-data seed, matching the report's external-validity run
python3 scripts/run_yahoo_lightgcn_smoke.py \
  --source kaggle \
  --output-name yahoo_full \
  --sample-frac 1.0 \
  --seed 0 \
  --epochs 20 \
  --device cuda

# Derived mechanism artifacts and final figures
python3 scripts/make_mechanism_artifacts.py
python3 scripts/make_report_figures.py

cd ..
pdflatex -interaction=nonstopmode final_report.tex
pdflatex -interaction=nonstopmode final_report.tex
```

The scripts write outputs under `reproduceable_workspace/artifacts/`, `reproduceable_workspace/figures/`, and `reproduceable_workspace/tables/`.

## Useful Debugging Commands

Audit whether the Yahoo files are present:

```bash
cd reproduceable_workspace
python3 scripts/run_yahoo_lightgcn_smoke.py --audit-only --source kaggle
```

Run a shorter Coat smoke test:

```bash
cd reproduceable_workspace
python3 scripts/run_coat_lightgcn.py --quick
```

Run a Yahoo 20% sample instead of the full run:

```bash
cd reproduceable_workspace
python3 scripts/run_yahoo_lightgcn_smoke.py \
  --source kaggle \
  --output-name yahoo_smoke \
  --sample-frac 0.20 \
  --seed 0 \
  --epochs 20 \
  --device cuda
```

`yahoo_smoke` outputs are ignored by git because they are debugging artifacts, not part of the submitted final report.

## How The Code Maps To The Report

- Section 3 dataset statistics come from the raw matrices/triples and cached dataset summaries.
- Section 4 model definitions are implemented in `build_sparse_operator`, `LightGCN.propagated`, and the BPR training loop in the two training scripts.
- Section 5 randomized-test metrics are computed in each script's `evaluate` function. Ranking is restricted to randomized-test rated items for each user.
- Popularity shift is measured with top-5 head-item share and Jensen-Shannon divergence against the randomized-test item distribution.
- Table 3 mechanism diagnostics come from `make_mechanism_artifacts.py` and `notebooks/02_mechanism_support_table.ipynb`.

## Notes On Reproducibility

- Coat results in the report are averaged over 20 seeds.
- Yahoo full-data results in the report are one seed, not a bootstrap or multi-seed estimate.
- The cached repository intentionally keeps CSV/JSON summaries but omits model checkpoints and learned embeddings to keep the GitHub repo small.
- Exact retraining numbers can vary slightly across hardware and PyTorch/CUDA environments. The retained CSV outputs are the values used to build the submitted report.
