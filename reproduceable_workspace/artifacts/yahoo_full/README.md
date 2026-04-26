# Yahoo! R3 Full-Data Replication

## Source

Downloaded the public Kaggle mirror `limitiao/yahoor3` using:

```bash
curl -L 'https://www.kaggle.com/api/v1/datasets/download/limitiao/yahoor3' -o /tmp/yahoo_kaggle_probe/yahoor3.zip
```

The extracted files are in `reproduceable_workspace/data_raw/yahoo_r3/kaggle/`.
This is labeled as a Kaggle mirror, not as an official Yahoo Webscope package.

## Data Used

- `user.txt`: observational/MNAR train; 311,704 ratings, 15,400 users, 1,000 items.
- `random.txt`: randomized test; 54,000 ratings, 5,400 users, 1,000 items.
- `sampling_data.txt`: not used because it is a smaller duplicate-containing sample, not the train/test protocol.

The full-data run uses all observed rows from train and test. Ratings are binarized with `r >= 3`.

## Run

```bash
/venv/main/bin/python reproduceable_workspace/scripts/run_yahoo_lightgcn_smoke.py \
  --source kaggle --output-name yahoo_full --sample-frac 1.0 --seed 0 --epochs 20 --device cuda
```

CUDA was used with PyTorch `2.11.0+cu128` on an NVIDIA GeForce RTX 3060. This is one full-data seed, not a multi-seed/bootstrapped run.

## Main Results

| model | NDCG@5 | Recall@5 | head share | JS(rec,test) |
|---|---:|---:|---:|---:|
| zero-hop | 0.6528 | 0.6999 | 0.1677 | 0.0351 |
| LightGCN K=1 | 0.6794 | 0.7263 | 0.1738 | 0.0457 |
| LightGCN K=2 | 0.6735 | 0.7293 | 0.1824 | 0.0609 |
| LightGCN K=3 | 0.6601 | 0.7167 | 0.1894 | 0.0742 |
| corrected K=2 | 0.6543 | 0.7069 | 0.1780 | 0.0559 |
| corrected K=2 stronger | 0.6468 | 0.7010 | 0.1802 | 0.0607 |

## Interpretation Guardrails

- Full Yahoo uses 3,922 evaluated users with at least one positive randomized-test item.
- RQ1 is supported: propagation improves over zero-hop, with best NDCG at K=1 and best Recall@5 at K=2.
- Deeper propagation amplifies popularity shift and hurts NDCG on Yahoo.
- RQ2 is not supported: corrected variants reduce some shift relative to vanilla K=2 but do not preserve accuracy.
- These results are incorporated into `../../../../final_report.tex` as a compute-limited full-data replication.
