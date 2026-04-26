#!/usr/bin/env python3
"""Run a one-seed 20% Yahoo! R3 LightGCN smoke test.

This script intentionally requires the official Yahoo! R3 train/test triple
files. The RL4Rec Yahoo matrices in related folders are simulation/model-output
artifacts and are not a substitute for the observational train split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data_raw" / "yahoo_r3"
ARTIFACT_DIR = ROOT / "artifacts" / "yahoo_smoke"
FIG_DIR = ROOT / "figures" / "yahoo_smoke"
TABLE_DIR = ROOT / "tables" / "yahoo_smoke"
TRAIN_NAME = "ydata-ymusic-rating-study-v1_0-train.txt"
TEST_NAME = "ydata-ymusic-rating-study-v1_0-test.txt"
KAGGLE_TRAIN_NAME = "kaggle/user.txt"
KAGGLE_TEST_NAME = "kaggle/random.txt"


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    model: str
    seed: int
    k_layers: int
    gamma: float
    residual: float
    embedding_dim: int = 32
    epochs: int = 20
    batch_size: int = 2048
    lr: float = 0.03
    weight_decay: float = 1e-5
    topk: int = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def expected_paths(data_dir: Path, source: str) -> tuple[Path, Path]:
    # keep source selection explicit because official and kaggle mirrors use different layouts.
    if source == "kaggle":
        return data_dir / KAGGLE_TRAIN_NAME, data_dir / KAGGLE_TEST_NAME
    return data_dir / TRAIN_NAME, data_dir / TEST_NAME


def audit_available_artifacts(data_dir: Path, rl4rec_zip: Path | None, source: str = "official") -> dict[str, object]:
    train_path, test_path = expected_paths(data_dir, source)
    official_train_path, official_test_path = expected_paths(data_dir, "official")
    kaggle_train_path, kaggle_test_path = expected_paths(data_dir, "kaggle")
    audit: dict[str, object] = {
        "selected_source": source,
        "expected_data_dir": str(data_dir),
        "selected_train_file": str(train_path),
        "selected_test_file": str(test_path),
        "selected_train_exists": train_path.exists(),
        "selected_test_exists": test_path.exists(),
        "official_train_file": str(official_train_path),
        "official_test_file": str(official_test_path),
        "official_train_exists": official_train_path.exists(),
        "official_test_exists": official_test_path.exists(),
        "kaggle_train_file": str(kaggle_train_path),
        "kaggle_test_file": str(kaggle_test_path),
        "kaggle_train_exists": kaggle_train_path.exists(),
        "kaggle_test_exists": kaggle_test_path.exists(),
        "rl4rec_zip": str(rl4rec_zip) if rl4rec_zip else None,
        "rl4rec_zip_exists": bool(rl4rec_zip and rl4rec_zip.exists()),
        "rl4rec_zip_files": [],
        "can_run_selected_source": train_path.exists() and test_path.exists(),
    }
    if rl4rec_zip and rl4rec_zip.exists():
        with zipfile.ZipFile(rl4rec_zip) as zf:
            audit["rl4rec_zip_files"] = [
                {"name": info.filename, "uncompressed_bytes": info.file_size}
                for info in zf.infolist()
            ]
        names = {row["name"] for row in audit["rl4rec_zip_files"]}  # type: ignore[index]
        audit["rl4rec_has_official_train_test"] = TRAIN_NAME in names and TEST_NAME in names
        audit["rl4rec_note"] = (
            "This zip is not used for the claimed Yahoo! R3 replication unless it "
            "contains the official train/test triple files."
        )
    return audit


def fail_if_missing_data(data_dir: Path, rl4rec_zip: Path | None, source: str) -> None:
    audit = audit_available_artifacts(data_dir, rl4rec_zip, source)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_DIR / "availability_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    if not audit["can_run_selected_source"]:
        missing = [
            str(path)
            for path in expected_paths(data_dir, source)
            if not path.exists()
        ]
        raise FileNotFoundError(
            f"Cannot run the Yahoo! R3 smoke test with source={source} because the selected train/test "
            f"files are missing: {missing}. Wrote availability audit to "
            f"{ARTIFACT_DIR / 'availability_audit.json'}."
        )


def read_triples(path: Path, split: str, source: str) -> pd.DataFrame:
    if source == "kaggle":
        df = pd.read_csv(path, sep=",", header=None, names=["user_id", "item_id", "rating"])
    else:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=["user_id", "item_id", "rating"])
    if df.empty:
        raise ValueError(f"{path} is empty")
    for col in ["user_id", "item_id", "rating"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{path} column {col} is not numeric")
    df = df.astype({"user_id": np.int32, "item_id": np.int32, "rating": np.int16})
    min_id = int(df[["user_id", "item_id"]].min().min())
    if source == "official":
        # normalize to zero-based ids so matrices match numpy and torch indexing.
        if min_id < 1:
            raise ValueError(f"{path} should use 1-based Yahoo! R3 user/item ids")
        df["user_id"] -= 1
        df["item_id"] -= 1
    else:
        if min_id < 0:
            raise ValueError(f"{path} should use nonnegative zero-based user/item ids")
    if df["rating"].min() < 1 or df["rating"].max() > 5:
        raise ValueError(f"{path} has ratings outside the expected 1-5 range")
    df["label"] = (df["rating"] >= 3).astype(np.int8)
    df["split"] = split
    return df


def sample_rows(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if not 0 < frac <= 1:
        raise ValueError("--sample-frac must be in (0, 1]")
    n = max(1, int(round(len(df) * frac)))
    # row sampling preserves the raw observational/random split distinction for smoke runs.
    return df.sample(n=n, random_state=seed, replace=False).sort_values(["user_id", "item_id"]).reset_index(drop=True)


def triples_to_matrix(df: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    mat = np.zeros(shape, dtype=np.int16)
    mat[df["user_id"].to_numpy(), df["item_id"].to_numpy()] = df["rating"].to_numpy()
    return mat


def top_share(degrees: np.ndarray, pct: float) -> float:
    values = np.sort(degrees.astype(float))[::-1]
    k = max(1, int(np.ceil(len(values) * pct / 100.0)))
    denom = values.sum()
    return float(values[:k].sum() / denom) if denom else 0.0


def dataset_summary(train: np.ndarray, test: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, mat in [("train_mnar", train), ("random_test", test)]:
        observed = mat > 0
        item_degrees = observed.sum(axis=0)
        user_degrees = observed.sum(axis=1)
        ratings = mat[observed]
        rows.append(
            {
                "dataset": "YahooR3",
                "split": name,
                "users": mat.shape[0],
                "items": mat.shape[1],
                "interactions": int(observed.sum()),
                "density": float(observed.mean()),
                "active_users": int((user_degrees > 0).sum()),
                "active_items": int((item_degrees > 0).sum()),
                "avg_user_degree": float(user_degrees.mean()),
                "avg_item_degree": float(item_degrees.mean()),
                # use the same relevance cutoff as Coat so Yahoo is an external-validity check.
                "positive_rate_r_ge_3": float((ratings >= 3).mean()) if len(ratings) else 0.0,
                "top_1_pct_share": top_share(item_degrees, 1),
                "top_5_pct_share": top_share(item_degrees, 5),
                "top_10_pct_share": top_share(item_degrees, 10),
            }
        )
    return pd.DataFrame(rows)


def load_and_sample_data(data_dir: Path, sample_frac: float, seed: int, source: str) -> dict[str, object]:
    train_path, test_path = expected_paths(data_dir, source)
    train_full = read_triples(train_path, "train_mnar", source)
    test_full = read_triples(test_path, "random_test", source)
    # shape uses the union of ids so users/items seen only in random test remain evaluable.
    n_users = int(max(train_full["user_id"].max(), test_full["user_id"].max()) + 1)
    n_items = int(max(train_full["item_id"].max(), test_full["item_id"].max()) + 1)

    train_sample = sample_rows(train_full, sample_frac, seed)
    # offset the test seed so train and random-test subsamples are not accidentally coupled.
    test_sample = sample_rows(test_full, sample_frac, seed + 10000)
    train = triples_to_matrix(train_sample, (n_users, n_items))
    test = triples_to_matrix(test_sample, (n_users, n_items))
    sample_tag = "full" if sample_frac == 1.0 else f"{int(round(sample_frac * 100))}pct"

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "data_cache").mkdir(parents=True, exist_ok=True)
    train_sample.to_csv(ARTIFACT_DIR / "data_cache" / f"yahoo_train_{sample_tag}_observed.csv", index=False)
    test_sample.to_csv(ARTIFACT_DIR / "data_cache" / f"yahoo_test_{sample_tag}_observed.csv", index=False)
    np.save(ARTIFACT_DIR / "data_cache" / f"yahoo_train_{sample_tag}.npy", train)
    np.save(ARTIFACT_DIR / "data_cache" / f"yahoo_test_{sample_tag}.npy", test)
    summary = dataset_summary(train, test)
    summary.to_csv(ARTIFACT_DIR / "dataset_summary.csv", index=False)
    return {
        "train": train,
        "test": test,
        "summary": summary,
        "source": source,
        "train_full_rows": len(train_full),
        "test_full_rows": len(test_full),
        "train_sample_rows": len(train_sample),
        "test_sample_rows": len(test_sample),
    }


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for model training. The data audit and parser are ready, "
            "but this environment does not currently have torch installed."
        ) from exc
    return torch


def build_sparse_operator(train: np.ndarray, gamma: float, device, torch) -> object:
    n_users, n_items = train.shape
    # mirror Coat: the graph is built from positive observational ratings only.
    pos = train >= 3
    user_ids, item_ids = np.where(pos)
    item_deg = pos.sum(axis=0).astype(np.float32)
    user_deg = pos.sum(axis=1).astype(np.float32)
    weights = []
    rows = []
    cols = []
    for user, item in zip(user_ids, item_ids):
        base = 1.0 / math.sqrt(max(user_deg[user], 1.0) * max(item_deg[item], 1.0))
        # gamma down-weights high-degree items to test whether popularity tempering helps.
        tempered = base / ((item_deg[item] + 1.0) ** gamma)
        u_node = int(user)
        i_node = int(n_users + item)
        rows.extend([u_node, i_node])
        cols.extend([i_node, u_node])
        weights.extend([tempered, tempered])
    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    val = torch.tensor(weights, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, (n_users + n_items, n_users + n_items)).coalesce()


def make_lightgcn_class(torch):
    class LightGCN(torch.nn.Module):
        def __init__(
            self,
            n_users: int,
            n_items: int,
            embedding_dim: int,
            k_layers: int,
            operator,
            residual: float,
        ) -> None:
            super().__init__()
            self.n_users = n_users
            self.n_items = n_items
            self.k_layers = k_layers
            self.operator = operator
            self.residual = residual
            self.user_embedding = torch.nn.Embedding(n_users, embedding_dim)
            self.item_embedding = torch.nn.Embedding(n_items, embedding_dim)
            torch.nn.init.normal_(self.user_embedding.weight, std=0.05)
            torch.nn.init.normal_(self.item_embedding.weight, std=0.05)

        def propagated(self):
            base = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            # layer averaging matches LightGCN and makes zero-hop the same model at k=0.
            out = base
            current = base
            for _ in range(self.k_layers):
                current = torch.sparse.mm(self.operator, current)
                if self.residual > 0:
                    # residual keeps corrected propagation anchored to the learned id embeddings.
                    current = self.residual * base + (1.0 - self.residual) * current
                out = out + current
            out = out / float(self.k_layers + 1)
            return out[: self.n_users], out[self.n_users :]

        def score_pairs(self, users, items):
            user_emb, item_emb = self.propagated()
            return (user_emb[users] * item_emb[items]).sum(dim=1)

        def score_all(self):
            user_emb, item_emb = self.propagated()
            return user_emb @ item_emb.T

    return LightGCN


def sample_batch(pos_pairs: np.ndarray, neg_pools: list[np.ndarray], batch_size: int, device, torch):
    idx = np.random.randint(0, len(pos_pairs), size=batch_size)
    users = pos_pairs[idx, 0]
    pos_items = pos_pairs[idx, 1]
    neg_items = np.array([np.random.choice(neg_pools[u]) for u in users], dtype=np.int64)
    return (
        torch.tensor(users, dtype=torch.long, device=device),
        torch.tensor(pos_items, dtype=torch.long, device=device),
        torch.tensor(neg_items, dtype=torch.long, device=device),
    )


def ndcg_at_k(labels: np.ndarray, k: int) -> float:
    gains = labels[:k] / np.log2(np.arange(2, min(k, len(labels)) + 2))
    ideal = np.sort(labels)[::-1]
    ideal_gains = ideal[:k] / np.log2(np.arange(2, min(k, len(ideal)) + 2))
    denom = ideal_gains.sum()
    return float(gains.sum() / denom) if denom > 0 else np.nan


def evaluate(model, test: np.ndarray, train: np.ndarray, cfg: RunConfig, torch) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        scores = model.score_all().detach().cpu().numpy()
    rows = []
    rec_rows = []
    item_degrees = (train >= 3).sum(axis=0)
    head_cutoff = np.quantile(item_degrees, 0.9)
    rec_counts = np.zeros(train.shape[1], dtype=float)
    test_counts = (test > 0).sum(axis=0).astype(float)

    for user_id in range(test.shape[0]):
        # restrict ranking to randomized ratings to avoid measuring reconstruction of logged edges.
        candidates = np.where(test[user_id] > 0)[0]
        if len(candidates) == 0:
            continue
        labels = (test[user_id, candidates] >= 3).astype(int)
        if labels.sum() == 0:
            continue
        order = np.argsort(scores[user_id, candidates])[::-1]
        ranked_items = candidates[order]
        ranked_labels = labels[order]
        top_items = ranked_items[: cfg.topk]
        top_labels = ranked_labels[: cfg.topk]
        rec_counts[top_items] += 1
        rows.append(
            {
                "user_id": user_id,
                "ndcg_at_5": ndcg_at_k(ranked_labels, cfg.topk),
                "recall_at_5": float(top_labels.sum() / labels.sum()),
                "top5_head_share": float((item_degrees[top_items] >= head_cutoff).mean()),
                "top5_avg_train_degree": float(item_degrees[top_items].mean()),
            }
        )
        for rank, item_id in enumerate(top_items, start=1):
            rec_rows.append(
                {
                    "user_id": user_id,
                    "rank": rank,
                    "item_id": int(item_id),
                    "rating": int(test[user_id, item_id]),
                    "label": int(test[user_id, item_id] >= 3),
                    "train_degree": int(item_degrees[item_id]),
                }
            )

    per_user = pd.DataFrame(rows)
    topk = pd.DataFrame(rec_rows)
    rec_dist = rec_counts / max(rec_counts.sum(), 1.0)
    test_dist = test_counts / max(test_counts.sum(), 1.0)
    metrics = {
        "ndcg_at_5": float(per_user["ndcg_at_5"].mean()),
        "recall_at_5": float(per_user["recall_at_5"].mean()),
        "head_item_share": float(per_user["top5_head_share"].mean()),
        "avg_recommended_train_degree": float(per_user["top5_avg_train_degree"].mean()),
        # js is reported alongside accuracy because lower shift is not useful if ndcg collapses.
        "js_recommendation_vs_random_test": float(jensenshannon(rec_dist, test_dist, base=2.0) ** 2),
        "n_eval_users": int(len(per_user)),
    }
    return {"metrics": metrics, "per_user": per_user, "topk": topk}


def train_one(train: np.ndarray, test: np.ndarray, cfg: RunConfig, device, torch) -> dict[str, object]:
    set_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    run_dir = ARTIFACT_DIR / "runs" / cfg.model / f"seed_{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # use identical bpr training across variants so only propagation changes the comparison.
    pos = train >= 3
    pos_pairs = np.column_stack(np.where(pos)).astype(np.int64)
    if len(pos_pairs) == 0:
        raise ValueError("sampled train split has no positive ratings under r >= 3")
    all_items = np.arange(train.shape[1], dtype=np.int64)
    neg_pools = [all_items[~pos[u]] for u in range(train.shape[0])]

    operator = build_sparse_operator(train, cfg.gamma, device, torch)
    LightGCN = make_lightgcn_class(torch)
    model = LightGCN(train.shape[0], train.shape[1], cfg.embedding_dim, cfg.k_layers, operator, cfg.residual).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []
    start = time.time()
    steps_per_epoch = max(1, math.ceil(len(pos_pairs) / cfg.batch_size))
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            users, pos_items, neg_items = sample_batch(pos_pairs, neg_pools, cfg.batch_size, device, torch)
            pos_scores = model.score_pairs(users, pos_items)
            neg_scores = model.score_pairs(users, neg_items)
            loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch in {1, cfg.epochs}:
            history.append({"epoch": epoch, "bpr_loss": float(np.mean(losses))})

    result = evaluate(model, test, train, cfg, torch)
    result["metrics"]["runtime_seconds"] = time.time() - start
    result["metrics"]["used_cuda"] = bool(device.type == "cuda")
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    result["per_user"].to_csv(run_dir / "per_user_metrics.csv", index=False)
    result["topk"].to_csv(run_dir / "topk_recommendations.csv", index=False)
    return {"config": cfg, **result}


def aggregate(results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    per_user_frames = []
    for result in results:
        cfg = result["config"]
        rows.append(asdict(cfg) | result["metrics"])
        per = result["per_user"].copy()
        per["model"] = cfg.model
        per["seed"] = cfg.seed
        per_user_frames.append(per)
    runs = pd.DataFrame(rows)
    (ARTIFACT_DIR / "summaries").mkdir(parents=True, exist_ok=True)
    runs.to_csv(ARTIFACT_DIR / "summaries" / "all_runs.csv", index=False)
    summary = (
        runs.groupby(["model", "k_layers", "gamma", "residual"], as_index=False)
        .agg(
            ndcg_mean=("ndcg_at_5", "mean"),
            recall_mean=("recall_at_5", "mean"),
            head_share_mean=("head_item_share", "mean"),
            js_mean=("js_recommendation_vs_random_test", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
            n_eval_users=("n_eval_users", "mean"),
        )
        .sort_values(["model", "k_layers", "gamma", "residual"])
    )
    summary.to_csv(ARTIFACT_DIR / "summaries" / "main_results.csv", index=False)
    pd.concat(per_user_frames, ignore_index=True).to_csv(ARTIFACT_DIR / "summaries" / "per_user_all.csv", index=False)
    return summary


def make_figures(train: np.ndarray, test: np.ndarray, data_summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)

    train_deg = (train > 0).sum(axis=0)
    test_deg = (test > 0).sum(axis=0)
    ranks = np.arange(1, train.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), dpi=240)
    axes[0].plot(ranks, np.sort(train_deg)[::-1], label="train", color="#2b6cb0", linewidth=2)
    axes[0].plot(ranks, np.sort(test_deg)[::-1], label="random test", color="#c05621", linewidth=2)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("item popularity rank")
    axes[0].set_ylabel("observed item degree")
    axes[0].set_title("a. item-degree rank curve", fontsize=10)
    axes[0].legend(frameon=True, fontsize=7, loc="upper right")

    head = data_summary.melt(
        id_vars=["split"],
        value_vars=["top_1_pct_share", "top_5_pct_share", "top_10_pct_share"],
        var_name="bucket",
        value_name="share",
    )
    head["bucket"] = head["bucket"].map({"top_1_pct_share": "top 1%", "top_5_pct_share": "top 5%", "top_10_pct_share": "top 10%"})
    sns.barplot(data=head, x="bucket", y="share", hue="split", palette=["#2b6cb0", "#c05621"], ax=axes[1])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("share of observed interactions")
    axes[1].set_title("b. head-item concentration", fontsize=10)
    axes[1].legend(frameon=True, fontsize=6, loc="upper left")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{100*x:.0f}%")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_yahoo_smoke_popularity_shift.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_yahoo_smoke_popularity_shift.png", bbox_inches="tight")
    plt.close(fig)

    candidate_counts = (test > 0).sum(axis=1)
    positive_counts = (test >= 3).sum(axis=1)
    active_candidates = candidate_counts[candidate_counts > 0]
    eval_positive_counts = positive_counts[positive_counts > 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), dpi=240)
    sns.countplot(x=active_candidates, color="#4a5568", ax=axes[0])
    axes[0].set_xlabel("test candidates per user")
    axes[0].set_ylabel("users")
    axes[0].set_title("a. candidate set sizes", fontsize=10)
    sns.countplot(x=eval_positive_counts, color="#805ad5", ax=axes[1])
    axes[1].set_xlabel("positive test items per evaluated user")
    axes[1].set_ylabel("users")
    axes[1].set_title("b. positives under r >= 3", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_yahoo_smoke_candidate_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_yahoo_smoke_candidate_diagnostics.png", bbox_inches="tight")
    plt.close(fig)

    results_path = ARTIFACT_DIR / "summaries" / "all_runs.csv"
    if results_path.exists():
        runs = pd.read_csv(results_path)
        depth = runs[runs["model"].isin(["zero_hop", "lightgcn_k1", "lightgcn_k2", "lightgcn_k3"])].copy()
        fig, ax = plt.subplots(figsize=(4.1, 2.6), dpi=240)
        sns.lineplot(data=depth, x="k_layers", y="ndcg_at_5", marker="o", errorbar=None, color="#2b6cb0", ax=ax)
        ax.set_xlabel("propagation depth k")
        ax.set_ylabel("random-test ndcg@5")
        ax.set_title("Yahoo depth sweep", fontsize=10)
        ax.set_xticks([0, 1, 2, 3])
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig2_yahoo_smoke_depth_sweep.pdf", bbox_inches="tight")
        fig.savefig(FIG_DIR / "fig2_yahoo_smoke_depth_sweep.png", bbox_inches="tight")
        plt.close(fig)

        model_order = ["zero_hop", "lightgcn_k1", "lightgcn_k2", "lightgcn_k3", "corrected_k2", "corrected_k2_stronger"]
        labels = {
            "zero_hop": "0-hop",
            "lightgcn_k1": "K1",
            "lightgcn_k2": "K2",
            "lightgcn_k3": "K3",
            "corrected_k2": "Corr",
            "corrected_k2_stronger": "Corr+",
        }
        compare = runs.set_index("model").loc[model_order].reset_index()
        compare["label"] = compare["model"].map(labels)
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), dpi=240)
        sns.barplot(data=compare, x="label", y="ndcg_at_5", color="#2b6cb0", ax=axes[0])
        axes[0].set_xlabel("")
        axes[0].set_ylabel("random-test ndcg@5")
        axes[0].set_title("a. ranking accuracy", fontsize=10)
        axes[0].set_ylim(max(0.0, compare["ndcg_at_5"].min() - 0.01), min(1.0, compare["ndcg_at_5"].max() + 0.01))
        axes[0].tick_params(axis="x", labelsize=8)

        sns.barplot(data=compare, x="label", y="js_recommendation_vs_random_test", color="#2f855a", ax=axes[1])
        axes[1].set_xlabel("")
        axes[1].set_ylabel("JS(rec, test)")
        axes[1].set_title("b. recommendation distribution shift", fontsize=10)
        axes[1].tick_params(axis="x", labelsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig3_yahoo_smoke_accuracy_bias.pdf", bbox_inches="tight")
        fig.savefig(FIG_DIR / "fig3_yahoo_smoke_accuracy_bias.png", bbox_inches="tight")
        plt.close(fig)

    data_summary.to_latex(TABLE_DIR / "dataset_summary.tex", index=False, float_format="%.4f", escape=False)
    if (ARTIFACT_DIR / "summaries" / "main_results.csv").exists():
        pd.read_csv(ARTIFACT_DIR / "summaries" / "main_results.csv").to_latex(
            TABLE_DIR / "main_results.tex", index=False, float_format="%.4f", escape=False
        )


def resolve_device(choice: str, torch):
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    global ARTIFACT_DIR, FIG_DIR, TABLE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--sample-frac", type=float, default=0.20)
    parser.add_argument("--source", choices=["kaggle", "official"], default="kaggle")
    parser.add_argument("--output-name", default="yahoo_smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-only", action="store_true", help="parse/sample data and make EDA plots without model training")
    parser.add_argument("--audit-only", action="store_true", help="write availability audit and exit")
    parser.add_argument("--rl4rec-zip", type=Path, default=Path("/tmp/yahoo_probe/yahoo-data.zip"))
    args = parser.parse_args()
    ARTIFACT_DIR = ROOT / "artifacts" / args.output_name
    FIG_DIR = ROOT / "figures" / args.output_name
    TABLE_DIR = ROOT / "tables" / args.output_name

    if args.audit_only:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        audit = audit_available_artifacts(args.data_dir, args.rl4rec_zip, args.source)
        with open(ARTIFACT_DIR / "availability_audit.json", "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
        print(json.dumps(audit, indent=2))
        return

    fail_if_missing_data(args.data_dir, args.rl4rec_zip, args.source)
    data = load_and_sample_data(args.data_dir, args.sample_frac, args.seed, args.source)
    make_figures(data["train"], data["test"], data["summary"])
    if args.data_only:
        print(json.dumps({"status": "ok", "mode": "data-only", "summary": str(ARTIFACT_DIR / "dataset_summary.csv")}, indent=2))
        return

    torch = import_torch()
    device = resolve_device(args.device, torch)
    dataset_name = "yahoo_r3_full" if args.sample_frac == 1.0 else f"yahoo_r3_sample{int(round(args.sample_frac * 100))}"
    configs = [
        # the one-seed yahoo replication mirrors the coat model grid without bootstrapping.
        RunConfig(dataset_name, "zero_hop", args.seed, 0, 0.0, 0.0, epochs=args.epochs),
        RunConfig(dataset_name, "lightgcn_k1", args.seed, 1, 0.0, 0.0, epochs=args.epochs),
        RunConfig(dataset_name, "lightgcn_k2", args.seed, 2, 0.0, 0.0, epochs=args.epochs),
        RunConfig(dataset_name, "lightgcn_k3", args.seed, 3, 0.0, 0.0, epochs=args.epochs),
        RunConfig(dataset_name, "corrected_k2", args.seed, 2, 0.35, 0.20, epochs=args.epochs),
        RunConfig(dataset_name, "corrected_k2_stronger", args.seed, 2, 0.70, 0.20, epochs=args.epochs),
    ]
    results = [train_one(data["train"], data["test"], cfg, device, torch) for cfg in configs]
    summary = aggregate(results)
    make_figures(data["train"], data["test"], data["summary"])
    manifest = {
        "status": "ok",
        "dataset": f"Yahoo! R3 {args.source} train/test, observed-row sample",
        "runs": len(results),
        "seed": args.seed,
        "sample_frac": args.sample_frac,
        "epochs": args.epochs,
        "device": str(device),
        "summary_rows": len(summary),
    }
    with open(ARTIFACT_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
