#!/usr/bin/env python3
"""run coat lightgcn experiments and cache plot-ready artifacts."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.spatial.distance import jensenshannon


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_raw" / "coat"
ARTIFACT_DIR = ROOT / "artifacts"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    model: str
    seed: int
    k_layers: int
    gamma: float
    residual: float
    embedding_dim: int = 32
    epochs: int = 180
    batch_size: int = 1024
    lr: float = 0.03
    weight_decay: float = 1e-5
    topk: int = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_ascii_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int16)


def matrix_to_observed_df(mat: np.ndarray, split: str) -> pd.DataFrame:
    users, items = np.where(mat > 0)
    ratings = mat[users, items]
    return pd.DataFrame(
        {
            "user_id": users.astype(np.int32),
            "item_id": items.astype(np.int32),
            "rating": ratings.astype(np.int16),
            "label": (ratings >= 3).astype(np.int8),
            "split": split,
        }
    )


def top_share(degrees: np.ndarray, pct: float) -> float:
    # head-share tracks exposure concentration without relying on bipartite clustering.
    values = np.sort(degrees.astype(float))[::-1]
    k = max(1, int(np.ceil(len(values) * pct / 100.0)))
    denom = values.sum()
    return float(values[:k].sum() / denom) if denom else 0.0


def load_and_cache_data() -> dict[str, object]:
    # cache the parsed matrices because all later stages reuse the same split definitions.
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "data_cache").mkdir(parents=True, exist_ok=True)
    train = read_ascii_matrix(DATA_DIR / "train.ascii")
    test = read_ascii_matrix(DATA_DIR / "test.ascii")
    np.save(ARTIFACT_DIR / "data_cache" / "coat_train.npy", train)
    np.save(ARTIFACT_DIR / "data_cache" / "coat_test.npy", test)

    train_df = matrix_to_observed_df(train, "train_mnar")
    test_df = matrix_to_observed_df(test, "random_test")
    train_df.to_csv(ARTIFACT_DIR / "data_cache" / "coat_train_observed.csv", index=False)
    test_df.to_csv(ARTIFACT_DIR / "data_cache" / "coat_test_observed.csv", index=False)

    rows = []
    for name, mat in [("train_mnar", train), ("random_test", test)]:
        observed = mat > 0
        item_degrees = observed.sum(axis=0)
        user_degrees = observed.sum(axis=1)
        ratings = mat[observed]
        rows.append(
            {
                "dataset": "Coat",
                "split": name,
                "users": mat.shape[0],
                "items": mat.shape[1],
                "interactions": int(observed.sum()),
                "density": float(observed.mean()),
                "avg_user_degree": float(user_degrees.mean()),
                "avg_item_degree": float(item_degrees.mean()),
                "positive_rate_r_ge_3": float((ratings >= 3).mean()),
                "top_1_pct_share": top_share(item_degrees, 1),
                "top_5_pct_share": top_share(item_degrees, 5),
                "top_10_pct_share": top_share(item_degrees, 10),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(ARTIFACT_DIR / "dataset_summary.csv", index=False)

    return {"train": train, "test": test, "summary": summary}


def build_sparse_operator(train: np.ndarray, gamma: float, device: torch.device) -> torch.Tensor:
    n_users, n_items = train.shape
    # use only liked observational edges so propagation shares preference signal, not every exposure.
    pos = train >= 3
    user_ids, item_ids = np.where(pos)
    item_deg = pos.sum(axis=0).astype(np.float32)
    user_deg = pos.sum(axis=1).astype(np.float32)
    weights = []
    rows = []
    cols = []
    for user, item in zip(user_ids, item_ids):
        base = 1.0 / math.sqrt(max(user_deg[user], 1.0) * max(item_deg[item], 1.0))
        # gamma implements the report's local popularity penalty on item messages.
        tempered = base / ((item_deg[item] + 1.0) ** gamma)
        u_node = user
        i_node = n_users + item
        rows.extend([u_node, i_node])
        cols.extend([i_node, u_node])
        weights.extend([tempered, tempered])
    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    val = torch.tensor(weights, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, (n_users + n_items, n_users + n_items)).coalesce()


class LightGCN(torch.nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        k_layers: int,
        operator: torch.Tensor,
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

    def propagated(self) -> tuple[torch.Tensor, torch.Tensor]:
        base = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        # keep the unpropagated layer in the average so k=0 is the matched zero-hop control.
        out = base
        current = base
        for _ in range(self.k_layers):
            current = torch.sparse.mm(self.operator, current)
            if self.residual > 0:
                # residual mixing limits how far the corrected variants drift from learned id embeddings.
                current = self.residual * base + (1.0 - self.residual) * current
            out = out + current
        out = out / float(self.k_layers + 1)
        return out[: self.n_users], out[self.n_users :]

    def score_pairs(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_emb, item_emb = self.propagated()
        return (user_emb[users] * item_emb[items]).sum(dim=1)

    def score_all(self) -> torch.Tensor:
        user_emb, item_emb = self.propagated()
        return user_emb @ item_emb.T


def sample_batch(pos_pairs: np.ndarray, neg_pools: list[np.ndarray], batch_size: int, device: torch.device):
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


def evaluate(model: LightGCN, test: np.ndarray, train: np.ndarray, cfg: RunConfig, device: torch.device) -> dict[str, object]:
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
        # rank only randomized-test candidates so evaluation targets forced-exposure outcomes.
        candidates = np.where(test[user_id] > 0)[0]
        if len(candidates) == 0:
            continue
        labels = (test[user_id, candidates] >= 3).astype(int)
        if labels.sum() == 0:
            # ndcg and recall are undefined when a user's randomized slate has no positives.
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
    # js compares recommended exposure against the randomized-test item distribution.
    js = float(jensenshannon(rec_dist, test_dist, base=2.0) ** 2)
    metrics = {
        "ndcg_at_5": float(per_user["ndcg_at_5"].mean()),
        "recall_at_5": float(per_user["recall_at_5"].mean()),
        "head_item_share": float(per_user["top5_head_share"].mean()),
        "avg_recommended_train_degree": float(per_user["top5_avg_train_degree"].mean()),
        "js_recommendation_vs_random_test": js,
        "n_eval_users": int(len(per_user)),
    }
    return {"metrics": metrics, "per_user": per_user, "topk": topk}


def train_one(train: np.ndarray, test: np.ndarray, cfg: RunConfig, device: torch.device) -> dict[str, object]:
    set_seed(cfg.seed)
    run_dir = ARTIFACT_DIR / "runs" / cfg.dataset / cfg.model / f"seed_{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # bpr trains on observed positives and contrasts them with uniformly sampled non-positives.
    pos = train >= 3
    pos_pairs = np.column_stack(np.where(pos)).astype(np.int64)
    all_items = np.arange(train.shape[1], dtype=np.int64)
    neg_pools = [all_items[~pos[u]] for u in range(train.shape[0])]

    operator = build_sparse_operator(train, cfg.gamma, device)
    model = LightGCN(train.shape[0], train.shape[1], cfg.embedding_dim, cfg.k_layers, operator, cfg.residual).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []
    start = time.time()
    steps_per_epoch = max(1, math.ceil(len(pos_pairs) / cfg.batch_size))
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            users, pos_items, neg_items = sample_batch(pos_pairs, neg_pools, cfg.batch_size, device)
            pos_scores = model.score_pairs(users, pos_items)
            neg_scores = model.score_pairs(users, neg_items)
            loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch in {1, 25, 50, 100, cfg.epochs}:
            history.append({"epoch": epoch, "bpr_loss": float(np.mean(losses))})

    runtime = time.time() - start
    result = evaluate(model, test, train, cfg, device)
    result["metrics"]["runtime_seconds"] = runtime
    result["metrics"]["used_cuda"] = bool(device.type == "cuda")

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    result["per_user"].to_csv(run_dir / "per_user_metrics.csv", index=False)
    result["topk"].to_csv(run_dir / "topk_recommendations.csv", index=False)
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    user_emb, item_emb = model.propagated()
    np.save(run_dir / "user_embeddings.npy", user_emb.detach().cpu().numpy())
    np.save(run_dir / "item_embeddings.npy", item_emb.detach().cpu().numpy())
    return {"config": cfg, **result}


def load_cached_run(cfg: RunConfig) -> dict[str, object] | None:
    run_dir = ARTIFACT_DIR / "runs" / cfg.dataset / cfg.model / f"seed_{cfg.seed}"
    needed = [run_dir / "metrics.json", run_dir / "per_user_metrics.csv", run_dir / "topk_recommendations.csv"]
    if not all(path.exists() for path in needed):
        return None
    with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "config": cfg,
        "metrics": metrics,
        "per_user": pd.read_csv(run_dir / "per_user_metrics.csv"),
        "topk": pd.read_csv(run_dir / "topk_recommendations.csv"),
    }


def aggregate_and_bootstrap(results: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    per_user_frames = []
    for result in results:
        cfg = result["config"]
        metrics = result["metrics"]
        row = asdict(cfg) | metrics
        rows.append(row)
        per = result["per_user"].copy()
        per["model"] = cfg.model
        per["seed"] = cfg.seed
        per_user_frames.append(per)
    runs = pd.DataFrame(rows)
    runs.to_csv(ARTIFACT_DIR / "summaries" / "all_runs.csv", index=False)

    summary = (
        runs.groupby(["model", "k_layers", "gamma", "residual"], as_index=False)
        .agg(
            ndcg_mean=("ndcg_at_5", "mean"),
            ndcg_std=("ndcg_at_5", "std"),
            recall_mean=("recall_at_5", "mean"),
            recall_std=("recall_at_5", "std"),
            head_share_mean=("head_item_share", "mean"),
            js_mean=("js_recommendation_vs_random_test", "mean"),
            avg_runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values(["model", "k_layers", "gamma", "residual"])
    )
    summary.to_csv(ARTIFACT_DIR / "summaries" / "main_results.csv", index=False)

    per_user_all = pd.concat(per_user_frames, ignore_index=True)
    per_user_all.to_csv(ARTIFACT_DIR / "summaries" / "per_user_all.csv", index=False)
    bootstrap_rows = []
    comparisons = [
        ("lightgcn_k1", "zero_hop"),
        ("lightgcn_k2", "zero_hop"),
        ("lightgcn_k3", "zero_hop"),
        ("lightgcn_k3", "lightgcn_k2"),
        ("corrected_k2", "lightgcn_k2"),
        ("corrected_k2_stronger", "lightgcn_k2"),
    ]
    rng = np.random.default_rng(20260424)
    for model_a, model_b in comparisons:
        a = per_user_all[per_user_all["model"] == model_a].groupby("user_id")["ndcg_at_5"].mean()
        b = per_user_all[per_user_all["model"] == model_b].groupby("user_id")["ndcg_at_5"].mean()
        common = a.index.intersection(b.index)
        # pair by user before bootstrapping so variance reflects within-user model differences.
        diff = (a.loc[common] - b.loc[common]).to_numpy()
        boot = [float(diff[rng.integers(0, len(diff), size=len(diff))].mean()) for _ in range(10000)]
        bootstrap_rows.append(
            {
                "comparison": f"{model_a} minus {model_b}",
                "mean_delta_ndcg_at_5": float(diff.mean()),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "n_users": int(len(diff)),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(ARTIFACT_DIR / "summaries" / "bootstrap_summary.csv", index=False)
    return summary, bootstrap


def make_figures(train: np.ndarray, test: np.ndarray, summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
    palette = {"train_mnar": "#2b6cb0", "random_test": "#c05621"}

    train_deg = (train > 0).sum(axis=0)
    test_deg = (test > 0).sum(axis=0)
    ranks = np.arange(1, train.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), dpi=240)
    axes[0].plot(ranks, np.sort(train_deg)[::-1], label="train (mnar)", color=palette["train_mnar"], linewidth=2)
    axes[0].plot(ranks, np.sort(test_deg)[::-1], label="random test", color=palette["random_test"], linewidth=2)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("item popularity rank")
    axes[0].set_ylabel("observed item degree")
    axes[0].set_title("a. item-degree rank curve", fontsize=10)
    axes[0].legend(frameon=True, fontsize=7, loc="upper right")
    head = summary.melt(
        id_vars=["split"],
        value_vars=["top_1_pct_share", "top_5_pct_share", "top_10_pct_share"],
        var_name="bucket",
        value_name="share",
    )
    head["bucket"] = head["bucket"].map({"top_1_pct_share": "top 1%", "top_5_pct_share": "top 5%", "top_10_pct_share": "top 10%"})
    head["split"] = head["split"].map({"train_mnar": "train", "random_test": "random test"})
    bar_palette = {"train": "#2b6cb0", "random test": "#c05621"}
    sns.barplot(data=head, x="bucket", y="share", hue="split", palette=bar_palette, ax=axes[1])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("share of observed interactions")
    axes[1].set_title("b. head-item concentration", fontsize=10)
    axes[1].legend(frameon=True, fontsize=7, loc="upper left")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{100*x:.0f}%")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_popularity_shift.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_popularity_shift.png", bbox_inches="tight")
    plt.close(fig)

    runs = pd.read_csv(ARTIFACT_DIR / "summaries" / "all_runs.csv")
    depth = runs[runs["model"].isin(["zero_hop", "lightgcn_k1", "lightgcn_k2", "lightgcn_k3"])].copy()
    fig, ax = plt.subplots(figsize=(4.1, 2.6), dpi=240)
    sns.lineplot(data=depth, x="k_layers", y="ndcg_at_5", marker="o", errorbar="sd", color="#2b6cb0", ax=ax)
    ax.set_xlabel("propagation depth k")
    ax.set_ylabel("random-test ndcg@5")
    ax.set_title("propagation depth improves randomized-test ranking", fontsize=10)
    ax.set_xticks([0, 1, 2, 3])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_depth_sweep.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_depth_sweep.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.7, 3.0), dpi=240)
    scatter = runs.groupby("model", as_index=False).agg(ndcg_at_5=("ndcg_at_5", "mean"), head_item_share=("head_item_share", "mean"), js=("js_recommendation_vs_random_test", "mean"))
    scatter["label"] = scatter["model"].map(
        {
            "zero_hop": "zero-hop",
            "lightgcn_k1": "k=1",
            "lightgcn_k2": "k=2",
            "lightgcn_k3": "k=3",
            "corrected_k2": "corr.",
            "corrected_k2_stronger": "corr.+",
        }
    )
    sns.scatterplot(data=scatter, x="ndcg_at_5", y="head_item_share", size="js", sizes=(55, 165), color="#2f855a", ax=ax, legend=False)
    offsets = {
        "zero-hop": (5, 2),
        "k=1": (16, -7),
        "k=2": (5, -9),
        "k=3": (-34, -12),
        "corr.": (5, -17),
        "corr.+": (18, 8),
    }
    for _, row in scatter.iterrows():
        ax.annotate(row["label"], (row["ndcg_at_5"], row["head_item_share"]), xytext=offsets.get(row["label"], (5, 3)), textcoords="offset points", fontsize=7)
    ax.set_xlabel("random-test ndcg@5")
    ax.set_ylabel("top-5 head-item share")
    ax.set_title("accuracy versus popularity amplification", fontsize=10)
    ax.margins(x=0.08, y=0.12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_accuracy_bias_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_accuracy_bias_tradeoff.png", bbox_inches="tight")
    plt.close(fig)

    per_user = pd.read_csv(ARTIFACT_DIR / "summaries" / "per_user_all.csv")
    a = per_user[per_user["model"] == "lightgcn_k2"].groupby("user_id")["ndcg_at_5"].mean()
    b = per_user[per_user["model"] == "zero_hop"].groupby("user_id")["ndcg_at_5"].mean()
    diff = (a.loc[a.index.intersection(b.index)] - b.loc[a.index.intersection(b.index)]).rename("delta_ndcg")
    fig, ax = plt.subplots(figsize=(4.2, 2.65), dpi=240)
    sns.histplot(diff, bins=24, color="#4a5568", ax=ax)
    ax.axvline(float(diff.mean()), color="#c05621", linewidth=2, label=f"mean = {diff.mean():.3f}")
    ax.set_xlabel("per-user delta ndcg@5: lightgcn k=2 minus zero-hop")
    ax.set_ylabel("users")
    ax.set_title("paired user-level effect distribution", fontsize=10)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_user_effects.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_user_effects.png", bbox_inches="tight")
    plt.close(fig)


def write_latex_tables() -> None:
    summary = pd.read_csv(ARTIFACT_DIR / "dataset_summary.csv")
    main = pd.read_csv(ARTIFACT_DIR / "summaries" / "main_results.csv")
    bootstrap = pd.read_csv(ARTIFACT_DIR / "summaries" / "bootstrap_summary.csv")
    dataset_table = summary.copy()
    for col in ["density", "positive_rate_r_ge_3", "top_1_pct_share", "top_5_pct_share", "top_10_pct_share"]:
        dataset_table[col] = dataset_table[col].map(lambda x: f"{x:.3f}")
    dataset_table.to_latex(TABLE_DIR / "dataset_summary.tex", index=False, escape=False)
    main_table = main[["model", "k_layers", "gamma", "residual", "ndcg_mean", "ndcg_std", "recall_mean", "recall_std", "head_share_mean", "js_mean"]].copy()
    for col in ["ndcg_mean", "ndcg_std", "recall_mean", "recall_std", "head_share_mean", "js_mean"]:
        main_table[col] = main_table[col].map(lambda x: f"{x:.3f}")
    main_table.to_latex(TABLE_DIR / "main_results.tex", index=False, escape=False)
    bootstrap.to_latex(TABLE_DIR / "bootstrap_summary.tex", index=False, float_format="%.3f", escape=False)


def regenerate_report_artifacts_only() -> None:
    data = load_and_cache_data()
    make_figures(data["train"], data["test"], data["summary"])
    write_latex_tables()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run fewer epochs for smoke tests")
    parser.add_argument("--artifacts-only", action="store_true", help="regenerate figures and latex tables from cached results")
    parser.add_argument("--seed-count", type=int, default=3, help="number of random seeds to run; 3 preserves the original seed list")
    parser.add_argument("--seed-start", type=int, default=0, help="first seed when seed-count is not 3")
    parser.add_argument("--no-reuse", action="store_true", help="force retraining even if cached run outputs already exist")
    args = parser.parse_args()
    if args.artifacts_only:
        regenerate_report_artifacts_only()
        print(json.dumps({"status": "ok", "mode": "artifacts-only"}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("cuda is required for this project run")
    device = torch.device("cuda")
    (ARTIFACT_DIR / "summaries").mkdir(parents=True, exist_ok=True)
    data = load_and_cache_data()
    train = data["train"]
    test = data["test"]
    epochs = 60 if args.quick else 180
    seeds = [7, 13, 29] if args.seed_count == 3 else list(range(args.seed_start, args.seed_start + args.seed_count))
    configs: list[RunConfig] = []
    for seed in seeds:
        configs.extend(
            [
                # only k, gamma, and residual vary so the ablation isolates propagation.
                RunConfig("coat", "zero_hop", seed, 0, 0.0, 0.0, epochs=epochs),
                RunConfig("coat", "lightgcn_k1", seed, 1, 0.0, 0.0, epochs=epochs),
                RunConfig("coat", "lightgcn_k2", seed, 2, 0.0, 0.0, epochs=epochs),
                RunConfig("coat", "lightgcn_k3", seed, 3, 0.0, 0.0, epochs=epochs),
                RunConfig("coat", "corrected_k2", seed, 2, 0.35, 0.20, epochs=epochs),
                RunConfig("coat", "corrected_k2_stronger", seed, 2, 0.70, 0.20, epochs=epochs),
            ]
        )
    results = []
    for cfg in configs:
        cached = None if args.no_reuse else load_cached_run(cfg)
        if cached is not None:
            results.append(cached)
        else:
            results.append(train_one(train, test, cfg, device))
    summary, bootstrap = aggregate_and_bootstrap(results)
    make_figures(train, test, data["summary"])
    write_latex_tables()
    with open(ARTIFACT_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"n_runs": len(results), "device": torch.cuda.get_device_name(0), "summary_rows": len(summary), "bootstrap_rows": len(bootstrap)}, f, indent=2)
    print(json.dumps({"status": "ok", "runs": len(results), "device": torch.cuda.get_device_name(0)}, indent=2))


if __name__ == "__main__":
    main()
