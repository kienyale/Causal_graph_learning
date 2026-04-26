from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"


def load_topk():
    frames = []
    runs = ARTIFACTS / "runs" / "coat"
    for model_dir in runs.iterdir():
        if not model_dir.is_dir():
            continue
        for seed_dir in model_dir.iterdir():
            path = seed_dir / "topk_recommendations.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["model"] = model_dir.name
            df["seed"] = int(seed_dir.name.split("_")[-1])
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def item_bucket(train):
    degrees = (train > 0).sum(axis=0)
    # rank first so equal-degree items are deterministically assigned to buckets
    return np.array(pd.qcut(pd.Series(degrees).rank(method="first"), 4, labels=["tail", "lower-mid", "upper-mid", "head"]).astype(str))


def bucket_recall(test, topk, buckets):
    rows = []
    for (model, seed), df in topk.groupby(["model", "seed"]):
        # convert each user's top-k list to a set so recall measures bucket hits, not rank.
        rec_by_user = {u: set(g.item_id.tolist()) for u, g in df.groupby("user_id")}
        for bucket in ["tail", "lower-mid", "upper-mid", "head"]:
            recalls = []
            for user in range(test.shape[0]):
                positives = set(np.where((test[user] >= 3) & (buckets == bucket))[0])
                if not positives:
                    continue
                hits = len(positives & rec_by_user.get(user, set()))
                recalls.append(hits / len(positives))
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "bucket": bucket,
                    "recall_at_5": float(np.mean(recalls)),
                    "users": len(recalls),
                }
            )
    return pd.DataFrame(rows)


def swap_diagnostics(topk):
    rows = []
    # compare corrected models against vanilla k=2 to isolate what the penalty removed or added.
    base = topk[topk.model == "lightgcn_k2"]
    for corr in ["corrected_k2", "corrected_k2_stronger"]:
        corr_df = topk[topk.model == corr]
        for seed in sorted(set(base.seed) & set(corr_df.seed)):
            b_seed = base[base.seed == seed]
            c_seed = corr_df[corr_df.seed == seed]
            for user in sorted(set(b_seed.user_id) & set(c_seed.user_id)):
                base_items = set(b_seed[b_seed.user_id == user].item_id)
                corr_items = set(c_seed[c_seed.user_id == user].item_id)
                vanilla_only = b_seed[(b_seed.user_id == user) & (~b_seed.item_id.isin(corr_items))]
                corr_only = c_seed[(c_seed.user_id == user) & (~c_seed.item_id.isin(base_items))]
                if len(vanilla_only) == 0 and len(corr_only) == 0:
                    continue
                rows.append(
                    {
                        "comparison": corr,
                        "seed": seed,
                        "user_id": user,
                        "overlap": len(base_items & corr_items),
                        "vanilla_only_label_rate": vanilla_only.label.mean() if len(vanilla_only) else np.nan,
                        "correction_only_label_rate": corr_only.label.mean() if len(corr_only) else np.nan,
                        "vanilla_only_degree": vanilla_only.train_degree.mean() if len(vanilla_only) else np.nan,
                        "correction_only_degree": corr_only.train_degree.mean() if len(corr_only) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def main():
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.82)
    train = np.load(ARTIFACTS / "data_cache" / "coat_train.npy")
    test = np.load(ARTIFACTS / "data_cache" / "coat_test.npy")
    degree = (train > 0).sum(axis=0)
    buckets = item_bucket(train)
    topk = load_topk()

    item_rows = []
    for item in range(test.shape[1]):
        observed = test[:, item] > 0
        if observed.sum() == 0:
            continue
        # item-level random-test relevance tests whether popularity is pure bias or mixed signal.
        item_rows.append(
            {
                "item_id": item,
                "train_degree": degree[item],
                "test_positive_rate": (test[:, item] >= 3).sum() / observed.sum(),
                "test_positive_count": int((test[:, item] >= 3).sum()),
            }
        )
    item_stats = pd.DataFrame(item_rows)
    item_stats["quartile"] = pd.qcut(
        item_stats.train_degree.rank(method="first"),
        4,
        labels=["q1 tail", "q2", "q3", "q4 head"],
    )
    quartiles = (
        item_stats.groupby("quartile", observed=False)
        .agg(
            n_items=("item_id", "count"),
            mean_train_degree=("train_degree", "mean"),
            random_test_positive_rate=("test_positive_rate", "mean"),
        )
        .reset_index()
    )

    bucket = bucket_recall(test, topk, buckets)
    bucket.to_csv(ARTIFACTS / "summaries" / "bucket_recall.csv", index=False)

    # swap summaries expose the correction error mode behind the aggregate ndcg drop.
    swaps = swap_diagnostics(topk)
    swaps.to_csv(ARTIFACTS / "summaries" / "swap_diagnostics.csv", index=False)

    # degree-relevance correlation is needed before interpreting a degree penalty as debiasing.
    spearman = spearmanr(item_stats.train_degree, item_stats.test_positive_rate).statistic
    pearson = pearsonr(item_stats.train_degree, item_stats.test_positive_rate).statistic
    diagnostics = pd.DataFrame(
        [
            {
                "diagnostic": "degree_vs_random_positive_rate_spearman",
                "value": spearman,
            },
            {
                "diagnostic": "degree_vs_random_positive_rate_pearson",
                "value": pearson,
            },
            {
                "diagnostic": "tail_quartile_random_positive_rate",
                "value": quartiles.iloc[0].random_test_positive_rate,
            },
            {
                "diagnostic": "head_quartile_random_positive_rate",
                "value": quartiles.iloc[-1].random_test_positive_rate,
            },
            {
                "diagnostic": "corrected_k2_vanilla_only_label_rate",
                "value": swaps[swaps.comparison == "corrected_k2"].vanilla_only_label_rate.mean(),
            },
            {
                "diagnostic": "corrected_k2_correction_only_label_rate",
                "value": swaps[swaps.comparison == "corrected_k2"].correction_only_label_rate.mean(),
            },
        ]
    )
    diagnostics.to_csv(ARTIFACTS / "summaries" / "mechanism_diagnostics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.25), dpi=240)

    dataset = pd.read_csv(ARTIFACTS / "dataset_summary.csv")
    plot = dataset.melt(
        id_vars=["split"],
        value_vars=["top_1_pct_share", "top_5_pct_share", "top_10_pct_share"],
        var_name="head_bucket",
        value_name="share",
    )
    plot["head_bucket"] = plot["head_bucket"].map(
        {
            "top_1_pct_share": "top 1%",
            "top_5_pct_share": "top 5%",
            "top_10_pct_share": "top 10%",
        }
    )
    plot["split"] = plot["split"].map({"train_mnar": "mnar train", "random_test": "random test"})
    sns.barplot(data=plot, x="head_bucket", y="share", hue="split", ax=axes[0], palette=["#2b6cb0", "#c05621"])
    axes[0].set_title("a. exposure skew", fontsize=9)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("share")
    axes[0].yaxis.set_major_formatter(lambda x, _: f"{100*x:.0f}%")
    axes[0].legend(fontsize=6, frameon=True, loc="upper left")

    sns.barplot(data=quartiles, x="quartile", y="random_test_positive_rate", ax=axes[1], color="#2f855a")
    axes[1].set_title("b. popularity predicts relevance", fontsize=9)
    axes[1].set_xlabel("train-degree quartile")
    axes[1].set_ylabel("positive rate")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{100*x:.0f}%")
    axes[1].tick_params(axis="x", labelrotation=25)

    selected = ["zero_hop", "lightgcn_k2", "lightgcn_k3", "corrected_k2_stronger"]
    labels = {
        "zero_hop": "zero-hop",
        "lightgcn_k2": "k=2",
        "lightgcn_k3": "k=3",
        "corrected_k2_stronger": "corr.+",
    }
    bucket_plot = bucket[bucket.model.isin(selected)].copy()
    bucket_plot["model_label"] = bucket_plot.model.map(labels)
    sns.pointplot(
        data=bucket_plot,
        x="bucket",
        y="recall_at_5",
        hue="model_label",
        order=["tail", "lower-mid", "upper-mid", "head"],
        errorbar="sd",
        ax=axes[2],
        palette=["#4a5568", "#2b6cb0", "#1a365d", "#c05621"],
        markers="o",
        linestyles="-",
    )
    axes[2].set_title("c. recall by item bucket", fontsize=9)
    axes[2].set_xlabel("item popularity")
    axes[2].set_ylabel("recall@5")
    axes[2].tick_params(axis="x", labelrotation=25)
    axes[2].legend(fontsize=6, frameon=True, loc="lower right")

    for ax in axes:
        ax.grid(axis="y", alpha=0.35)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG_DIR / "fig5_causal_mechanism_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_causal_mechanism_diagnostics.png", bbox_inches="tight")
    plt.close(fig)

    swap_summary = (
        swaps.groupby("comparison")
        .agg(
            mean_top5_overlap=("overlap", "mean"),
            changed_items=("overlap", lambda s: 5 - s.mean()),
            vanilla_only_label_rate=("vanilla_only_label_rate", "mean"),
            correction_only_label_rate=("correction_only_label_rate", "mean"),
            vanilla_only_degree=("vanilla_only_degree", "mean"),
            correction_only_degree=("correction_only_degree", "mean"),
        )
        .reset_index()
    )
    swap_summary.to_csv(ARTIFACTS / "summaries" / "swap_summary.csv", index=False)

    with open(TABLE_DIR / "mechanism_diagnostics.tex", "w") as f:
        f.write("\\begin{tabular}{lrr}\\toprule\n")
        f.write("Diagnostic & Value & Interpretation \\\\\\midrule\n")
        f.write(f"Spearman(train degree, random positive rate) & {spearman:.3f} & popularity carries preference signal \\\\\n")
        f.write(f"Tail quartile random positive rate & {quartiles.iloc[0].random_test_positive_rate:.3f} & low-degree items \\\\\n")
        f.write(f"Head quartile random positive rate & {quartiles.iloc[-1].random_test_positive_rate:.3f} & high-degree items \\\\\n")
        f.write(f"Vanilla-only swapped item label rate & {swap_summary.iloc[0].vanilla_only_label_rate:.3f} & removed by correction \\\\\n")
        f.write(f"Correction-only swapped item label rate & {swap_summary.iloc[0].correction_only_label_rate:.3f} & added by correction \\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")

    print({"status": "ok", "spearman": round(float(spearman), 4), "figure": "fig5_causal_mechanism_diagnostics"})


if __name__ == "__main__":
    main()
