#!/usr/bin/env python3
"""Generate all publication-quality figures for the final report.

Run from reproduceable_workspace/:
    python3 scripts/make_report_figures.py
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Palette / style ──────────────────────────────────────────────────────────
PALETTE = {
    "zero_hop": "#6b7280",
    "lightgcn_k1": "#93c5fd",
    "lightgcn_k2": "#2563eb",
    "lightgcn_k3": "#1e3a8a",
    "corrected_k2": "#f97316",
    "corrected_k2_stronger": "#b45309",
}
LABEL_SHORT = {
    "zero_hop": "Zero-hop",
    "lightgcn_k1": "$K=1$",
    "lightgcn_k2": "$K=2$",
    "lightgcn_k3": "$K=3$",
    "corrected_k2": "Corr. A",
    "corrected_k2_stronger": "Corr. B",
}

sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)


def savefig(fig, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", dpi=240)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=240)
    plt.close(fig)
    print(f"  saved {name}")


# ── Fig 1: Popularity shift (train vs random test) ───────────────────────────
def fig1_popularity_shift() -> None:
    train = np.load(ARTIFACTS / "data_cache" / "coat_train.npy")
    test = np.load(ARTIFACTS / "data_cache" / "coat_test.npy")
    train_deg = (train > 0).sum(axis=0)
    test_deg = (test > 0).sum(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), dpi=240)

    # Panel a: log-log rank curve
    ax = axes[0]
    ranks = np.arange(1, len(train_deg) + 1)
    ax.plot(ranks, np.sort(train_deg)[::-1], label="MNAR train",
            color="#2563eb", linewidth=2, solid_capstyle="round")
    ax.plot(ranks, np.sort(test_deg)[::-1], label="Random test",
            color="#f97316", linewidth=2, linestyle="--", solid_capstyle="round")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Item popularity rank", fontsize=8)
    ax.set_ylabel("Item degree (log scale)", fontsize=8)
    ax.set_title("(a) Item degree rank curves", fontsize=9, fontweight="bold")
    ax.legend(frameon=True, fontsize=7, loc="upper right", framealpha=0.9)
    ax.grid(axis="both", alpha=0.3)

    # Panel b: head-item concentration bars
    ax = axes[1]
    summary = pd.read_csv(ARTIFACTS / "dataset_summary.csv")
    plot = summary.melt(
        id_vars=["split"],
        value_vars=["top_1_pct_share", "top_5_pct_share", "top_10_pct_share"],
        var_name="bucket", value_name="share",
    )
    plot["bucket"] = plot["bucket"].map({
        "top_1_pct_share": "Top 1%", "top_5_pct_share": "Top 5%",
        "top_10_pct_share": "Top 10%",
    })
    plot["split"] = plot["split"].map({"train_mnar": "MNAR train", "random_test": "Random test"})
    bar_pal = {"MNAR train": "#2563eb", "Random test": "#f97316"}
    sns.barplot(data=plot, x="bucket", y="share", hue="split", palette=bar_pal,
                ax=ax, width=0.6)
    ax.set_xlabel("", fontsize=8)
    ax.set_ylabel("Share of interactions", fontsize=8)
    ax.set_title("(b) Head-item concentration", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, frameon=True, loc="upper left", framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{100*x:.0f}%"))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(w_pad=1.5)
    savefig(fig, "fig1_popularity_shift")


# ── Fig 2: Depth sweep (2-panel: NDCG + Recall) ─────────────────────────────
def fig2_depth_sweep() -> None:
    runs = pd.read_csv(ARTIFACTS / "summaries" / "all_runs.csv")
    # the depth plot excludes corrected variants because rq1 is only about vanilla propagation.
    depth = runs[runs["model"].isin(
        ["zero_hop", "lightgcn_k1", "lightgcn_k2", "lightgcn_k3"])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.9), dpi=240, sharey=False)

    for ax, metric, ylabel, title_tag in [
        (axes[0], "ndcg_at_5", "NDCG@5", "(a) NDCG@5 vs propagation depth"),
        (axes[1], "recall_at_5", "Recall@5", "(b) Recall@5 vs propagation depth"),
    ]:
        grp = depth.groupby("k_layers")[metric]
        means = grp.mean()
        stds = grp.std()
        ks = sorted(means.index)
        mu = [means[k] for k in ks]
        sg = [stds[k] for k in ks]

        ax.fill_between(ks,
                        [m - s for m, s in zip(mu, sg)],
                        [m + s for m, s in zip(mu, sg)],
                        alpha=0.12, color="#2563eb")
        ax.errorbar(ks, mu, yerr=sg,
                    color="#2563eb", marker="o", markersize=5.5, linewidth=2.2,
                    capsize=4, capthick=1.5, elinewidth=1.5, zorder=5)

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["$K=0$\n(zero-hop)", "$K=1$", "$K=2$", "$K=3$"], fontsize=7.5)
        ax.set_xlim(-0.25, 3.25)
        ax.set_xlabel("Propagation depth", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(title_tag, fontsize=9, fontweight="bold", pad=8)
        ax.grid(axis="y", alpha=0.3)

        y_min = min(m - s for m, s in zip(mu, sg))
        y_max = max(m + s for m, s in zip(mu, sg))
        y_range = y_max - y_min
        label_pad = max(y_range * 0.055, 0.0015)
        ax.set_ylim(y_min - y_range * 0.10, y_max + label_pad * 4.0)
        for k, m, s in zip(ks, mu, sg):
            ax.text(k, m + s + label_pad, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=7, color="#1e3a8a",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85),
                    clip_on=True)

    fig.tight_layout(w_pad=1.8)
    savefig(fig, "fig2_depth_sweep")


# ── Fig 3: Bootstrap CIs (RQ1 + RQ2 summary) ─────────────────────────────────
def fig3_bootstrap_ci() -> None:
    boot = pd.read_csv(ARTIFACTS / "summaries" / "bootstrap_summary.csv")

    # separate rq1 and rq2 so positive propagation effects are visually distinct from correction losses.
    # Reorder: RQ1 comparisons first (positive, top of chart), RQ2 below
    rq1_rows = boot[boot["comparison"].str.contains("minus zero_hop|k3 minus k2", regex=True)]
    rq2_rows = boot[boot["comparison"].str.contains("corrected", regex=False)]
    # Sort RQ1 by effect size descending so largest is at top
    rq1_sorted = rq1_rows.sort_values("mean_delta_ndcg_at_5", ascending=True)
    rq2_sorted = rq2_rows.sort_values("mean_delta_ndcg_at_5", ascending=True)
    boot = pd.concat([rq2_sorted, rq1_sorted], ignore_index=True)

    label_map = {
        "lightgcn_k1 minus zero_hop": r"$K\!=\!1$ vs zero-hop",
        "lightgcn_k2 minus zero_hop": r"$K\!=\!2$ vs zero-hop",
        "lightgcn_k3 minus zero_hop": r"$K\!=\!3$ vs zero-hop",
        "lightgcn_k3 minus lightgcn_k2": r"$K\!=\!3$ vs $K\!=\!2$",
        "corrected_k2 minus lightgcn_k2": r"Corr. A vs $K\!=\!2$",
        "corrected_k2_stronger minus lightgcn_k2": r"Corr. B vs $K\!=\!2$",
    }
    boot["label"] = boot["comparison"].map(label_map)
    boot["color"] = boot["mean_delta_ndcg_at_5"].apply(
        lambda x: "#2563eb" if x > 0 else "#dc2626")
    boot["is_rq2"] = boot["comparison"].str.contains("corrected")

    n = len(boot)
    fig, ax = plt.subplots(figsize=(6.2, 3.5), dpi=240)

    # Shade the RQ2 region (bottom 2 rows = indices 0,1)
    n_rq2 = boot["is_rq2"].sum()
    ax.axhspan(-0.5, n_rq2 - 0.5, color="#fee2e2", alpha=0.35, zorder=0)
    ax.axhspan(n_rq2 - 0.5, n - 0.5, color="#dbeafe", alpha=0.25, zorder=0)

    for i, (_, row) in enumerate(boot.iterrows()):
        color = row["color"]
        # Draw CI line
        ax.plot([row["ci_low"], row["ci_high"]], [i, i],
                color=color, linewidth=2.8, solid_capstyle="round", zorder=3)
        # Draw mean point
        ax.scatter([row["mean_delta_ndcg_at_5"]], [i], color=color,
                   s=60, zorder=5, edgecolors="white", linewidths=0.6)

        # Put every value to the right of its interval so it cannot collide with y labels.
        val = row["mean_delta_ndcg_at_5"]
        x_txt = row["ci_high"] + 0.0007
        ax.text(x_txt, i, f"{val:+.4f}",
                ha="left", va="center", fontsize=7.5, color=color, fontweight="bold")

    ax.axvline(0, color="#374151", linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)
    ax.set_yticks(list(range(n)))
    ax.set_yticklabels(boot["label"].tolist(), fontsize=8.5)
    ax.set_xlabel(r"Paired $\Delta$NDCG@5  (user-level bootstrap 95% CI,  $N\!=\!281$ users)", fontsize=8)
    ax.set_title("Bootstrap confidence intervals for paired comparisons", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, zorder=1)

    # RQ labels on the right margin inside shaded region
    rq1_center = (n_rq2 - 0.5 + n - 0.5) / 2  # middle of top (blue) band
    rq2_center = (n_rq2 - 0.5 - 0.5) / 2        # middle of bottom (red) band

    # Use axis fraction x for right-aligned labels
    ax.text(1.01, rq1_center / (n - 1), "RQ1", transform=ax.get_yaxis_transform(),
            fontsize=9, color="#1e3a8a", fontweight="bold", va="center", ha="left")
    ax.text(1.01, rq2_center / (n - 1), "RQ2", transform=ax.get_yaxis_transform(),
            fontsize=9, color="#991b1b", fontweight="bold", va="center", ha="left")

    # Separator line between RQ2 and RQ1
    ax.axhline(n_rq2 - 0.5, color="#9ca3af", linewidth=0.9, linestyle=":", zorder=2)

    ax.set_xlim(boot["ci_low"].min() - 0.006, boot["ci_high"].max() + 0.012)
    fig.tight_layout()
    savefig(fig, "fig3_bootstrap_ci")


# ── Fig 4 (was accuracy bias, now removed) – kept as stub ─────────────────────
# User requested removal of accuracy vs popularity amplification figure.
# The function is left here but not called in main().


# ── Fig 5: Causal mechanism diagnostics (3-panel) ─────────────────────────────
def fig5_mechanism() -> None:
    train = np.load(ARTIFACTS / "data_cache" / "coat_train.npy")
    test = np.load(ARTIFACTS / "data_cache" / "coat_test.npy")
    degree = (train > 0).sum(axis=0)

    # panel b uses randomized-test labels to avoid treating observational popularity as relevance.
    # Item-level stats
    item_rows = []
    for item in range(test.shape[1]):
        observed = test[:, item] > 0
        if observed.sum() == 0:
            continue
        item_rows.append({
            "item_id": item,
            "train_degree": int(degree[item]),
            "test_positive_rate": float((test[:, item] >= 3).sum() / observed.sum()),
        })
    item_stats = pd.DataFrame(item_rows)
    item_stats["quartile"] = pd.qcut(
        item_stats.train_degree.rank(method="first"), 4,
        labels=["Q1\n(tail)", "Q2", "Q3", "Q4\n(head)"])
    quartiles = (item_stats.groupby("quartile", observed=False)
                 .agg(random_test_positive_rate=("test_positive_rate", "mean"),
                      n=("item_id", "count"))
                 .reset_index())

    bucket = pd.read_csv(ARTIFACTS / "summaries" / "bucket_recall.csv")
    # these four curves show the baseline, useful propagation, deeper propagation, and correction.
    selected = ["zero_hop", "lightgcn_k2", "lightgcn_k3", "corrected_k2_stronger"]
    label_map = {"zero_hop": "Zero-hop", "lightgcn_k2": "$K=2$",
                 "lightgcn_k3": "$K=3$", "corrected_k2_stronger": "Corr. B"}
    # Use distinct line styles + colors for accessibility
    style_map = {
        "zero_hop":             {"color": "#6b7280", "ls": (0, (4, 2)),    "lw": 1.8, "marker": "s", "ms": 4},
        "lightgcn_k2":          {"color": "#2563eb", "ls": "-",             "lw": 2.0, "marker": "o", "ms": 4},
        "lightgcn_k3":          {"color": "#1e3a8a", "ls": "-",             "lw": 2.0, "marker": "D", "ms": 3.5},
        "corrected_k2_stronger":{"color": "#b45309", "ls": (0, (2, 1.5)),  "lw": 1.8, "marker": "^", "ms": 4},
    }

    summary = pd.read_csv(ARTIFACTS / "dataset_summary.csv")
    plot = summary.melt(
        id_vars=["split"],
        value_vars=["top_1_pct_share", "top_5_pct_share", "top_10_pct_share"],
        var_name="head_bucket", value_name="share",
    )
    plot["head_bucket"] = plot["head_bucket"].map({
        "top_1_pct_share": "Top 1%", "top_5_pct_share": "Top 5%",
        "top_10_pct_share": "Top 10%"})
    plot["split"] = plot["split"].map({
        "train_mnar": "MNAR train", "random_test": "Random test"})

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.55), dpi=240)

    # Panel a: exposure skew
    ax = axes[0]
    bar_pal = {"MNAR train": "#2563eb", "Random test": "#f97316"}
    sns.barplot(data=plot, x="head_bucket", y="share", hue="split",
                palette=bar_pal, ax=ax, width=0.65)
    ax.set_title("(a) Exposure skew", fontsize=9, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Share of interactions", fontsize=7.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{100*x:.0f}%"))
    ax.legend(fontsize=6.5, frameon=True, loc="upper left", framealpha=0.9,
              title="Split", title_fontsize=6)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Panel b: popularity predicts relevance
    ax = axes[1]
    bar_colors = ["#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"]
    bars = ax.bar(quartiles.index, quartiles.random_test_positive_rate,
                  color=bar_colors, width=0.6, edgecolor="white", linewidth=0.5)
    ax.set_xticks(quartiles.index)
    ax.set_xticklabels(quartiles.quartile.astype(str), fontsize=7)
    ax.set_title("(b) Popularity predicts relevance", fontsize=9, fontweight="bold")
    ax.set_xlabel("Train-degree quartile", fontsize=7.5)
    ax.set_ylabel("Rand.-test positive rate", fontsize=7.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{100*x:.0f}%"))
    # Annotate bars above them, with enough clearance
    ymax = quartiles.random_test_positive_rate.max()
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + ymax * 0.015,
                f"{100*h:.1f}%", ha="center", va="bottom", fontsize=7,
                fontweight="bold")
    ax.set_ylim(0, ymax * 1.18)
    ax.grid(axis="y", alpha=0.3)

    # Panel c: bucket recall by model – clean line plot with distinct styles
    ax = axes[2]
    bucket_order = ["tail", "lower-mid", "upper-mid", "head"]
    bucket_plot = bucket[bucket.model.isin(selected)].copy()

    for model in selected:
        sub = bucket_plot[bucket_plot.model == model]
        grp = sub.groupby("bucket")["recall_at_5"]
        means = grp.mean()
        stds = grp.std()
        xs = np.array([bucket_order.index(b) for b in bucket_order])
        ys = np.array([means.get(b, np.nan) for b in bucket_order])
        es = np.array([stds.get(b, np.nan) for b in bucket_order])
        st = style_map[model]
        ax.plot(xs, ys, label=label_map[model],
                color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
                marker=st["marker"], markersize=st["ms"], zorder=4)
        # Thin error shading (no error bars, to reduce clutter)
        ax.fill_between(xs, ys - es, ys + es,
                        alpha=0.10, color=st["color"])

    ax.set_xticks(range(4))
    ax.set_xticklabels(["Tail", "Lower-\nmid", "Upper-\nmid", "Head"],
                       fontsize=7)
    ax.set_title("(c) Recall@5 by item popularity", fontsize=9, fontweight="bold")
    ax.set_xlabel("Item popularity bucket", fontsize=7.5)
    ax.set_ylabel("Recall@5", fontsize=7.5)
    # Legend outside the plot area to avoid overlap
    ax.legend(fontsize=7, frameon=True, loc="upper left", framealpha=0.9,
              handlelength=1.8, labelspacing=0.3)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(w_pad=1.4)
    savefig(fig, "fig5_mechanism")


# ── Fig 6: Per-user effect distribution (appendix) ───────────────────────────
def fig6_user_effects() -> None:
    per_user = pd.read_csv(ARTIFACTS / "summaries" / "per_user_all.csv")
    a = per_user[per_user["model"] == "lightgcn_k2"].groupby("user_id")["ndcg_at_5"].mean()
    b = per_user[per_user["model"] == "zero_hop"].groupby("user_id")["ndcg_at_5"].mean()
    both = a.index.intersection(b.index)
    diff = (a.loc[both] - b.loc[both]).rename("delta_ndcg")

    n_pos = (diff > 0).sum()
    n_neg = (diff < 0).sum()

    fig, ax = plt.subplots(figsize=(4.5, 2.7), dpi=240)
    neg_vals = diff[diff <= 0]
    pos_vals = diff[diff > 0]
    bins = np.linspace(diff.min() - 0.01, diff.max() + 0.01, 22)
    ax.hist(neg_vals, bins=bins, color="#dc2626", alpha=0.70,
            label=f"Hurt ({n_neg})", edgecolor="white", linewidth=0.3)
    ax.hist(pos_vals, bins=bins, color="#2563eb", alpha=0.70,
            label=f"Improved ({n_pos})", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="#374151", linewidth=1.0, linestyle="--")
    ax.axvline(float(diff.mean()), color="#f97316", linewidth=2.0,
               label=f"Mean = {diff.mean():.3f}")
    ax.set_xlabel("Per-user $\\Delta$NDCG@5 ($K=2$ $-$ zero-hop, avg. over 20 seeds)", fontsize=8)
    ax.set_ylabel("Number of users", fontsize=8)
    ax.set_title("Per-user effect distribution ($N=281$ users)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "fig6_user_effects")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    fig1_popularity_shift()
    fig2_depth_sweep()
    fig3_bootstrap_ci()
    # fig4_accuracy_bias removed per user request
    fig5_mechanism()
    fig6_user_effects()
    print("Done.")
