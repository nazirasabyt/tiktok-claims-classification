import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np

# ── Global theme ──────────────────────────────────────────────────────────────

BG      = "#f7f7f8"

GRID    = "#e5e7eb"
TEXT    = "#4b5563"
SUBTEXT = "#9ca3af"

# ── Bubble / category colors ─────────────────────────────────────────────────

MINT_GREEN    = "#8ccfb9"
SOFT_ORANGE   = "#faa78a"
DUSTY_BLUE    = "#aeb7d6"
SOFT_PINK     = "#dca2cb"
LIGHT_LIME    = "#b6d27a"
PASTEL_YELLOW = "#fee16b"

# ── Core comparison colors ────────────────────────────────────────────────────

CLAIM   = DUSTY_BLUE
OPINION = SOFT_ORANGE

# ── Supporting palette ────────────────────────────────────────────────────────

SUCCESS    = MINT_GREEN
NEUTRAL    = SOFT_PINK

DUR_COLORS = [PASTEL_YELLOW, LIGHT_LIME, DUSTY_BLUE]   # short / medium / long

HEATMAP_CMAP = LinearSegmentedColormap.from_list("pastel_corr", [BG, DUSTY_BLUE])

plt.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     BG,
    "axes.edgecolor":     "none",
    "axes.labelcolor":    SUBTEXT,
    "axes.labelsize":     9,
    "axes.titlecolor":    TEXT,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.titlepad":      14,
    "axes.axisbelow":     True,
    "axes.grid":          True,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "grid.color":         GRID,
    "grid.alpha":         1.0,
    "grid.linewidth":     0.7,
    "grid.linestyle":     "--",
    "xtick.color":        SUBTEXT,
    "ytick.color":        SUBTEXT,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "xtick.bottom":       False,
    "ytick.left":         False,
    "legend.facecolor":   "none",
    "legend.edgecolor":   "none",
    "legend.labelcolor":  TEXT,
    "legend.fontsize":    8.5,
    "legend.framealpha":  0,
    "text.color":         TEXT,
    "font.family":        "sans-serif",
})

BAR_KW = {"alpha": 0.75, "edgecolor": "white", "linewidth": 1.5, "zorder": 3}


def fmt_k(v):
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.0f}k"
    return f"{v:.4f}" if v < 0.01 else f"{v:.3f}"


def add_bar_labels(ax, fmt=None, color=SUBTEXT, fontsize=8, padding=3):
    for bar in ax.patches:
        h = bar.get_height()
        if h == 0:
            continue
        label = fmt(h) if fmt else fmt_k(h)
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + padding,
            label, ha="center", va="bottom",
            color=color, fontsize=fontsize, fontweight="bold",
        )


def clean_ax(ax):
    ax.tick_params(length=0)


def generate_visualizations(df, out="reports/figures"):
    data = df.copy()

    data["likes_per_view"]    = data["video_like_count"]    / data["video_view_count"]
    data["comments_per_view"] = data["video_comment_count"] / data["video_view_count"]
    data["shares_per_view"]   = data["video_share_count"]   / data["video_view_count"]
    data["duration_category"] = pd.cut(
        data["video_duration_sec"],
        bins=[0, 20, 40, float("inf")],
        labels=["Short\n(≤20s)", "Medium\n(21–40s)", "Long\n(>40s)"],
    )

    metrics       = ["video_view_count", "video_like_count", "video_share_count", "video_comment_count"]
    metric_labels = ["Views", "Likes", "Shares", "Comments"]
    ratio_cols    = ["likes_per_view", "comments_per_view", "shares_per_view"]
    w = 0.35

    # ── Figure 1 — Main Dashboard (2×3) ──────────────────────────────────────

    fig = plt.figure(figsize=(20, 12), facecolor=BG)
    fig.suptitle(
        "TikTok Claims Classification — Engagement Dashboard",
        fontsize=18, fontweight="bold", color=TEXT, y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38,
                           top=0.91, bottom=0.08, left=0.06, right=0.97)

    ax1 = fig.add_subplot(gs[0, 0])
    means = data.groupby("claim_status")[metrics].mean()
    x = np.arange(len(metric_labels))
    ax1.bar(x - w/2, means.loc["claim"],   w, color=CLAIM,   label="Claim",   **BAR_KW)
    ax1.bar(x + w/2, means.loc["opinion"], w, color=OPINION, label="Opinion", **BAR_KW)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels)
    ax1.set_title("Avg Engagement: Claims vs Opinions")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_k(v)))
    ax1.legend()
    add_bar_labels(ax1, fmt=fmt_k, padding=500)
    clean_ax(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    groups = [
        data[data["claim_status"] == "claim"]["video_view_count"].dropna(),
        data[data["claim_status"] == "opinion"]["video_view_count"].dropna(),
    ]
    bp = ax2.boxplot(
        groups, patch_artist=True, widths=0.45, zorder=3,
        medianprops={"color": SUBTEXT, "linewidth": 1.5},
        whiskerprops={"color": GRID, "linewidth": 1.2},
        capprops={"color": GRID, "linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.2,
                    "markerfacecolor": SUBTEXT, "markeredgecolor": "none"},
    )
    bp["boxes"][0].set_facecolor(CLAIM);   bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor(OPINION); bp["boxes"][1].set_alpha(0.75)
    for box in bp["boxes"]:
        box.set_linewidth(0)
    ax2.set_xticklabels(["Claim", "Opinion"])
    ax2.set_title("View Count Distribution")
    ax2.set_ylabel("Views", color=SUBTEXT)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1000:.0f}k"
    ))
    clean_ax(ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    ratio_labels = ["Likes/View", "Comments/View", "Shares/View"]
    r_means = data.groupby("claim_status")[ratio_cols].mean()
    xr = np.arange(len(ratio_labels))
    ax3.bar(xr - w/2, r_means.loc["claim"],   w, color=CLAIM,   label="Claim",   **BAR_KW)
    ax3.bar(xr + w/2, r_means.loc["opinion"], w, color=OPINION, label="Opinion", **BAR_KW)
    ax3.set_xticks(xr)
    ax3.set_xticklabels(ratio_labels)
    ax3.set_title("Engagement Ratios: Claims vs Opinions")
    ax3.legend()
    add_bar_labels(ax3, fmt=lambda v: f"{v:.3f}", padding=0.002)
    clean_ax(ax3)

    ax4 = fig.add_subplot(gs[1, 0])
    ban_counts = data.groupby(["author_ban_status", "claim_status"]).size().unstack(fill_value=0)
    ban_x = np.arange(len(ban_counts))
    ax4.bar(ban_x - w/2, ban_counts["claim"],   w, color=CLAIM,   label="Claim",   **BAR_KW)
    ax4.bar(ban_x + w/2, ban_counts["opinion"], w, color=OPINION, label="Opinion", **BAR_KW)
    ax4.set_xticks(ban_x)
    ax4.set_xticklabels([s.title() for s in ban_counts.index], fontsize=9)
    ax4.set_title("Video Count by Author Ban Status")
    ax4.legend()
    add_bar_labels(ax4, padding=30)
    clean_ax(ax4)

    ax5 = fig.add_subplot(gs[1, 1])
    med_shares = data.groupby(["author_ban_status", "claim_status"])["video_share_count"].median().unstack()
    ax5.bar(ban_x - w/2, med_shares["claim"],   w, color=CLAIM,   label="Claim",   **BAR_KW)
    ax5.bar(ban_x + w/2, med_shares["opinion"], w, color=OPINION, label="Opinion", **BAR_KW)
    ax5.set_xticks(ban_x)
    ax5.set_xticklabels([s.title() for s in med_shares.index], fontsize=9)
    ax5.set_title("Median Shares by Author Ban Status")
    ax5.legend()
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_k(v)))
    add_bar_labels(ax5, fmt=fmt_k, padding=100)
    clean_ax(ax5)

    ax6 = fig.add_subplot(gs[1, 2])
    corr = data[metrics].corr()
    corr.index = corr.columns = metric_labels
    sns.heatmap(
        corr, ax=ax6, annot=True, fmt=".2f",
        cmap=HEATMAP_CMAP,
        vmin=0.5, vmax=1, linewidths=1.5, linecolor=BG,
        annot_kws={"size": 10, "weight": "bold", "color": TEXT},
        cbar_kws={"shrink": 0.8},
    )
    ax6.set_title("Engagement Correlation Matrix")
    ax6.tick_params(axis="x", rotation=30, labelsize=9, length=0)
    ax6.tick_params(axis="y", rotation=0,  labelsize=9, length=0)
    ax6.collections[0].colorbar.ax.yaxis.set_tick_params(color=SUBTEXT, labelcolor=SUBTEXT, length=0)
    ax6.collections[0].colorbar.outline.set_visible(False)

    fig.savefig(f"{out}/dashboard_overview.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  Saved: dashboard_overview.png")

    # ── Figure 2 — Engagement Ratios Deep-Dive ────────────────────────────────

    fig2, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    fig2.suptitle(
        "Engagement Ratios by Claim Status × Author Ban Status",
        fontsize=15, fontweight="bold", color=TEXT, y=1.02,
    )

    ratio_info = [
        ("likes_per_view",    "Likes per View",    "How much liking per view"),
        ("comments_per_view", "Comments per View", "Discussion intensity per view"),
        ("shares_per_view",   "Shares per View",   "Virality per view"),
    ]

    for ax, (col, title, subtitle) in zip(axes, ratio_info):
        pivot = data.groupby(["author_ban_status", "claim_status"])[col].median().unstack()
        bx = np.arange(len(pivot))
        ax.bar(bx - w/2, pivot["claim"],   w, color=CLAIM,   label="Claim",   **BAR_KW)
        ax.bar(bx + w/2, pivot["opinion"], w, color=OPINION, label="Opinion", **BAR_KW)
        ax.set_xticks(bx)
        ax.set_xticklabels([s.title() for s in pivot.index], fontsize=9)
        ax.set_title(title)
        ax.text(0.5, -0.18, subtitle, transform=ax.transAxes,
                ha="center", fontsize=8, color=SUBTEXT)
        ax.legend()
        add_bar_labels(ax, fmt=lambda v: f"{v:.4f}", padding=0.00005)
        clean_ax(ax)

    fig2.tight_layout()
    fig2.savefig(f"{out}/engagement_ratios.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig2)
    print("  Saved: engagement_ratios.png")

    # ── Figure 3 — Verified vs Not Verified ──────────────────────────────────

    ver_col = data["verified_status"]
    if ver_col.dtype == object:
        ver_numeric = ver_col.map({"verified": 1, "not verified": 0})
    else:
        ver_numeric = ver_col
    data = data.copy()
    data["_ver"] = ver_numeric

    VER_COLORS = {0: NEUTRAL, 1: SUCCESS}
    VER_LABEL  = {0: "Not Verified", 1: "Verified"}

    fig3, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig3.suptitle("Verified vs Not Verified Authors",
                  fontsize=15, fontweight="bold", color=TEXT, y=1.02)

    ver_metrics = data.groupby("_ver")[
        ["video_view_count", "video_like_count", "video_share_count"]
    ].mean()
    bv = np.arange(3)
    for i, (status, color) in enumerate(VER_COLORS.items()):
        offset = (i - 0.5) * w
        axes[0].bar(bv + offset, ver_metrics.loc[status], w,
                    color=color, label=VER_LABEL[status], **BAR_KW)
    axes[0].set_xticks(bv)
    axes[0].set_xticklabels(["Views", "Likes", "Shares"])
    axes[0].set_title("Avg Engagement by Verification Status")
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_k(v)))
    add_bar_labels(axes[0], fmt=fmt_k, padding=500)
    clean_ax(axes[0])

    ver_ratios = data.groupby("_ver")[ratio_cols].mean()
    br = np.arange(3)
    for i, (status, color) in enumerate(VER_COLORS.items()):
        offset = (i - 0.5) * w
        axes[1].bar(br + offset, ver_ratios.loc[status], w,
                    color=color, label=VER_LABEL[status], **BAR_KW)
    axes[1].set_xticks(br)
    axes[1].set_xticklabels(["Likes/View", "Comments/View", "Shares/View"])
    axes[1].set_title("Avg Engagement Ratios by Verification Status")
    axes[1].legend()
    add_bar_labels(axes[1], fmt=lambda v: f"{v:.4f}", padding=0.00005)
    clean_ax(axes[1])

    fig3.tight_layout()
    fig3.savefig(f"{out}/verified_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig3)
    print("  Saved: verified_comparison.png")

    # ── Figure 4 — Duration Analysis ─────────────────────────────────────────

    dur_labels = ["Short\n(≤20s)", "Medium\n(21–40s)", "Long\n(>40s)"]

    fig4, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
    fig4.suptitle("Video Duration vs Engagement",
                  fontsize=15, fontweight="bold", color=TEXT, y=1.02)

    dur_agg = data.groupby("duration_category", observed=True).agg(
        avg_views= ("video_view_count",  "mean"),
        avg_shares=("video_share_count", "mean"),
        avg_likes= ("video_like_count",  "mean"),
    ).reset_index()

    for ax, (col, title, subtitle) in zip(axes, [
        ("avg_views",  "Avg Views",  "Total reach by duration"),
        ("avg_likes",  "Avg Likes",  "Audience approval by duration"),
        ("avg_shares", "Avg Shares", "Virality by duration"),
    ]):
        ax.bar(np.arange(3), dur_agg[col], color=DUR_COLORS, width=0.55, **BAR_KW)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(dur_labels, fontsize=9)
        ax.set_title(title)
        ax.text(0.5, -0.18, subtitle, transform=ax.transAxes,
                ha="center", fontsize=8, color=SUBTEXT)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_k(v)))
        ymin = dur_agg[col].min() * 0.95
        ax.set_ylim(bottom=ymin)
        add_bar_labels(ax, fmt=fmt_k, padding=(dur_agg[col].max() - ymin) * 0.015)
        clean_ax(ax)

    fig4.tight_layout()
    fig4.savefig(f"{out}/duration_analysis.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig4)
    print("  Saved: duration_analysis.png")

    print(f"\n  All figures saved to {out}/")


if __name__ == "__main__":
    df = pd.read_csv("data/raw/tiktok_dataset.csv")
    generate_visualizations(df)
