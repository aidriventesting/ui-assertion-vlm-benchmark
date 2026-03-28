#!/usr/bin/env python3
"""
Generate dataset insight figures (tags, cognitive levels) and a slides-ready
markdown report. Output: docs/figures/*.png and docs/figures/insights_slides.md
"""
import argparse
import sys
from pathlib import Path

# Allow importing from scripts/ when run as script
_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from analyze_dataset import DATASET_DIR, DIFFICULTY_TAGS, compute_dataset_stats


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_tags_distribution(stats, out_dir: Path, fmt: str = "png", dpi: int = 300):
    """Bar chart: tag counts (top N), sorted by count."""
    plt = setup_matplotlib()
    sorted_tags = sorted(stats["tags"].items(), key=lambda x: x[1], reverse=True)
    top_n = min(20, len(sorted_tags))
    labels = [t for t, _ in sorted_tags[:top_n]]
    counts = [c for _, c in sorted_tags[:top_n]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), counts, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Tag distribution (top {})".format(top_n))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tags_distribution.{}".format(fmt)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_cognitive_levels(stats, out_dir: Path, fmt: str = "png", dpi: int = 300):
    """Pie chart: L1 / L2 / L3."""
    plt = setup_matplotlib()
    levels = list(stats["levels"].keys())
    counts = [stats["levels"][l] for l in levels]
    total = sum(counts)
    pct = [100 * c / total for c in counts] if total else counts
    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=levels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ecc71", "#3498db", "#9b59b6"],
    )
    for t in texts:
        t.set_fontsize(10)
    ax.set_title("Cognitive levels (L1/L2/L3)")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cognitive_levels.{}".format(fmt)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_outcomes_balance(stats, out_dir: Path, fmt: str = "png", dpi: int = 300):
    """Bar chart: PASS vs FAIL counts."""
    plt = setup_matplotlib()
    outcomes = ["PASS", "FAIL"]
    counts = [stats["outcomes"].get(o, 0) for o in outcomes]
    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(outcomes, counts, color=["#27ae60", "#e74c3c"], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Count")
    ax.set_title("Outcome balance (PASS / FAIL)")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, str(c), ha="center", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "outcomes_balance.{}".format(fmt)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def write_insights_slides(stats, out_dir: Path):
    """Write insights_slides.md with summary, top tags, L1/L2/L3, and 2–3 insight sentences."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total_tests = stats["total_tests"]
    total_shots = stats["total_shots"]
    pass_count = stats["outcomes"].get("PASS", 0)
    fail_count = stats["outcomes"].get("FAIL", 0)
    sorted_tags = sorted(stats["tags"].items(), key=lambda x: x[1], reverse=True)
    top_tags = sorted_tags[:10]
    levels = stats["levels"]
    level_total = sum(levels.values()) or 1
    # Insight sentences
    dominant = top_tags[0][0] if top_tags else "—"
    l2_l3_pct = 100 * (levels.get("L2 (Spatial/Multi)", 0) + levels.get("L3 (Reasoning)", 0)) / level_total
    difficulty_tags = [(t, c) for t, c in sorted_tags if t in DIFFICULTY_TAGS]
    top_difficulty = difficulty_tags[0][0] if difficulty_tags else "—"
    lines = [
        "# Dataset insights (for slides)",
        "",
        "## Summary",
        "- **Total screenshots**: {}".format(total_shots),
        "- **Total test cases**: {}".format(total_tests),
        "- **Balance**: {} PASS / {} FAIL".format(pass_count, fail_count),
        "",
        "## Top 10 tags",
        "| Tag | Count | % |",
        "|-----|-------|---|",
    ]
    for tag, count in top_tags:
        pct = 100 * count / total_tests if total_tests else 0
        lines.append("| {} | {} | {:.1f}% |".format(tag, count, pct))
    lines.extend([
        "",
        "## Cognitive levels (L1 / L2 / L3)",
        "| Level | Count | % |",
        "|-------|-------|---|",
    ])
    for level in ["L1 (Perception)", "L2 (Spatial/Multi)", "L3 (Reasoning)"]:
        c = levels.get(level, 0)
        pct = 100 * c / level_total
        lines.append("| {} | {} | {:.1f}% |".format(level, c, pct))
    lines.extend([
        "",
        "## Insights (copy-paste for slides)",
        "- Presence and text-matching tags dominate; top tag is **{}**.".format(dominant),
        "- **{:.0f}%** of tests are L2 or L3 (spatial/multi-step or reasoning).".format(l2_l3_pct),
        "- Most frequent difficulty tag: **{}**.".format(top_difficulty),
        "",
    ])
    path = out_dir / "insights_slides.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Plot dataset insights for slides.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "docs" / "figures",
        help="Output directory for figures and insights_slides.md",
    )
    parser.add_argument("--svg", action="store_true", help="Also export figures as SVG")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG (default 300)")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset screenshots root (default: dataset/screenshots)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    dataset_dir = args.dataset_dir or DATASET_DIR
    stats = compute_dataset_stats(dataset_dir)
    formats = ["png"]
    if args.svg:
        formats.append("svg")
    for fmt in formats:
        plot_tags_distribution(stats, out_dir, fmt=fmt, dpi=args.dpi)
        plot_cognitive_levels(stats, out_dir, fmt=fmt, dpi=args.dpi)
        plot_outcomes_balance(stats, out_dir, fmt=fmt, dpi=args.dpi)
    write_insights_slides(stats, out_dir)
    print("Figures and report written to:", out_dir)
    print("  - tags_distribution.png", "(+ .svg)" if args.svg else "")
    print("  - cognitive_levels.png", "(+ .svg)" if args.svg else "")
    print("  - outcomes_balance.png", "(+ .svg)" if args.svg else "")
    print("  - insights_slides.md")


if __name__ == "__main__":
    main()
