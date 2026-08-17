import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def summarize_by_k(
    frame: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    return (
        frame.groupby("k", as_index=False)[value_column]
        .agg(["mean", "std"])
        .reset_index()
        .fillna(0.0)
    )


def plot_with_band(
    ax,
    summary: pd.DataFrame,
    label: str,
    color: str,
) -> None:
    x = summary["k"]
    mean = summary["mean"]
    std = summary["std"]

    ax.plot(x, mean, marker="o", linewidth=2.2, label=label, color=color)
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.16,
        color=color,
        linewidth=0,
    )


def finish_plot(ax, title: str, x_label: str, y_label: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)


def save_figure(fig, output_base: Path, formats: list[str]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        fig.savefig(
            output_base.with_suffix(f".{file_format}"),
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def build_scree_plot(
    input_dir: Path,
    config: str,
    output_dir: Path,
    formats: list[str],
) -> None:
    frame = pd.read_csv(input_dir / f"{config}_scree.csv")
    explained = summarize_by_k(frame, "explained_variance_ratio")
    cumulative = summarize_by_k(frame, "cumulative_explained_variance")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    plot_with_band(
        ax,
        explained,
        label="Single PC explained variance",
        color="#2563eb",
    )
    plot_with_band(
        ax,
        cumulative,
        label="Cumulative explained variance",
        color="#dc2626",
    )
    finish_plot(
        ax,
        title=f"{config}: Scree Plot",
        x_label="Top-k principal components",
        y_label="Explained variance",
    )
    ax.set_ylim(0.0, 1.0)
    save_figure(fig, output_dir / f"{config}_scree", formats)


def build_stability_plot(
    input_dir: Path,
    config: str,
    output_dir: Path,
    formats: list[str],
) -> None:
    frame = pd.read_csv(input_dir / f"{config}_stability.csv")
    stability = summarize_by_k(frame, "subspace_overlap")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    plot_with_band(
        ax,
        stability,
        label="Pairwise round overlap",
        color="#16a34a",
    )
    finish_plot(
        ax,
        title=f"{config}: Cross-Round PCA Stability",
        x_label="Top-k principal components",
        y_label="Mean subspace overlap",
    )
    ax.set_ylim(0.0, 1.0)
    save_figure(fig, output_dir / f"{config}_stability", formats)


def build_random_baseline_plot(
    input_dir: Path,
    config: str,
    output_dir: Path,
    formats: list[str],
) -> None:
    frame = pd.read_csv(input_dir / f"{config}_random_baseline.csv")
    pca = summarize_by_k(frame, "pca_cumulative_explained")
    random_baseline = summarize_by_k(frame, "random_cumulative_explained_mean")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    plot_with_band(
        ax,
        pca,
        label="PCA subspace",
        color="#7c3aed",
    )
    plot_with_band(
        ax,
        random_baseline,
        label="Random subspace",
        color="#f97316",
    )
    finish_plot(
        ax,
        title=f"{config}: PCA vs Random Subspace",
        x_label="Top-k dimensions",
        y_label="Captured update energy",
    )
    ax.set_ylim(0.0, 1.0)
    save_figure(fig, output_dir / f"{config}_random_baseline", formats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", default="./results/pca_validation")
    parser.add_argument("--output-dir", default="./results/pca_validation/plots")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "svg"],
        choices=["png", "svg", "pdf"],
        help="Plot file formats to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    plt.style.use("seaborn-v0_8-whitegrid")

    build_scree_plot(input_dir, args.config, output_dir, args.formats)
    build_stability_plot(input_dir, args.config, output_dir, args.formats)
    build_random_baseline_plot(input_dir, args.config, output_dir, args.formats)

    print(f"Wrote PCA validation plots to {output_dir}")


if __name__ == "__main__":
    main()
