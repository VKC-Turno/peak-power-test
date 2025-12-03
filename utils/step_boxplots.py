from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_SUMMARY = Path("/home/kcv/Desktop/peak_power_test/post_processing/data/eda_summary.csv")


def _normalize_palette(values: pd.Series) -> dict:
    """Return {step_no: color} mapping using Viridis based on step mean values."""
    if values.empty:
        return {}
    if values.max() == values.min():
        normalized = pd.Series(0.5, index=values.index)
    else:
        normalized = (values - values.min()) / (values.max() - values.min())
    cmap = plt.colormaps["viridis"]
    return {step: cmap(val) for step, val in normalized.items()}


def _plot_axes(ax, subset: pd.DataFrame, value_col: str, title: str) -> None:
    if subset.empty:
        ax.set_visible(False)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    palette = _normalize_palette(subset.groupby("Step No")[value_col].mean())
    sns.boxplot(
        ax=ax,
        data=subset,
        x="Step name",
        y=value_col,
        hue="Step No",
        palette=palette if palette else "viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("Step name")
    ax.set_ylabel(value_col)
    ax.legend(title="Step No", bbox_to_anchor=(1.02, 1), loc="upper left")


def create_step_boxplots(
    value_col: str,
    output_dir: str | Path,
    summary_path: str | Path = DEFAULT_SUMMARY,
    duration_values: Iterable[int] = (25, 35),
) -> None:
    """
    Generate box plots for charge and discharge steps using a summary CSV.

    Parameters
    ----------
    value_col : str
        Column in the summary CSV to plot on the y-axis (e.g., 'Volt_max').
    output_dir : str | Path
        Directory to write image files.
    summary_path : str | Path
        Path to the summary CSV.
    duration_values : Iterable[int]
        Inclusive duration range in seconds (min, max). Defaults to (25, 35).
    """

    summary_path = Path(summary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    if value_col not in df.columns:
        raise KeyError(f"{value_col} not found in {summary_path}")

    duration_values = tuple(duration_values)
    if len(duration_values) != 2:
        raise ValueError("duration_values must be an iterable with two entries: (min_seconds, max_seconds)")
    min_duration, max_duration = duration_values

    df["duration_s"] = df["Duration_s"]
    df = df.loc[df["duration_s"].between(min_duration, max_duration)].copy()
    df = df.assign(cell_id=lambda d: d["Cell"].str.extract(r"_(\d{4})"))
    df = df[~df["Step name"].str.contains("rest", case=False, na=False)]

    charge_mask = df["Step name"].str.contains("Chg", case=False) & ~df["Step name"].str.contains("DChg", case=False)
    discharge_mask = df["Step name"].str.contains("DChg", case=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    _plot_axes(axes[0], df.loc[charge_mask], value_col, f"{value_col} — Charge Steps")
    _plot_axes(axes[1], df.loc[discharge_mask], value_col, f"{value_col} — Discharge Steps")

    plt.tight_layout()
    output_path = output_dir / f"{value_col}_charge_discharge_boxplot.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved combined boxplots to {output_path}")

    # Heatmap
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=False)
    for ax, (mask, title) in zip(
        axes,
        [
            (charge_mask, f"{value_col} Heatmap — Charge ({min_duration}-{max_duration} s)"),
            (discharge_mask, f"{value_col} Heatmap — Discharge ({min_duration}-{max_duration} s)"),
        ],
    ):
        subset = df.loc[mask]
        if subset.empty:
            ax.set_visible(False)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        pivot = (
            subset.pivot_table(
                index="cell_id",
                columns="Step No",
                values=value_col,
                aggfunc="max",
            )
            .sort_index()
        )

        sns.heatmap(pivot, annot=False, cmap="viridis", cbar_kws={"label": value_col}, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Step No")
        ax.set_ylabel("Cell ID")

    plt.tight_layout()
    heatmap_path = output_dir / f"{value_col}_charge_discharge_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close(fig)
    print(f"Saved combined heatmap to {heatmap_path}")


if __name__ == "__main__":
    create_step_boxplots(
        value_col="Volt_max",
        output_dir="/home/kcv/Desktop/peak_power_test/post_processing/plots",
    )
