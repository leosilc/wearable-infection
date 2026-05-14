from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RESULTS_DIR = Path("results")
INPUT_CSV = RESULTS_DIR / "confusion_matrices_by_patient.csv"
OUTPUT_DIR = RESULTS_DIR / "patient_metric_boxplots"

METADATA_COLUMNS = {"patient", "mechanism", "rate", "n_runs"}
MATRIX_COLUMNS = {
    "green_green",
    "green_yellow",
    "green_red",
    "yellow_green",
    "yellow_yellow",
    "yellow_red",
    "red_green",
    "red_yellow",
    "red_red",
}

PALETTE = {
    "MCAR": "#1f77b4",
    "MAR": "#ff7f0e",
    "MNAR": "#0173b2",
}


def load_patient_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Patient-level CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["rate_num"] = df["rate"].str.replace("pct", "", regex=False).astype(int)
    df["rate_label"] = df["rate_num"].astype(str) + "%"
    return df


def metric_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return alert and warning metrics organized by type."""
    all_metrics = {
        "alert": [],
        "warning": [],
    }
    
    alert_metrics = ["accuracy_alert", "precision_alert", "recall_alert"]
    warning_metrics = ["accuracy_warning", "precision_warning", "recall_warning"]
    
    all_metrics["alert"] = [m for m in alert_metrics if m in df.columns]
    all_metrics["warning"] = [m for m in warning_metrics if m in df.columns]
    
    return all_metrics


def plot_metrics_group(df: pd.DataFrame, metrics: list[str], group_type: str) -> None:
    """Plot a group of metrics (alert or warning) side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    rate_order = sorted(df["rate_num"].unique())
    rate_labels = [f"{rate}%" for rate in rate_order]
    mechanism_order = ["MCAR", "MAR", "MNAR"]
    rate_palette = sns.color_palette("Blues", n_colors=len(rate_labels))

    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        sns.boxplot(
            data=df,
            x="mechanism",
            y=metric,
            hue="rate_label",
            order=mechanism_order,
            hue_order=rate_labels,
            palette=rate_palette,
            width=0.72,
            fliersize=2,
            linewidth=1.2,
            ax=ax,
        )
        
        ax.set_xlabel("Mechanism")
        metric_label = metric.replace(f"_{group_type}", "").replace("_", " ").title()
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.get_legend().remove()

    # Create single legend for all plots
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Missing Rate",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(rate_labels),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.4,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{group_type}_metrics.png"
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_patient_metrics(INPUT_CSV)
    sns.set_theme(style="whitegrid", context="talk")

    metrics_by_type = metric_columns(df)
    
    for group_type, metrics in metrics_by_type.items():
        if metrics:
            plot_metrics_group(df, metrics, group_type)
            print(f"Saved {group_type} metrics plot -> {OUTPUT_DIR / f'{group_type}_metrics.png'}")


if __name__ == "__main__":
    main()