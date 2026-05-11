from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RESULTS_DIR = Path("results")
SUMMARY_CSV = RESULTS_DIR / "confusion_matrices_summary_by_mechanism_rate.csv"
OUTPUT_DIR = RESULTS_DIR / "summary_metric_plots"


def load_summary(summary_csv: Path) -> pd.DataFrame:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    df["rate_num"] = df["rate"].str.replace("pct", "", regex=False).astype(int)
    return df


def metric_columns(df: pd.DataFrame) -> list[str]:
    metrics = []
    for column in df.columns:
        if not column.endswith("_mean"):
            continue
        metric_name = column.removesuffix("_mean")
        if metric_name.endswith("both"):
            continue
        if metric_name in {"n_patients", "rate_num"}:
            continue
        metrics.append(metric_name)
    return metrics


def plot_metric(df: pd.DataFrame, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    palette = {"MCAR": "#1f77b4", "MAR": "#ff7f0e", "MNAR": "#2ca02c"}
    markers = {"MCAR": "o", "MAR": "s", "MNAR": "^"}

    for mechanism in sorted(df["mechanism"].unique()):
        mech_data = df[df["mechanism"] == mechanism].sort_values("rate_num")

        if mech_data.empty:
            continue

        ax.errorbar(
            mech_data["rate_num"],
            mech_data[f"{metric}_mean"],
            yerr=mech_data[f"{metric}_std"],
            label=mechanism,
            color=palette.get(mechanism, None),
            marker=markers.get(mechanism, "o"),
            linewidth=2,
            markersize=7,
            capsize=4,
            capthick=1.5,
            alpha=0.85,
        )

    ax.set_title(f"{metric.replace('_', ' ').title()} by Mechanism and Missing Rate")
    ax.set_xlabel("Missing Data Rate (%)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xticks(sorted(df["rate_num"].unique()))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="best")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{metric}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_summary(SUMMARY_CSV)
    sns.set_theme(style="whitegrid")

    for metric in metric_columns(df):
        plot_metric(df, metric)
        print(f"Saved plot for {metric} -> {OUTPUT_DIR / f'{metric}.png'}")


if __name__ == "__main__":
    main()