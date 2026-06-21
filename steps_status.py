"""
Overnight steps statistics.

NightSignal only builds the overnight RHR from records where ST_Value == 0
(within 00:00-06:59). Any overnight record with steps != 0 is ignored by the
algorithm. Since the missing-data injection targets the whole overnight window
regardless of steps, part of the "removed" data was never used in the first
place. This script measures, per patient and overall, how much of the overnight
window has steps != 0, so the nominal vs. effective missing rate can be
discussed.

Input : data/processing/{patient_id}/{patient_id}_temp.csv
Output: results/overnight_steps_stats.csv  (+ console summary + optional bar plot)
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROCESSING_DIR = Path("data/processing")
RESULTS_DIR = Path("results")
NIGHT_END_HOUR = 6  # overnight window is hour 0..6 inclusive (00:00-06:59), matching NightSignal


def analyze_patient(patient_id: str, csv_path: Path) -> dict | None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["ST_Value"] = pd.to_numeric(df["ST_Value"], errors="coerce").fillna(0)
    # HR may be missing in degraded files; in processing files it is complete.
    hr_valid = pd.to_numeric(df["HR_Value"], errors="coerce").notna()

    night = df[df["Datetime"].dt.hour <= NIGHT_END_HOUR]
    n_night = len(night)
    if n_night == 0:
        return None

    steps_zero = night["ST_Value"] == 0
    steps_nonzero = ~steps_zero

    # rows actually usable by NightSignal: overnight, steps == 0, valid HR
    usable = steps_zero & hr_valid.loc[night.index]

    return {
        "patient": patient_id,
        "n_overnight": n_night,
        "n_steps_zero": int(steps_zero.sum()),
        "n_steps_nonzero": int(steps_nonzero.sum()),
        "pct_steps_nonzero": round(steps_nonzero.sum() / n_night * 100, 2),
        "n_usable_by_algo": int(usable.sum()),
        "pct_usable_by_algo": round(usable.sum() / n_night * 100, 2),
    }


def save_barplot(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[df["patient"] != "ALL"].sort_values("pct_steps_nonzero", ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(plot_df)), 5))
    ax.bar(plot_df["patient"], plot_df["pct_steps_nonzero"], color="#c1671f")
    ax.set_ylabel("% de registros durante a noite com steps != 0")
    ax.set_xlabel("Patient")
    ax.set_title("registros de noite já ignorados pelo NightSignal")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not PROCESSING_DIR.exists():
        print(f"[ERROR] Folder '{PROCESSING_DIR}' not found. Run run_all.py first.")
        return

    rows = []
    for patient_folder in sorted(p for p in PROCESSING_DIR.iterdir() if p.is_dir()):
        patient_id = patient_folder.name
        csv_path = patient_folder / f"{patient_id}_temp.csv"
        if not csv_path.exists():
            print(f"[WARNING] {csv_path} not found, skipping.")
            continue
        result = analyze_patient(patient_id, csv_path)
        if result:
            rows.append(result)
            print(
                f"{patient_id}: {result['n_overnight']:>7} overnight rows | "
                f"steps!=0: {result['n_steps_nonzero']:>6} ({result['pct_steps_nonzero']:>5.2f}%) | "
                f"usable by algo: {result['pct_usable_by_algo']:>5.2f}%"
            )

    if not rows:
        print("[WARNING] No patient data analyzed.")
        return

    df = pd.DataFrame(rows)

    # Overall aggregate (pooled across all patients, not a mean of percentages)
    total_night = df["n_overnight"].sum()
    total_nonzero = df["n_steps_nonzero"].sum()
    total_usable = df["n_usable_by_algo"].sum()
    overall = {
        "patient": "ALL",
        "n_overnight": int(total_night),
        "n_steps_zero": int(df["n_steps_zero"].sum()),
        "n_steps_nonzero": int(total_nonzero),
        "pct_steps_nonzero": round(total_nonzero / total_night * 100, 2),
        "n_usable_by_algo": int(total_usable),
        "pct_usable_by_algo": round(total_usable / total_night * 100, 2),
    }
    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "overnight_steps_stats.csv"
    df.to_csv(out_csv, index=False)

    save_barplot(df, RESULTS_DIR / "overnight_steps_stats.png")

    print("\n" + "=" * 60)
    print(f"OVERALL: {overall['pct_steps_nonzero']:.2f}% of overnight records have steps != 0")
    print(f"         (these are ignored by NightSignal regardless of missingness)")
    print(f"Saved table -> {out_csv}")
    print(f"Saved plot  -> {RESULTS_DIR / 'overnight_steps_stats.png'}")


if __name__ == "__main__":
    main()