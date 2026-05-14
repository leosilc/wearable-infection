import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RESULTS_DIR = Path("results")


def load_results(json_file: Path) -> dict[str, str]:
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    return {
        " ".join(entry["date"].split()): entry["val"]
        for entry in data["nightsignal"]
    }


def empty_matrix() -> list[list[int]]:
    return [[0 for _ in range(3)] for _ in range(3)]


def add_matrices(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    return [[left[i][j] + right[i][j] for j in range(3)] for i in range(3)]


def matrix_from_results(original: dict[str, str], missing: dict[str, str]) -> list[list[int]]:
    matrix = empty_matrix()
    dates = set(original.keys()) & set(missing.keys())

    for date in dates:
        original_value = int(original[date])
        missing_value = int(missing[date])
        matrix[original_value][missing_value] += 1

    return matrix


def metrics_from_matrix(matrix: list[list[int]]) -> dict[str, float]:
    safe_div = lambda num, den: num / den if den else 0

    return {
        "recall_warning": safe_div(matrix[1][1], sum(matrix[1][j] for j in range(3))),
        "recall_alert": safe_div(matrix[2][2], sum(matrix[2][j] for j in range(3))),
        "precision_warning": safe_div(matrix[1][1], sum(matrix[i][1] for i in range(3))),
        "precision_alert": safe_div(matrix[2][2], sum(matrix[i][2] for i in range(3))),
        "accuracy_alert": safe_div(
            matrix[2][2],
            sum(matrix[i][2] for i in range(3)) + sum(matrix[2][j] for j in range(3)) - matrix[2][2],
        ),
        "accuracy_warning": safe_div(
            matrix[1][1],
            sum(matrix[i][1] for i in range(3)) + sum(matrix[1][j] for j in range(3)) - matrix[1][1],
        ),
    }


def find_run_jsons(patient_folder: Path, mechanism: str, rate: str) -> list[tuple[str, Path]]:
    rate_folder = patient_folder / "missing" / mechanism / rate
    if not rate_folder.exists():
        return []

    direct_json = rate_folder / f"{patient_folder.name}_signals.json"
    if direct_json.exists():
        return [("", direct_json)]

    seed_folders = sorted(
        (folder for folder in rate_folder.iterdir() if folder.is_dir()),
        key=lambda folder: int(folder.name),
    )
    return [
        (seed_folder.name, seed_folder / f"{patient_folder.name}_signals.json")
        for seed_folder in seed_folders
    ]


def save_heatmap(matrix: list[list[int]], title: str, output_path: Path) -> None:
    labels = ["Green (0)", "Yellow (1)", "Red (2)"]
    df = pd.DataFrame(matrix, index=labels, columns=labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("With missing data")
    ax.set_ylabel("Original")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not RESULTS_DIR.exists():
        print(f"[ERROR] Folder '{RESULTS_DIR}' not found.")
        return

    patient_rows = []

    patient_folders = sorted(folder for folder in RESULTS_DIR.iterdir() if folder.is_dir())
    if not patient_folders:
        print(f"[WARNING] No patient folders found in {RESULTS_DIR}/")
        return

    for patient_folder in patient_folders:
        patient_id = patient_folder.name
        original_json = patient_folder / f"{patient_id}_signals.json"

        if not original_json.exists():
            continue

        original = load_results(original_json)
        missing_root = patient_folder / "missing"
        if not missing_root.exists():
            continue

        for mechanism_folder in sorted(folder for folder in missing_root.iterdir() if folder.is_dir()):
            mechanism = mechanism_folder.name

            for rate_folder in sorted(
                (folder for folder in mechanism_folder.iterdir() if folder.is_dir()),
                key=lambda folder: int(folder.name.replace("pct", "")),
            ):
                rate = rate_folder.name
                runs = find_run_jsons(patient_folder, mechanism, rate)

                if not runs:
                    continue

                patient_matrix = empty_matrix()
                run_count = 0

                for seed, json_file in runs:
                    if not json_file.exists():
                        continue

                    missing = load_results(json_file)
                    run_matrix = matrix_from_results(original, missing)
                    patient_matrix = add_matrices(patient_matrix, run_matrix)
                    run_count += 1

                if run_count == 0:
                    continue

                metrics = metrics_from_matrix(patient_matrix)
                
                # Save heatmap per patient
                heatmap_dir = RESULTS_DIR / "heatmaps_by_patient" / patient_id
                heatmap_path = heatmap_dir / f"{mechanism}_{rate}.png"
                save_heatmap(patient_matrix, f"Patient {patient_id} - {mechanism} - {rate}", heatmap_path)
                
                patient_rows.append(
                    {
                        "patient": patient_id,
                        "mechanism": mechanism,
                        "rate": rate,
                        "n_runs": run_count,
                        "green_green": patient_matrix[0][0],
                        "green_yellow": patient_matrix[0][1],
                        "green_red": patient_matrix[0][2],
                        "yellow_green": patient_matrix[1][0],
                        "yellow_yellow": patient_matrix[1][1],
                        "yellow_red": patient_matrix[1][2],
                        "red_green": patient_matrix[2][0],
                        "red_yellow": patient_matrix[2][1],
                        "red_red": patient_matrix[2][2],
                        **metrics,
                    }
                )

    if not patient_rows:
        print("[WARNING] No matrices were generated.")
        return

    patient_df = pd.DataFrame(patient_rows)
    patient_df["_rate_num"] = patient_df["rate"].str.replace("pct", "", regex=False).astype(int)
    patient_df = patient_df.sort_values(["mechanism", "_rate_num", "patient"]).drop(columns=["_rate_num"])
    patient_df.to_csv(RESULTS_DIR / "confusion_matrices_by_patient.csv", index=False)

    # Calculate mean and std of metrics by mechanism and rate from patient-level values
    metric_cols = [
        "accuracy_alert",
        "accuracy_warning",
        "precision_warning",
        "precision_alert",
        "recall_warning",
        "recall_alert",
    ]

    summary_rows = []
    for (mechanism, rate), group in patient_df.groupby(["mechanism", "rate"], sort=False):
        row = {
            "mechanism": mechanism,
            "rate": rate,
            "n_patients": int(group["patient"].nunique()),
        }

        # Add mean and std of each metric computed from patient-level values
        for metric_name in metric_cols:
            row[f"{metric_name}_mean"] = float(group[metric_name].mean())
            row[f"{metric_name}_std"] = float(group[metric_name].std(ddof=1)) if len(group) > 1 else 0.0

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df["_rate_num"] = summary_df["rate"].str.replace("pct", "", regex=False).astype(int)
    summary_df = summary_df.sort_values(["mechanism", "_rate_num"]).drop(columns=["_rate_num"])
    summary_df.to_csv(RESULTS_DIR / "confusion_matrices_summary_by_mechanism_rate.csv", index=False)

    print(f"Saved per-patient matrices to {RESULTS_DIR / 'confusion_matrices_by_patient.csv'}")
    print(f"Saved per-patient heatmaps to {RESULTS_DIR / 'heatmaps_by_patient/'}")
    print(f"Saved summary (mean/std by mechanism and rate) to {RESULTS_DIR / 'confusion_matrices_summary_by_mechanism_rate.csv'}")


if __name__ == "__main__":
    main()