import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# ALTERAR RECALL E PRECISÃO E CRIAR ACURÁCIA PARA ALERTA

RESULTS_DIR = Path("results")
MISSING_DIR = RESULTS_DIR / "missing" 


def load_results(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        if file:
            print(f"Data successfully loaded from {json_file}")
    return {
        " ".join(entry["date"].split()): entry["val"]
        for entry in data["nightsignal"]
}

def metrics(original: dict[str, str], missing: dict[str, str]) -> tuple[list[list[list[str]]], dict[str, str]]:
    matriz = [[[] for _ in range(3)] for _ in range(3)] # matriz 3x3 com listas vazias para armazenar as datas

    dates = set(original.keys()) & set(missing.keys())
    metrics_calculated = {
        "precision_warning": 0,
        "precision_alert": 0,
        "recall_warning": 0,
        "recall_alert": 0,
        "recall_both": 0,
        "accuracy_alert": 0
    }

    for date in dates:
        o = int(original[date])
        m = int(missing[date])

        matriz[o][m].append(date) # adiciona a data na posição correspondente da matriz

    # somar elementos das linhas 2 e 3 (índices 1 e 2)
    safe_div = lambda num, den: num / den if den else 0

    metrics_calculated["precision_warning"] = safe_div(
        len(matriz[1][1]), sum(len(matriz[1][j]) for j in range(3))
    )
    metrics_calculated["precision_alert"] = safe_div(
        len(matriz[2][2]), sum(len(matriz[2][j]) for j in range(3))
    )
    metrics_calculated["recall_warning"] = safe_div(
        len(matriz[1][1]), sum(len(matriz[i][1]) for i in range(3))
    )
    metrics_calculated["recall_alert"] = safe_div(
        len(matriz[2][2]), sum(len(matriz[i][2]) for i in range(3))
    )

    # mudar nome
    metrics_calculated["recall_both"] = safe_div(
        (len(matriz[1][0]) + len(matriz[2][0])), (len(matriz[1][1]) + len(matriz[2][2]))
    )
    
    
    metrics_calculated["accuracy_alert"] = safe_div(
        len(matriz[2][2]),
        sum(len(matriz[i][j]) for i in range(3) for j in range(3)) - len(matriz[0][0]),
    )

    # retorna matriz com dias e horários de cada true ou false posição, e um dicionário com as métricas calculadas  
    return matriz, metrics_calculated

# protegente de ZeroDivisionError, caso a divisão seja por zero, retorna 0 para a métrica
def format(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:.4f}"
 
 # apenas para visualizar e testar
def format_report(
    patient_id: str,
    mechanism: str,
    rate: str,
    matriz: list[list[list[str]]],
    mc: dict,
) -> str:
    labels = ["Green (0)", "Yellow (1)", "Red (2)"]
    rows = []
 
    rows.append(f"Patient : {patient_id}")
    rows.append(f"Mechanism: {mechanism}  |  Rate: {rate}")
    rows.append("")
 
    rows.append("Confusion Matrix (rows=original, columns=with missing data):")
    header = f"{'':>14}" + "".join(f"{l:>14}" for l in labels)
    rows.append(header)
    for i, row in enumerate(matriz):
        counts = "".join(f"{len(cell):>14}" for cell in row)
        rows.append(f"{labels[i]:>14}{counts}")
    rows.append("")
 
    rows.append("Metrics:")
    rows.append(f"  Accuracy Red           : {format(mc['accuracy_alert'])}")
    rows.append(f"  Precision Yellow   : {format(mc['precision_warning'])}")
    rows.append(f"  Precision Red      : {format(mc['precision_alert'])}")
    rows.append(f"  Recall Yellow      : {format(mc['recall_warning'])}")
    rows.append(f"  Recall Red         : {format(mc['recall_alert'])}")
    rows.append(f"  Recall Both        : {format(mc['recall_both'])}")
 
    lost_red_alerts  = matriz[2][0]
    lost_yellow_warnings = matriz[1][0]
    new_false_alerts     = matriz[0][1] + matriz[0][2]
 
    if lost_red_alerts:
        rows.append(f"\n  CRITICAL - Lost red alerts ({len(lost_red_alerts)} day(s)):")
        rows.append("    " + ", ".join(sorted(lost_red_alerts)))
    if lost_yellow_warnings:
        rows.append(f"\n  Lost yellow alerts ({len(lost_yellow_warnings)} day(s)):")
        rows.append("    " + ", ".join(sorted(lost_yellow_warnings)))
    if new_false_alerts:
        rows.append(f"\n  New false alerts ({len(new_false_alerts)} day(s)):")
        rows.append("    " + ", ".join(sorted(new_false_alerts)))
 
    rows.append("\n" + "-" * 60)
    return "\n".join(rows)

def plot(df: pd.DataFrame):
    # plotar o boxplot de cada paciente, comparando as métricas entre os mecanismos e taxas

    sns.boxplot(data=df, x="rate", y="accuracy_alert", palette="viridis")
    
    # Adiciona os pontinhos de cada paciente para mostrar a dispersão
    sns.stripplot(data=df, x="rate", y="accuracy_alert", color="black", alpha=0.3)

    plt.title("NightSignal — Accuracy by Missing Data Rate")
    plt.ylabel("Mean Accuracy (0.0 to 1.0)")
    plt.xlabel("Missing Data Rate")
    plt.ylim(0, 1.05)
 
    plt.savefig("accuracy_alert_plot.png")
    plt.show()


def plot_per_patient_metrics(metrics_csv: Path):
    """
    Cria gráficos para cada paciente mostrando cada métrica.
    Para cada métrica, mostra linhas para MCAR, MAR e MNAR com erro bar.
    Eixo X: taxa de dados faltantes (5, 10, 20)
    Eixo Y: valor da métrica
    """
    
    # Carrega o CSV
    if not metrics_csv.exists():
        print(f"[ERROR] Metrics CSV not found at {metrics_csv}")
        return
    
    df = pd.read_csv(metrics_csv)
    print(f"\nLoaded {len(df)} rows from metrics.csv")
    
    # Cria pasta para gráficos se não existir
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Lista de métricas (todas as colunas que terminam com "_mean")
    metric_cols = [col.replace("_mean", "") for col in df.columns if col.endswith("_mean")]
    print(f"Found metrics: {metric_cols}\n")
    
    # Extrai taxa em forma numérica (5, 10, 20 de "5pct", "10pct", "20pct")
    df['rate_numeric'] = df['rate'].str.replace('pct', '').astype(int)
    
    # Para cada paciente
    for patient_id in df['patient'].unique():
        patient_df = df[df['patient'] == patient_id].copy()
        patient_dir = plots_dir / patient_id
        patient_dir.mkdir(exist_ok=True)
        
        print(f"Creating plots for patient: {patient_id}")
        
        # Para cada métrica
        for metric in metric_cols:
            metric_col = f"{metric}_mean"
            metric_std_col = f"{metric}_std"
            
            # Verifica se as colunas existem
            if metric_col not in patient_df.columns or metric_std_col not in patient_df.columns:
                continue
            
            # Cria figura
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Para cada mecanismo (MCAR, MAR, MNAR)
            mechanisms = patient_df['mechanism'].unique()
            colors = {'MCAR': '#1f77b4', 'MAR': '#ff7f0e', 'MNAR': '#2ca02c'}
            markers = {'MCAR': 'o', 'MAR': 's', 'MNAR': '^'}
            
            for mechanism in sorted(mechanisms):
                mech_data = patient_df[patient_df['mechanism'] == mechanism].sort_values('rate_numeric')
                
                if len(mech_data) > 0:
                    ax.errorbar(
                        mech_data['rate_numeric'],
                        mech_data[metric_col],
                        yerr=mech_data[metric_std_col],
                        marker=markers.get(mechanism, 'o'),
                        label=mechanism,
                        color=colors.get(mechanism),
                        linewidth=2,
                        markersize=8,
                        capsize=5,
                        capthick=2,
                        alpha=0.8
                    )
            
            # Formatação
            ax.set_xlabel('Missing Data Rate (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_title(f'Patient {patient_id} - {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.set_xticks(sorted(patient_df['rate_numeric'].unique()))
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=11)
            ax.set_ylim(0, 1.05)
            
            # Salva figura
            metric_filename = metric.replace(' ', '_').lower()
            output_path = patient_dir / f"{metric_filename}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {output_path}")
            plt.close()


def main():
    if not RESULTS_DIR.exists():
        print(f"[ERROR] Folder '{RESULTS_DIR}' not found.")
        return

    metrics_list = []
    
 
    # descobre todos os pacientes automaticamente
    patients = sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir())
 
    if not patients:
        print(f"None patient found in {RESULTS_DIR}/")
        return
 
    print(f"\nFounded patients: {[p.name for p in patients]}\n")
 
    for patient_folder in patients:
        patient_id = patient_folder.name
 
        json_original = patient_folder / f"{patient_id}_signals.json"
        if not json_original.exists():
            print(f"[WARNING] {patient_id}: JSON original not found, skipping.")
            continue
 
        original = load_results(json_original)
        print(f"{'='*60}")
        print(f"Patient: {patient_id}  ({len(original)} days on original data)")
 
        missing_root = patient_folder / "missing"
        if not missing_root.exists():
            print(f"  There is not 'missing/', skipping.\n")
            continue
 
        # descobre mecanismos e taxas automaticamente pela estrutura de pastas
        mechanisms = sorted(m for m in missing_root.iterdir() if m.is_dir())

 
        report_blocks = []
 
        for mechanism_folder in mechanisms:
            mechanism = mechanism_folder.name
            rates = sorted(
                (t for t in mechanism_folder.iterdir() if t.is_dir()),
                key=lambda folder: int(folder.name.replace("pct", "")),
            )
 
            for rate_folder in rates:
                rate = rate_folder.name

                direct_json = rate_folder / f"{patient_id}_signals.json"
                if direct_json.exists():
                    runs = [("", direct_json)]
                else:
                    seed_folders = sorted(
                        (s for s in rate_folder.iterdir() if s.is_dir()),
                        key=lambda folder: int(folder.name),
                    )
                    runs = [
                        (seed_folder.name, seed_folder / f"{patient_id}_signals.json")
                        for seed_folder in seed_folders
                    ]

                if not runs:
                    print(f"  [{mechanism}/{rate}] JSON nao encontrado, pulando.")
                    continue

                for seed, json_deg in runs:
                    if not json_deg.exists():
                        if seed:
                            print(f"  [{mechanism}/{rate}/seed={seed}] JSON nao encontrado, pulando.")
                        else:
                            print(f"  [{mechanism}/{rate}] JSON nao encontrado, pulando.")
                        continue

                    missing = load_results(json_deg)
                    matriz, mc = metrics(original, missing)

                    run_suffix = f" | seed={seed}" if seed else ""
                    print(f"\n  [{mechanism}] Rate: {rate}{run_suffix}")
                    print(f"    Accuracy Alert         : {format(mc['accuracy_alert'])}")
                    print(f"    Yellow Precision : {format(mc['precision_warning'])}")
                    print(f"    Red Precision    : {format(mc['precision_alert'])}")
                    print(f"    Yellow Recall    : {format(mc['recall_warning'])}")
                    print(f"    Red Recall       : {format(mc['recall_alert'])}")

                    mechanism_report = mechanism if not seed else f"{mechanism} (seed {seed})"
                    report_blocks.append(
                        format_report(patient_id, mechanism_report, rate, matriz, mc)
                    )


                    metrics_list.append({
                        "patient": patient_id,
                        "mechanism": mechanism,
                        "seed": seed,
                        "rate": rate,
                        "accuracy_alert": mc['accuracy_alert'],
                        "precision_warning": mc['precision_warning'],
                        "precision_alert": mc['precision_alert'],
                        "recall_warning": mc['recall_warning'],
                        "recall_alert": mc['recall_alert'],
                        "recall_both": mc['recall_both']
                    })

                
 
        # salva txt na pasta do paciente dentro de results/
        if report_blocks:
            txt_path = patient_folder / f"{patient_id}_metrics.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"METRICS REPORT — {patient_id}\n")
                f.write("=" * 60 + "\n\n")
                f.write("\n\n".join(report_blocks))
            print(f"\n  Saved in: {txt_path}")

    df = pd.DataFrame(metrics_list)

    if df.empty:
        print("[WARNING] No metrics were generated. Skipping CSV and plot.")
        return

    grouped = df.groupby(["patient", "mechanism", "rate"], as_index=False).agg(
        n_runs=("accuracy_alert", "size"),
        accuracy_mean=("accuracy_alert", "mean"),
        accuracy_std=("accuracy_alert", "std"),
        precision_warning_mean=("precision_warning", "mean"),
        precision_warning_std=("precision_warning", "std"),
        precision_alert_mean=("precision_alert", "mean"),
        precision_alert_std=("precision_alert", "std"),
        recall_warning_mean=("recall_warning", "mean"),
        recall_warning_std=("recall_warning", "std"),
        recall_alert_mean=("recall_alert", "mean"),
        recall_alert_std=("recall_alert", "std"),
        recall_both_mean=("recall_both", "mean"),
        recall_both_std=("recall_both", "std"),
    )

    # casos com desvio padrão vazio, em que só ocorreu uma execução
    std_cols = [c for c in grouped.columns if c.endswith("_std")]
    grouped[std_cols] = grouped[std_cols].fillna(0.0)

    grouped["_rate_num"] = grouped["rate"].str.replace("pct", "", regex=False).astype(int)
    grouped = grouped.sort_values(["patient", "mechanism", "_rate_num"]).drop(columns=["_rate_num"])

    grouped.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    plot(grouped)
    
    # Gera gráficos individuais por paciente
    print(f"\n{'='*60}")
    print("Generating per-patient metric plots...")
    plot_per_patient_metrics(RESULTS_DIR / "metrics.csv")
 
    print(f"\n{'='*60}")
    print("Concluido.")
 
 
if __name__ == "__main__":
    main()