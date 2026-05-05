import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
MISSING_DIR = RESULTS_DIR / "missing" 


def load_results(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        if file:
            print(f"Dados carregados com sucesso de {json_file}")
    return {
        " ".join(entry["date"].split()): entry["val"]
        for entry in data["nightsignal"]
}

def metrics(original: dict[str, str], missing: dict[str, str]) -> tuple[list[list[list[str]]], dict[str, str]]:
    matriz = [[[] for _ in range(3)] for _ in range(3)] # matriz 3x3 com listas vazias para armazenar as datas

    dates = set(original.keys()) 
    metrics_calculated = {
        "precision_warning": 0,
        "precision_alert": 0,
        "recall_warning": 0,
        "recall_alert": 0,
        "recall_both": 0,
        "accuracy": 0
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
    
    
    metrics_calculated["accuracy"] = safe_div(
        len(matriz[1][1]) + len(matriz[2][2]),
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
def formatar_relatorio(
    patient_id: str,
    mecanismo: str,
    taxa: str,
    matriz: list[list[list[str]]],
    mc: dict,
) -> str:
    labels = ["Verde (0)", "Amarelo (1)", "Vermelho (2)"]
    linhas = []
 
    linhas.append(f"Paciente : {patient_id}")
    linhas.append(f"Mecanismo: {mecanismo}  |  Taxa: {taxa}")
    linhas.append("")
 
    linhas.append("Matriz de Confusao (linhas=original, colunas=com ausencia):")
    header = f"{'':>14}" + "".join(f"{l:>14}" for l in labels)
    linhas.append(header)
    for i, row in enumerate(matriz):
        counts = "".join(f"{len(cell):>14}" for cell in row)
        linhas.append(f"{labels[i]:>14}{counts}")
    linhas.append("")
 
    linhas.append("Metricas:")
    linhas.append(f"  Acuracia           : {format(mc['accuracy'])}"
                  "  (diagonal / total excluindo verde->verde)")
    linhas.append(f"  Precisao Amarelo   : {format(mc['precision_warning'])}")
    linhas.append(f"  Precisao Vermelho  : {format(mc['precision_alert'])}")
    linhas.append(f"  Recall Amarelo     : {format(mc['recall_warning'])}")
    linhas.append(f"  Recall Vermelho    : {format(mc['recall_alert'])}")
    linhas.append(f"  Recall Ambos       : {format(mc['recall_both'])}")
 
    perdidos_alerta  = matriz[2][0]
    perdidos_warning = matriz[1][0]
    novos_falsos     = matriz[0][1] + matriz[0][2]
 
    if perdidos_alerta:
        linhas.append(f"\n  CRITICO - Alertas vermelhos perdidos ({len(perdidos_alerta)} dia(s)):")
        linhas.append("    " + ", ".join(sorted(perdidos_alerta)))
    if perdidos_warning:
        linhas.append(f"\n  Avisos amarelos perdidos ({len(perdidos_warning)} dia(s)):")
        linhas.append("    " + ", ".join(sorted(perdidos_warning)))
    if novos_falsos:
        linhas.append(f"\n  Alertas falsos novos ({len(novos_falsos)} dia(s)):")
        linhas.append("    " + ", ".join(sorted(novos_falsos)))
 
    linhas.append("\n" + "-" * 60)
    return "\n".join(linhas)

def plot(patients: list[Path])
    # plotar o boxplot de cada paciente, comparando as métricas entre os mecanismos e taxas



def main():
    if not RESULTS_DIR.exists():
        print(f"[ERROR] Paste '{RESULTS_DIR}' not found.")
        return
 
    # descobre todos os pacientes automaticamente
    patients = sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir())
 
    if not patients:
        print(f"None patient found in {RESULTS_DIR}/")
        return
 
    print(f"\nFounded patients: {[p.name for p in patients]}\n")
 
    for paste_paciente in patients:
        patient_id = paste_paciente.name
 
        json_original = paste_paciente / f"{patient_id}_signals.json"
        if not json_original.exists():
            print(f"[WARNING] {patient_id}: JSON original not found, skipping.")
            continue
 
        original = load_results(json_original)
        print(f"{'='*60}")
        print(f"Paciente: {patient_id}  ({len(original)} days on original data)")
 
        missing_root = paste_paciente / "missing"
        if not missing_root.exists():
            print(f"  There is not 'missing/', skipping.\n")
            continue
 
        # descobre mecanismos e taxas automaticamente pela estrutura de pastas
        mechanisms = sorted(m for m in missing_root.iterdir() if m.is_dir())
 
        blocos_relatorio = []
 
        for paste_mec in mechanisms:
            mecanismo = paste_mec.name
            rates = sorted(t for t in paste_mec.iterdir() if t.is_dir())
 
            for paste_rate in rates:
                rate = paste_rate.name
 
                json_deg = paste_rate / f"{patient_id}_signals.json"
                if not json_deg.exists():
                    print(f"  [{mecanismo}/{rate}] JSON nao encontrado, pulando.")
                    continue
 
                missing = load_results(json_deg)
                matriz, mc = metrics(original, missing)
 
                print(f"\n  [{mecanismo}] Taxa: {rate}")
                print(f"    Acuracia         : {format(mc['accuracy'])}")
                print(f"    Precisao Amarelo : {format(mc['precision_warning'])}")
                print(f"    Precisao Vermelho: {format(mc['precision_alert'])}")
                print(f"    Recall Amarelo   : {format(mc['recall_warning'])}")
                print(f"    Recall Vermelho  : {format(mc['recall_alert'])}")
 
                blocos_relatorio.append(
                    formatar_relatorio(patient_id, mecanismo, rate, matriz, mc)
                )
 
        # salva txt na pasta do paciente dentro de results/
        if blocos_relatorio:
            txt_path = paste_paciente / f"{patient_id}_metrics.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"RELATORIO DE METRICAS — {patient_id}\n")
                f.write("=" * 60 + "\n\n")
                f.write("\n\n".join(blocos_relatorio))
            print(f"\n  Salvo em: {txt_path}")
 
    print(f"\n{'='*60}")
    print("Concluido.")
 
 
if __name__ == "__main__":
    main()