import json
from pathlib import Path

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
    matriz = [[[] for _ in range(3)] for _ in range(3)] # matriz 3x3 com listas vaizas para armazenar as datas

    dates = set(original.keys()) 

    for date in dates:
        o = int(original[date])
        m = int(missing[date])

        matriz[o][m].append(date) # adiciona a data na posição correspondente da matriz

    for i in range(3):
        
    # calcular recall, precisao conforme afonso orientou
    return matriz

def main():
    o_results = load_results('results/P111019/P111019_signals.json')
    d_results = load_results('results/P111019/missing/MCAR/10pct/P111019_signals.json')

    metrics_matrix = metrics(o_results, d_results)
    datas_erro = metrics_matrix[2][1]
    print(f"Datas que mudaram de 1 para 1: {datas_erro}")

    

if __name__ == "__main__":
    main()