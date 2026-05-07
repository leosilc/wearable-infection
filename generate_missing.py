# import shutil
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from mdatagen.univariate.uMCAR import uMCAR
import os
import subprocess


PROCESSING_DIR = Path("data/processing")
MISSING_DIR = Path("data/missing")
RESULTS_DIR = Path("results")

# Taxas de ausência que serão geradas
RATES = [99]

SEED = 42

# Nome da coluna que receberá os dados ausentes
HR_COLUMN = "HR_Value"

def mnar(df: pd.DataFrame, taxa: int) -> pd.DataFrame:
    # implementar
    print("MNAR not yet implemented")

def mar(df: pd.DataFrame, taxa: int) -> pd.DataFrame:
    # implementar
    print("MAR not yet implemented")



def mcar(df: pd.DataFrame, taxa: int) -> pd.DataFrame:

    # Recebe o DataFrame original e devolve uma CÓPIA com alguns valores de HR_Value substituídos por NaN, de forma completamente aleatória (MCAR).

    df_out = df.copy()

    # uMCAR exige:
    #   X = DataFrame com a coluna que vai receber os valores ausentes
    #   y = array numpy qualquer (MCAR não usa rótulos — passamos zeros)
    #   missing_rate 
    #   x_miss = nome da coluna que vai receber os NaNs
    #   seed 

    X = df_out[[HR_COLUMN]].astype(float)
    y = np.zeros(len(X))

    generator = uMCAR(X=X, y=y, missing_rate=taxa, x_miss=HR_COLUMN, seed=SEED)

    # .random() é o método que gera os NaNs e os RETORNA num novo DataFrame.
    X_with_missing = generator.random()


    #adicionando a coluna de Steps e Batimentos ao final
    df_out["HR_MISSING"] = df[HR_COLUMN].astype(int)  # cópia da coluna original para referência
    df_out["ST_MISSING"] = df["ST_Value"]#.astype(int) # se precisar ser float em algum momento


    # Substitui a coluna HR no DataFrame completo
    df_out[HR_COLUMN] = X_with_missing[HR_COLUMN].values

    # # Converte mantendo os NaN como nulos, mas tratando o resto como inteiro
    df_out[HR_COLUMN] = df_out[HR_COLUMN].astype('Int64')
   

    return df_out


def save_csv_missing(df: pd.DataFrame, patient_id: str, taxa: int, mechanism: str):
    """
    Salva o CSV com ausências em data/missing/MCAR/{taxa}pct/{patient_id}/

    O formato das colunas é:
        Datetime, HR_Value, f0_, ST_Value, HR_missing, ST_missing

    Isso garante que o NightSignal lê o arquivo sem nenhuma modificação —
    ele vai encontrar campo vazio onde havia um valor de HR e, com a pequena
    correção no nightsignal.py, vai pular esse minuto no cálculo da média.

    NaN é escrito como campo vazio ("") — padrão CSV para dado ausente.
    """
    folder = MISSING_DIR / mechanism / f"{taxa}pct" / patient_id
    folder.mkdir(parents=True, exist_ok=True)

    output_path = folder / f"{patient_id}_temp.csv"

    # Datetime de volta para string antes de salvar
    df_save = df.copy()
    df_save["Datetime"] = df_save["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # na_rep="" → campo vazio no CSV em vez da palavra "nan"
    df_save.to_csv(output_path, index=False, na_rep="")

    return output_path


def process_patient(patient_id: str, input_csv: Path, mechanism: str):
    """
    Fluxo completo para um paciente:
      1. Lê o CSV processado
      2. Salva cópia original (uma vez)
      3. Para cada taxa: aplica MCAR e salva CSV com ausência
    """
    print(f"\Patient: {patient_id}")

    # Lê o CSV — HR e Steps como float para aceitar NaN depois
    df = pd.read_csv(input_csv, dtype={HR_COLUMN: float, "ST_Value": int})
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    total_rows = len(df)
    print(f"  {total_rows} rows loaded from {input_csv}")

    # # Salva cópia do original (referência para comparação futura)
    # salvar_original(csv_entrada, patient_id)
    # print(f"  Original copiado para: data/missing/original/{patient_id}/")

    # Gera um CSV para cada taxa
    for rate in RATES:
        df_ausente = mcar(df, rate)

        nan_count = df_ausente[HR_COLUMN].isna().sum()
        actual_rate = nan_count / total_rows * 100

        output_path = save_csv_missing(df_ausente, patient_id, rate, mechanism)

        print(f"  {mechanism} {rate:>2}%: {nan_count:>5} NaNs / {total_rows} linhas "
              f"= {actual_rate:.1f}% real  →  {output_path}")
        print("Running NightSignal for this patient with missing data...")
        run_patient_process(rate, patient_id, mechanism)



def run_patient_process(taxa: int, patient_id: str, mechanism: str):
    print("Starting NightSignal processing...")
 

    patient_dir = MISSING_DIR / mechanism / f"{taxa}pct" / patient_id
    
    if not os.path.exists(patient_dir):
        print(f"Folder not found: {patient_dir}. Skipping...")
        return
 
    # Define o caminho do CSV de entrada
    input_csv = patient_dir / f"{patient_id}_temp.csv"
    
    # Define e cria a pasta de destino
    output_dir = RESULTS_DIR / patient_id / "missing" / mechanism / f"{taxa}pct"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(input_csv):
        print(f"  > Running {patient_id}...")
        
        # COMANDO PARA RODAR O NIGHTSIGNAL
        try:
            command = [
                "python3", "nightsignal.py",
                "--device=Fitbit",
                f"--restinghr={input_csv}"
            ]
            subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing patient {patient_id}: {e}")
    else:
        print(f"CSV not found for {patient_id}")


    # organizando arquivos gerados
    if os.path.exists("NS-signals.json"):
        shutil.move("NS-signals.json", output_dir / f"{patient_id}_signals.json")
    
    if os.path.exists("NightSignalResult.pdf"):
        shutil.move("NightSignalResult.pdf", output_dir / f"{patient_id}_plot.pdf")

        

print("\nOperation complete! Check the 'results' folder.")


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    if not PROCESSING_DIR.exists():
        print(f"[ERRO] Pasta '{PROCESSING_DIR}' não encontrada.")
        print("Run run_all.py first to generate the processed CSVs.")
        return

    # Encontra todas as pastas de pacientes em data/processing/
    folders_patients = sorted(
        p for p in PROCESSING_DIR.iterdir() if p.is_dir()
    )

    if not folders_patients:
        print(f"[WARNING] No patient folder found in {PROCESSING_DIR}.")
        return

    print(f"{'='*55}")
    print(f"Generating MCAR missing data — rates: {RATES}%  |  seed: {SEED}")
    print(f"{'='*55}")

    for folder in folders_patients:
        patient_id = folder.name
        input_csv = folder / f"{patient_id}_temp.csv"

        if not input_csv.exists():
            print(f"\n[WARNING] {input_csv} not found, skipping.")
            continue

        process_patient(patient_id, input_csv, "MCAR")



if __name__ == "__main__":
    main()