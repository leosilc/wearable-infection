# import shutil
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from mdatagen.univariate.uMCAR import uMCAR
from mdatagen.univariate.uMNAR import uMNAR
from mdatagen.univariate.uMAR import uMAR
import os
import subprocess
import missingno as msno
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


PROCESSING_DIR = Path("data/processing")
MISSING_DIR = Path("data/missing")
RESULTS_DIR = Path("results")

# Taxas de ausência que serão geradas
RATES = [5,10,20]
MCAR_SEEDS = range(1, 6)


# np.random.seed(SEED)

# Nome da coluna que receberá os dados ausentes
HR_COLUMN = "HR_Value"


def mnar(df: pd.DataFrame, rate: int, seed: int) -> pd.DataFrame:
    
    df_out = df.copy()

    if seed is not None:
        np.random.seed(seed)

    # marca true para todos os horários dentro da madrugada
    night_mask = df_out["Datetime"].dt.hour < 6
    # cria cópia do df apenas pegando os horários acima e selecionando apenas o HR
    df_night = df_out.loc[night_mask, ["Datetime", HR_COLUMN]].copy().reset_index(drop=True)

    # converte para minutos desde meia-noite — valor numérico que uMAR consegue rankear
    df_night["time"] = (df_night["Datetime"].dt.hour * 60 + df_night["Datetime"].dt.minute)

    

    # print(df_out[["time", HR_COLUMN]].head())  # Verificar as primeiras linhas para garantir que as colunas estão corretas

    X = df_night[["time", HR_COLUMN]].reset_index(drop=True)

    # X = df_out[[HR_COLUMN]].astype(float)
    y = np.zeros(len(X))

    # threshold=1.0 + deterministic=True faz o gerador escolher apenas a cauda superior
    # da ordenação de HR, ou seja, os maiores batimentos dentro da janela noturna.
    generator = uMNAR(X=X, y=y, threshold=1.0, missing_rate=rate, x_miss=HR_COLUMN)

    X_with_missing = generator.run(deterministic=True)

    #adicionando a coluna de Steps e Batimentos ao final
    df_out["HR_MISSING"] = df[HR_COLUMN].astype(int)  # cópia da coluna original para referência
    df_out["ST_MISSING"] = df["ST_Value"]#.astype(int) # se precisar ser float em algum momento


    df_out.loc[night_mask, HR_COLUMN] = X_with_missing[HR_COLUMN].values


    # Substitui a coluna HR no DataFrame completo
    # df_out[HR_COLUMN] = X_with_missing[HR_COLUMN].values

    # # Converte mantendo os NaN como nulos, mas tratando o resto como inteiro
    df_out[HR_COLUMN] = df_out[HR_COLUMN].astype('Int64')

    

    return df_out

def mar(df: pd.DataFrame, rate: int) -> pd.DataFrame:
    # Cria uma cópia do DataFrame original
    df_out = df.copy()

    # marca true para todos os horários dentro da madrugada
    night_mask = df_out["Datetime"].dt.hour < 6
    # cria cópia do df apenas pegando os horários acima e selecionando apenas o HR
    df_night = df_out.loc[night_mask, ["Datetime", HR_COLUMN]].copy()

    # converte para minutos desde meia-noite — valor numérico que uMAR consegue rankear
    df_night["time"] = (df_night["Datetime"].dt.hour * 60 + df_night["Datetime"].dt.minute)

    

    # print(df_out[["time", HR_COLUMN]].head())  # Verificar as primeiras linhas para garantir que as colunas estão corretas

    X = df_night[["time",HR_COLUMN]]

    # y volta a ser um array neutro, pois a mdatagen usa o X para a lógica MAR
    y = np.zeros(len(X))

    # Instancia o gerador apontando x_obs para a nova coluna do X
    generator = uMAR(X=X, y=y, missing_rate=rate, x_miss=HR_COLUMN, x_obs="time")
    X_with_missing = generator.lowest()

    # Atualiza a coluna no DataFrame principal
    # df_out[HR_COLUMN] = X_with_missing[HR_COLUMN].values

    # Adiciona as colunas de referência originais para validação futura
    df_out["HR_MISSING"] = df[HR_COLUMN].astype(int)
    df_out["ST_MISSING"] = df["ST_Value"]

    df_out.loc[night_mask, HR_COLUMN] = X_with_missing[HR_COLUMN].values

    df_out[HR_COLUMN] = df_out[HR_COLUMN].astype('Int64')

    return df_out




def mcar(df: pd.DataFrame, rate: int, seed: int) -> pd.DataFrame:

    # Recebe o DataFrame original e devolve uma CÓPIA com alguns valores de HR_Value substituídos por NaN, de forma completamente aleatória (MCAR).

    df_out = df.copy()

    # uMCAR exige:
    #   X = DataFrame com a coluna que vai receber os valores ausentes
    #   y = array numpy qualquer (MCAR não usa rótulos — passamos zeros)
    #   missing_rate 
    #   x_miss = nome da coluna que vai receber os NaNs
    #   seed 

    # marca true para todos os horários dentro da madrugada
    night_mask = df_out["Datetime"].dt.hour < 6
    # cria cópia do df apenas pegando os horários acima e selecionando apenas o HR
    df_night = df_out.loc[night_mask, [HR_COLUMN]].copy()

    X = df_night
    # print(X.head())

    # X = df_out[[HR_COLUMN]].astype(float)
    y = np.zeros(len(X))

    generator = uMCAR(X=X, y=y, missing_rate=rate, x_miss=HR_COLUMN, seed=seed)

    X_with_missing = generator.random()

    #adicionando a coluna de Steps e Batimentos ao final
    df_out["HR_MISSING"] = df[HR_COLUMN].astype(int)  # cópia da coluna original para referência
    df_out["ST_MISSING"] = df["ST_Value"]#.astype(int) # se precisar ser float em algum momento


    df_out.loc[night_mask, HR_COLUMN] = X_with_missing[HR_COLUMN].values


    # Substitui a coluna HR no DataFrame completo
    # df_out[HR_COLUMN] = X_with_missing[HR_COLUMN].values

    # # Converte mantendo os NaN como nulos, mas tratando o resto como inteiro
    df_out[HR_COLUMN] = df_out[HR_COLUMN].astype('Int64')
    
    return df_out




def save_csv_missing(
    df: pd.DataFrame,
    patient_id: str,
    taxa: int,
    mechanism: str,
    seed: int | None = None,
):
    """
    Salva o CSV com ausências em data/missing/MCAR/{taxa}pct/{patient_id}/

    O formato das colunas é:
        Datetime, HR_Value, f0_, ST_Value, HR_missing, ST_missing

    Isso garante que o NightSignal lê o arquivo sem nenhuma modificação —
    ele vai encontrar campo vazio onde havia um valor de HR e, com a pequena
    correção no nightsignal.py, vai pular esse minuto no cálculo da média.

    NaN é escrito como campo vazio ("") — padrão CSV para dado ausente.
    """
    folder = MISSING_DIR / mechanism / f"{taxa}pct"
    if seed is not None:
        folder = folder / str(seed)
    folder = folder / patient_id
    folder.mkdir(parents=True, exist_ok=True)

    output_path = folder / f"{patient_id}_temp.csv"

    # Datetime de volta para string antes de salvar
    df_save = df.copy()
    df_save["Datetime"] = df_save["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # na_rep="" → campo vazio no CSV em vez da palavra "nan"
    df_save.to_csv(output_path, index=False, na_rep="")

    return output_path


def process_patient(
    patient_id: str,
    input_csv: Path,
    mechanism: str,
    rate: int,
    seed: int | None = None,
) -> pd.DataFrame:
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


    if mechanism == "MCAR":
        if seed is None:
            raise ValueError("MCAR requires a seed value.")

        missing_df = mcar(df, rate, seed=seed)
        nan_count = missing_df[HR_COLUMN].isna().sum()
        actual_rate = nan_count / total_rows * 100
        output_path = save_csv_missing(missing_df, patient_id, rate, mechanism, seed=seed)

        print(f"  {mechanism} seed={seed} {rate:>2}%: {nan_count:>5} NaNs / {total_rows} linhas "
            f"= {actual_rate:.1f}% real  →  {output_path}")
        print("Running NightSignal for this patient with missing data...")
        run_patient_process(rate, patient_id, mechanism, seed=seed)

    elif mechanism == "MNAR":
        if seed is None:
            raise ValueError("MNAR requires a seed value.")

        missing_df = mnar(df, rate, seed=seed)
        nan_count = missing_df[HR_COLUMN].isna().sum()
        actual_rate = nan_count / total_rows * 100
        output_path = save_csv_missing(missing_df, patient_id, rate, mechanism, seed=seed)

        print(f"  {mechanism} seed={seed} {rate:>2}%: {nan_count:>5} NaNs / {total_rows} linhas "
            f"= {actual_rate:.1f}% real  →  {output_path}")
        print("Running NightSignal for this patient with missing data...")
        run_patient_process(rate, patient_id, mechanism, seed=seed)

    elif mechanism == "MAR":
        missing_df = mar(df, rate)
        nan_count = missing_df[HR_COLUMN].isna().sum()
        actual_rate = nan_count / total_rows * 100
        output_path = save_csv_missing(missing_df, patient_id, rate, mechanism)

        print(f"  {mechanism} {rate:>2}%: {nan_count:>5} NaNs / {total_rows} linhas "
            f"= {actual_rate:.1f}% real  →  {output_path}")
        print("Running NightSignal for this patient with missing data...")
        run_patient_process(rate, patient_id, mechanism)

    return missing_df

def run_patient_process(taxa: int, patient_id: str, mechanism: str, seed: int | None = None):
    print("Starting NightSignal processing...")
 
    patient_dir = MISSING_DIR / mechanism / f"{taxa}pct"
    if seed is not None:
        patient_dir = patient_dir / str(seed)
    patient_dir = patient_dir / patient_id
    
    if not os.path.exists(patient_dir):
        print(f"Folder not found: {patient_dir}. Skipping...")
        return
 
    # Define o caminho do CSV de entrada
    input_csv = patient_dir / f"{patient_id}_temp.csv"
    
    # Define e cria a pasta de destino
    output_dir = RESULTS_DIR / patient_id / "missing" / mechanism / f"{taxa}pct"
    if seed is not None:
        output_dir = output_dir / str(seed)
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
    
    df_mcar = pd.DataFrame()
    df_mar = pd.DataFrame()
    df_mnar = pd.DataFrame()   

    # Encontra todas as pastas de pacientes em data/processing/
    patients = sorted(p for p in PROCESSING_DIR.iterdir() if p.is_dir())

    if not patients:
        print(f"[WARNING] No patient folder found in {PROCESSING_DIR}.")
        return

    print(f"{'='*55}")
    print(f"Generating MCAR missing data — rates: {RATES}%  |  seeds: {list(MCAR_SEEDS)}")
    print(f"{'='*55}")

    for patient_folder in patients:
        patient_id = patient_folder.name
        input_csv = patient_folder / f"{patient_id}_temp.csv"

        if not input_csv.exists():
            print(f"\n[WARNING] {input_csv} not found, skipping.")
            continue

        for rate in RATES:
            df_mar = process_patient(patient_id, input_csv, "MAR", rate)

            for seed in MCAR_SEEDS:
                df_mcar = process_patient(patient_id, input_csv, "MCAR", rate, seed=seed)
                df_mnar = process_patient(patient_id, input_csv, "MNAR", rate, seed=seed)


                date_end = df_mcar['Datetime'].min() + pd.Timedelta(days=4)
                df_mcar_final = df_mcar[df_mcar['Datetime'] < date_end].copy()
                df_mar_final = df_mar[df_mar['Datetime'] < date_end].copy()
                df_mnar_final = df_mnar[df_mnar['Datetime'] < date_end].copy()

                df_mcar_final.set_index('Datetime', inplace=True)
                df_mar_final.set_index('Datetime', inplace=True)
                df_mnar_final.set_index('Datetime', inplace=True)

                fig, axes = plt.subplots(1, 3, figsize=(20, 6))

                # 1. Plot MCAR (Verde) no primeiro espaço (axes[0])
                # Usamos sparkline=False para evitar bugs de layout quando colocados lado a lado
                msno.matrix(df_mcar_final[[HR_COLUMN]], ax=axes[0], sparkline=False, color=(0.2, 0.8, 0.2), freq='12h', fontsize=10)
                axes[0].set_title(f"MCAR - {rate}pct - seed {seed}", fontsize=16)

                # 2. Plot MAR (Vermelho) no segundo espaço (axes[1])
                msno.matrix(df_mar_final[[HR_COLUMN]], ax=axes[1], sparkline=False, color=(0.8, 0.2, 0.2), freq='12h', fontsize=10)
                axes[1].set_title(f"MAR - {rate}pct", fontsize=16)

                # 3. Plot MNAR (Azul) no terceiro espaço (axes[2])
                msno.matrix(df_mnar_final[[HR_COLUMN]], ax=axes[2], sparkline=False, color=(0.2, 0.2, 0.8), freq='12h', fontsize=10)
                axes[2].set_title(f"MNAR - {rate}pct - seed {seed}", fontsize=16)

                # Ajusta os espaçamentos para os títulos não encavalarem
                plt.tight_layout()

                # Salva por taxa e seed para não sobrescrever resultados.
                plot_dir = RESULTS_DIR / patient_id / "missing" / "MNAR" / f"{rate}pct" / str(seed)
                plot_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(plot_dir / f"{patient_id}_mm.png", dpi=300, bbox_inches='tight')

                # Exibe a imagem na tela para você conferir na hora
                plt.show()


if __name__ == "__main__":
    main()