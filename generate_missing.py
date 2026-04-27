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
TAXAS = [10]

SEED = 42

# Nome da coluna que receberá os dados ausentes
COLUNA_HR = "HR_Value"

def mnar(df: pd.DataFrame, taxa: int) -> pd.DataFrame:
    # implementar
    print("MNAR ainda não implementado")

def mar(df: pd.DataFrame, taxa: int) -> pd.DataFrame:
    # implementar
    print("MAR ainda não implementado")



def mcar(df: pd.DataFrame, taxa: int) -> pd.DataFrame:

    # Recebe o DataFrame original e devolve uma CÓPIA com alguns valores de HR_Value substituídos por NaN, de forma completamente aleatória (MCAR).

    df_out = df.copy()

    # uMCAR exige:
    #   X = DataFrame com a coluna que vai receber os valores ausentes
    #   y = array numpy qualquer (MCAR não usa rótulos — passamos zeros)
    #   missing_rate 
    #   x_miss = nome da coluna que vai receber os NaNs
    #   seed 

    X = df_out[[COLUNA_HR]].astype(float)
    y = np.zeros(len(X))

    gerador = uMCAR(X=X, y=y, missing_rate=taxa, x_miss=COLUNA_HR, seed=SEED)

    # .random() é o método que gera os NaNs e os RETORNA num novo DataFrame.
    X_com_ausencia = gerador.random()


    #adicionando a coluna de Steps e Batimentos ao final
    df_out["HR_MISSING"] = df[COLUNA_HR].astype(int)  # cópia da coluna original para referência
    df_out["ST_MISSING"] = df["ST_Value"]#.astype(int) # se precisar ser float em algum momento


    # Substitui a coluna HR no DataFrame completo
    df_out[COLUNA_HR] = X_com_ausencia[COLUNA_HR].values

    # # Converte mantendo os NaN como nulos, mas tratando o resto como inteiro
    df_out[COLUNA_HR] = df_out[COLUNA_HR].astype('Int64')
   

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
    pasta = MISSING_DIR / mechanism / f"{taxa}pct" / patient_id
    pasta.mkdir(parents=True, exist_ok=True)

    caminho = pasta / f"{patient_id}_temp.csv"

    # Datetime de volta para string antes de salvar
    df_salvar = df.copy()
    df_salvar["Datetime"] = df_salvar["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # na_rep="" → campo vazio no CSV em vez da palavra "nan"
    df_salvar.to_csv(caminho, index=False, na_rep="")

    return caminho


def processar_paciente(patient_id: str, csv_entrada: Path, mechanism: str):
    """
    Fluxo completo para um paciente:
      1. Lê o CSV processado
      2. Salva cópia original (uma vez)
      3. Para cada taxa: aplica MCAR e salva CSV com ausência
    """
    print(f"\nPaciente: {patient_id}")

    # Lê o CSV — HR e Steps como float para aceitar NaN depois
    df = pd.read_csv(csv_entrada, dtype={COLUNA_HR: float, "ST_Value": int})
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    total_linhas = len(df)
    print(f"  {total_linhas} linhas carregadas de {csv_entrada}")

    # # Salva cópia do original (referência para comparação futura)
    # salvar_original(csv_entrada, patient_id)
    # print(f"  Original copiado para: data/missing/original/{patient_id}/")

    # Gera um CSV para cada taxa
    for taxa in TAXAS:
        df_ausente = mcar(df, taxa)

        nan_count = df_ausente[COLUNA_HR].isna().sum()
        taxa_real = nan_count / total_linhas * 100

        caminho_saida = save_csv_missing(df_ausente, patient_id, taxa, mechanism)

        print(f"  {mechanism} {taxa:>2}%: {nan_count:>5} NaNs / {total_linhas} linhas "
              f"= {taxa_real:.1f}% real  →  {caminho_saida}")
        print("Rodando NightSignal para este paciente com os dados ausentes...")
        run_patient_process(taxa, patient_id, mechanism)



def run_patient_process(taxa: int, patient_id: str, mechanism: str):
    print("Iniciando processamento do NightSignal...")
 

    diretorio_paciente = MISSING_DIR / mechanism / f"{taxa}pct" / patient_id
    
    if not os.path.exists(diretorio_paciente):
        print(f"Pasta não encontrada: {diretorio_paciente}. Pulando...")
        return
 
    # Define o caminho do CSV de entrada
    input_csv = diretorio_paciente / f"{patient_id}_temp.csv"
    
    # Define e cria a pasta de destino
    output_dir = RESULTS_DIR / patient_id / "missing" / mechanism / f"{taxa}pct"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(input_csv):
        print(f"  > Executando {patient_id}...")
        
        # COMANDO PARA RODAR O NIGHTSIGNAL
        try:
            command = [
                "python3", "nightsignal.py",
                "--device=Fitbit",
                f"--restinghr={input_csv}"
            ]
            subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Erro no paciente {patient_id}: {e}")
    else:
        print(f"CSV não encontrado para {patient_id}")


    # organizando arquivos gerados
    if os.path.exists("NS-signals.json"):
        shutil.move("NS-signals.json", output_dir / f"{patient_id}_signals.json")
    
    if os.path.exists("NightSignalResult.pdf"):
        shutil.move("NightSignalResult.pdf", output_dir / f"{patient_id}_plot.pdf")

        

print("\n✅ Operação concluída! Verifique a pasta 'results'.")


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    if not PROCESSING_DIR.exists():
        print(f"[ERRO] Pasta '{PROCESSING_DIR}' não encontrada.")
        print("Execute o run_all.py primeiro para gerar os CSVs processados.")
        return

    # Encontra todas as pastas de pacientes em data/processing/
    pastas_pacientes = sorted(
        p for p in PROCESSING_DIR.iterdir() if p.is_dir()
    )

    if not pastas_pacientes:
        print(f"[AVISO] Nenhuma pasta de paciente encontrada em {PROCESSING_DIR}.")
        return

    print(f"{'='*55}")
    print(f"Gerando ausências MCAR — taxas: {TAXAS}%  |  seed: {SEED}")
    print(f"{'='*55}")

    for pasta in pastas_pacientes:
        patient_id = pasta.name
        csv_entrada = pasta / f"{patient_id}_temp.csv"

        if not csv_entrada.exists():
            print(f"\n[AVISO] {csv_entrada} não encontrado, pulando.")
            continue

        processar_paciente(patient_id, csv_entrada, "MCAR")



if __name__ == "__main__":
    main()