"""
generate_mcar.py
================
Lê os CSVs gerados pelo run_all.py (em data/processing/) e cria versões
com dados ausentes (MCAR) em data/missing/.

Execute DEPOIS do run_all.py, quando quiser gerar os cenários de ausência:
    python generate_mcar.py

Estrutura de pastas criada:
    data/
    ├── processing/          ← gerado pelo run_all.py (não tocamos aqui)
    │   └── P001/
    │       └── P001_temp.csv
    └── missing/
        ├── original/        ← cópia dos CSVs originais (referência)
        │   └── P001/
        │       └── P001_temp.csv
        └── MCAR/
            ├── 10pct/
            │   └── P001/
            │       └── P001_temp.csv   ← 10% dos valores de HR removidos
            ├── 30pct/
            │   └── P001/
            │       └── P001_temp.csv
            └── 50pct/
                └── P001/
                    └── P001_temp.csv
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from mdatagen.univariate.uMCAR import uMCAR


PROCESSING_DIR = Path("data/processing")
MISSING_DIR = Path("data/missing")

# Taxas de ausência que serão geradas
TAXAS = [10, 30, 50]

SEED = 42

# Nome da coluna que receberá os dados ausentes
COLUNA_HR = "HR_Value"


# =============================================================================
# FUNÇÕES
# =============================================================================

def salvar_original(csv_entrada: Path, patient_id: str):
    """
    Copia o CSV original para data/missing/original/
    Serve como referência para comparar os resultados depois.
    Só faz a cópia uma vez — se já existir, pula.
    """
    destino = MISSING_DIR / "original" / patient_id / csv_entrada.name
    if destino.exists():
        return  # já copiado em execução anterior

    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_entrada, destino)


def aplicar_mcar(df: pd.DataFrame, taxa: int) -> pd.DataFrame:
    """
    Recebe o DataFrame original e devolve uma CÓPIA com alguns valores
    de HR_Value substituídos por NaN, de forma completamente aleatória (MCAR).

    Como o uMCAR funciona internamente?
    ------------------------------------
    1. Calcula quantos NaNs inserir: N = round(n_linhas * taxa / 100)
    2. Sorteia N posições aleatórias no array de HR (sem repetição)
    3. Substitui essas posições por NaN
    4. Devolve o DataFrame modificado
    """
    df_out = df.copy()

    # uMCAR exige:
    #   X = DataFrame só com colunas numéricas (a coluna que vai receber NaN)
    #   y = array numpy qualquer (MCAR não usa rótulos — passamos zeros)
    #   missing_rate = INTEIRO entre 1 e 99 (não decimal: 30, não 0.3)
    #   x_miss = nome da coluna que vai receber os NaNs
    #   seed = para reprodutibilidade
    X = df_out[[COLUNA_HR]].astype(float)
    y = np.zeros(len(X))

    gerador = uMCAR(X=X, y=y, missing_rate=taxa, x_miss=COLUNA_HR, seed=SEED)

    # .random() É o método que gera os NaNs e os RETORNA num novo DataFrame.
    X_com_ausencia = gerador.random()

    # Substitui a coluna HR no DataFrame completo
    df_out[COLUNA_HR] = X_com_ausencia[COLUNA_HR].values

    return df_out


def salvar_csv_ausente(df: pd.DataFrame, patient_id: str, taxa: int):
    """
    Salva o CSV com ausências em data/missing/MCAR/{taxa}pct/{patient_id}/

    O formato das colunas é IDÊNTICO ao original:
        Datetime, HR_Value, f0_, ST_Value

    Isso garante que o NightSignal lê o arquivo sem nenhuma modificação —
    ele vai encontrar campo vazio onde havia um valor de HR e, com a pequena
    correção no nightsignal.py, vai pular esse minuto no cálculo da média.

    NaN é escrito como campo vazio ("") — padrão CSV para dado ausente.
    """
    pasta = MISSING_DIR / "MCAR" / f"{taxa}pct" / patient_id
    pasta.mkdir(parents=True, exist_ok=True)

    caminho = pasta / f"{patient_id}_temp.csv"

    # Datetime de volta para string antes de salvar
    df_salvar = df.copy()
    df_salvar["Datetime"] = df_salvar["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # na_rep="" → campo vazio no CSV em vez da palavra "nan"
    df_salvar.to_csv(caminho, index=False, na_rep="")

    return caminho


def processar_paciente(patient_id: str, csv_entrada: Path):
    """
    Fluxo completo para um paciente:
      1. Lê o CSV processado
      2. Salva cópia original (uma vez)
      3. Para cada taxa: aplica MCAR e salva CSV com ausência
    """
    print(f"\nPaciente: {patient_id}")

    # Lê o CSV — HR e Steps como float para aceitar NaN depois
    df = pd.read_csv(csv_entrada, dtype={COLUNA_HR: float, "ST_Value": float})
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    total_linhas = len(df)
    print(f"  {total_linhas} linhas carregadas de {csv_entrada}")

    # Salva cópia do original (referência para comparação futura)
    salvar_original(csv_entrada, patient_id)
    print(f"  Original copiado para: data/missing/original/{patient_id}/")

    # Gera um CSV para cada taxa
    for taxa in TAXAS:
        df_ausente = aplicar_mcar(df, taxa)

        nan_count = df_ausente[COLUNA_HR].isna().sum()
        taxa_real = nan_count / total_linhas * 100

        caminho_saida = salvar_csv_ausente(df_ausente, patient_id, taxa)

        print(f"  MCAR {taxa:>2}%: {nan_count:>5} NaNs / {total_linhas} linhas "
              f"= {taxa_real:.1f}% real  →  {caminho_saida}")


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

        processar_paciente(patient_id, csv_entrada)



if __name__ == "__main__":
    main()