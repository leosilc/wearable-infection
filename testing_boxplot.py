import pandas as pd
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURAÇÃO ---
# Caminho para o CSV que você mostrou no print
CAMINHO_CSV = "data/SourceData_COVID19_Positives.csv" # Ajuste o nome real do arquivo se necessário
PASTA_RESULTADOS = Path("results")      # Onde ficam as pastas PXXXXXX
COLUNA_ID = "Participant ID"
COLUNA_DATA = "COVID-19 Symptom Onset"

# 1. Carregar os metadados
df_meta = pd.read_csv(CAMINHO_CSV)
df_meta.columns = df_meta.columns.str.strip()

# FORÇANDO O FORMATO: %Y-%m-%d (Ano-Mês-Dia)
# Isso impede que o Python inverta o 04 com o 11
df_meta[COLUNA_DATA] = pd.to_datetime(
    df_meta[COLUNA_DATA], 
    format='%Y-%m-%d', 
    errors='coerce'
)

#precisa? 
# Remove linhas que falharam na conversão
df_meta = df_meta.dropna(subset=[COLUNA_DATA])

# teste debug: mostra as 3 primeiras datas convertidas para você conferir no terminal
print("Amostra das datas convertidas:")
print(df_meta[COLUNA_DATA].head(10))

#tirar ate aqui


dados_para_plot = []

for index, row in df_meta.iterrows():
    p_id = str(row[COLUNA_ID])
    data_sintoma = row[COLUNA_DATA]
    caminho_json = PASTA_RESULTADOS / p_id / f"{p_id}_signals.json"
    
    if caminho_json.exists():
        with open(caminho_json, 'r') as f:
            signals = json.load(f)
        
        alert_entries = signals.get("nightsignal", [])
        dias_com_alerta = []
        
        for entry in alert_entries:
            if entry.get("val") == "1":
                data_alerta = pd.to_datetime(entry.get("date"))
                diff = (data_alerta - data_sintoma).days
                
                # DEBUG: Remova o '#' da linha abaixo para ver as diferenças no terminal
                # print(f"ID: {p_id} | Alerta: {data_alerta.date()} | Sintoma: {data_sintoma.date()} | Diff: {diff}")
                
                if -14 <= diff <= 7:
                    dias_com_alerta.append(diff)
        
        if dias_com_alerta:
            primeiro_alerta = min(dias_com_alerta)
            dados_para_plot.append({'ID': p_id, 'Lead_Time': primeiro_alerta})

# --- PARTE DO GRÁFICO COM PROTEÇÃO ---
if not dados_para_plot:
    # print("\n[AVISO]: Nenhum alerta foi encontrado dentro da janela de -14 a +7 dias.")
    # print("Verifique se as datas do CSV e do JSON pertencem ao mesmo ano!")
else:
    df_final = pd.DataFrame(dados_para_plot)
    plt.figure(figsize=(6, 8))
    sns.set_theme(style="whitegrid")

    sns.boxplot(y=df_final['Lead_Time'], color="#a1c9f4", width=0.4)
    sns.swarmplot(y=df_final['Lead_Time'], color="#d62728", alpha=0.7)

    plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Início dos Sintomas')
    plt.title(f"Alert Initiation (Base Original)\nN = {len(df_final)} pacientes")
    plt.ylabel("Dias Relativos ao Início do Sintoma")
    plt.ylim(-15, 5)

    plt.savefig("boxplot_original_dataset.png", dpi=300)
    print(f"\nGráfico gerado para {len(df_final)} pacientes.")