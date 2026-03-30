import pandas as pd
import numpy as np
import os
import subprocess
import shutil

def process_fitbit_data(hr_file, st_file, output_file):
    """
    Agrega os dados brutos: média de batimentos por minuto (arredondado para cima no .5)
    e soma de passos por minuto.
    """
    # carregando dados, ele já lê a primeira linha e a considera o cabeçalho
    df_hr = pd.read_csv(hr_file)
    df_st = pd.read_csv(st_file)

    # conversão para datetime
    df_hr['datetime'] = pd.to_datetime(df_hr['datetime'])
    df_st['datetime'] = pd.to_datetime(df_st['datetime'])

    # definindo que o índice das linhas não vão ser números, mas sim a coluna datetime
    df_hr.set_index('datetime', inplace=True)
    # cria as divisões de 1 em 1 minuto e calcula a média dos batimentos naquele minuto
    hr_minute = df_hr['heartrate'].resample('1min').mean()
    # nessa parte que posso trabalhar com os dados faltantes
    
    
    df_st.set_index('datetime', inplace=True)
    st_minute = df_st['steps'].resample('1min').sum()

    # unindo
    # crio um dataframe com hr_minute e st_minute
    merged = pd.concat([hr_minute, st_minute], axis=1) # pegando índice e alinha as linhas com mesmo minuto
    merged['steps'] = merged['steps'].fillna(0).astype(int)
    # astype converte pra int, fillna coloca 0 onde não há medida de passos
    merged = merged.dropna(subset=['heartrate'])
    merged.reset_index(inplace=True) # índice volta a ser 0, 1, 2...
    
    # outro dataframe
    final = pd.DataFrame()
    # pegando primeira coluna do dataframe merge
    final['Datetime'] = merged['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # --- LÓGICA DE ARREDONDAMENTO PARA CIMA (.5 -> 1) ---
    # Somamos 0.5 e usamos floor para garantir que 62.5 vire 63 e 62.4 vire 62
    final['HR_Value'] = np.floor(merged['heartrate'] + 0.5).astype(int)
    
    final['f0_'] = final['Datetime']
    final['ST_Value'] = merged['steps']

    final.to_csv(output_file, index=False)
    return output_file

def run_patient_process(patient_id):
    """
    Executa o NightSignal para um paciente específico.
    """
    hr_file = f"{patient_id}-hr.csv"
    st_file = f"{patient_id}-st.csv"
    unified_file = f"{patient_id}_rhr_temp.csv"

    if not os.path.exists(hr_file) or not os.path.exists(st_file):
        return

    print(f">>> Processando Paciente: {patient_id}")

    # Processamento com a nova regra de arredondamento
    process_fitbit_data(hr_file, st_file, unified_file)

    # Execução do NightSignal
    command = [
        "python3", "nightsignal.py",
        "--device=Fitbit",
        f"--restinghr={unified_file}"
    ]
    subprocess.run(command, capture_output=True, text=True)

    # Organização de arquivos
    if os.path.exists("NS-signals.json"):
        shutil.move("NS-signals.json", f"{patient_id}_signals.json")
    if os.path.exists("NightSignalResult.pdf"):
        shutil.move("NightSignalResult.pdf", f"{patient_id}_plot.pdf")

    if os.path.exists(unified_file):
        os.remove(unified_file)
    
    print(f">>> Finalizado: {patient_id}\n")

def main():
    all_files = os.listdir('.')
    patient_ids = sorted(list(set(f.split("-")[0] for f in all_files if f.endswith("-hr.csv"))))

    print(f"Iniciando processamento de {len(patient_ids)} pacientes com arredondamento 'Round Half Up'.\n")

    for pid in patient_ids:
        run_patient_process(pid)

if __name__ == "__main__":
    main()