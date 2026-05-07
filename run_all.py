import pandas as pd
import numpy as np
import os
import subprocess
import shutil
from pathlib import Path

# definindo caminhos
BASE_DIR = Path(".")
DATA_RAW = Path("data/raw")
PROCESSING_DIR = Path("data/processing")
RESULTS_DIR = Path("results")

def process_fitbit_data(hr_file, st_file, output_file):
   
    # carregando dados, ele já lê a primeira linha e a considera o cabeçalho
    hr_df = pd.read_csv(hr_file)
    st_df = pd.read_csv(st_file)

    # conversão para datetime
    hr_df['datetime'] = pd.to_datetime(hr_df['datetime'])
    st_df['datetime'] = pd.to_datetime(st_df['datetime'])

    # definindo que o índice das linhas não vão ser números, mas sim a coluna datetime
    hr_df.set_index('datetime', inplace=True)
    # cria as divisões de 1 em 1 minuto e calcula a média dos batimentos naquele minuto
    hr_minute = hr_df['heartrate'].resample('1min').mean()

    st_df.set_index('datetime', inplace=True)
    st_minute = st_df['steps'].resample('1min').sum()

    # unindo
    # crio um dataframe com hr_minute e st_minute
    merged_df = pd.concat([hr_minute, st_minute], axis=1) # pegando índice e alinha as linhas com mesmo minuto
    merged_df['steps'] = merged_df['steps'].fillna(0).astype(int)
    # astype converte pra int, fillna coloca 0 onde não há medida de passos
    merged_df = merged_df.dropna(subset=['heartrate'])
    merged_df.reset_index(inplace=True) # índice volta a ser 0, 1, 2...
    
    # outro dataframe
    output_df = pd.DataFrame()
    # pegando primeira coluna do dataframe merge
    output_df['Datetime'] = merged_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # arredondando para cima 
    output_df['HR_Value'] = np.floor(merged_df['heartrate'] + 0.5).astype(int)
    
    output_df['f0_'] = output_df['Datetime']
    output_df['ST_Value'] = merged_df['steps']

    output_df.to_csv(output_file, index=False)
    return output_file


# def process_applewatch_data(hr_in, st_in, hr_out, st_out):
#     """
#     Formata os arquivos NonFitbit para o padrão Apple Watch exigido pelo NightSignal.
#     """
#     # batimentos
#     df_hr = pd.read_csv(hr_in)
#     df_hr['datetime'] = pd.to_datetime(df_hr['datetime'])
    
#     hr_final = pd.DataFrame()
#     hr_final['Device'] = df_hr['device']
#     hr_final['Start_Date'] = df_hr['datetime'].dt.strftime('%Y-%m-%d')
#     hr_final['Start_Time'] = df_hr['datetime'].dt.strftime('%H:%M:%S')
#     hr_final['Heartrate'] = df_hr['heartrate']
    
#     hr_final.to_csv(hr_out, index=False)

#     # passos
#     df_st = pd.read_csv(st_in)
#     df_st['start_datetime'] = pd.to_datetime(df_st['start_datetime'])
#     df_st['end_datetime'] = pd.to_datetime(df_st['end_datetime'])
    
#     st_final = pd.DataFrame()
#     st_final['Device'] = df_st['device']
#     st_final['Start_Date'] = df_st['start_datetime'].dt.strftime('%Y-%m-%d')
#     st_final['Start_Time'] = df_st['start_datetime'].dt.strftime('%H:%M:%S')
#     st_final['End_Date'] = df_st['end_datetime'].dt.strftime('%Y-%m-%d')
#     st_final['End_Time'] = df_st['end_datetime'].dt.strftime('%H:%M:%S')
#     st_final['Steps'] = df_st['steps']
    
#     st_final.to_csv(st_out, index=False)

def run_patient_process(patient_folder):
    """
    Execução do algoritmo para um paciente específico.
    """
    
    patient_id = patient_folder.name  # nome da pasta da base é o ID do paciente
    
    # esse é o padrão para dados da FitBit
    fitbit_hr_file = patient_folder / "Orig_Fitbit_HR.csv"
    fitbit_st_file = patient_folder / "Orig_Fitbit_ST.csv"
    fitbit_unified_file = BASE_DIR / f"{patient_id}_rhr_temp.csv" # arquivo temporário para o processamento, dados unificados

    # esse é o padrão para dados da nonFitBit
    nonfitbit_hr_file = patient_folder / "Orig_NonFitbit_HR.csv"
    nonfitbit_st_file = patient_folder / "Orig_NonFitbit_ST.csv"
    
    # criando pastas de resultados e processamento para o paciente
    patient_dir = RESULTS_DIR / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
        
    # define o caminho para os arquivos temporários
    patient_dir_temp = PROCESSING_DIR / patient_id
    patient_dir_temp.mkdir(parents=True, exist_ok=True)
    
    if fitbit_hr_file.exists() and fitbit_st_file.exists():
    
        print(f"Processing FitBit patient: {patient_id}")

        # processando cada paciente
        process_fitbit_data(fitbit_hr_file, fitbit_st_file, fitbit_unified_file)

        # executando algoritmo
        command = [
            "python3", "nightsignal.py",
            "--device=Fitbit",
            f"--restinghr={fitbit_unified_file}"
        ]
        subprocess.run(command, capture_output=True, text=True)
        
    elif nonfitbit_hr_file.exists() and nonfitbit_st_file.exists():
        print(f"Processing NonFitBit patient: {patient_id}")
        
    
        # device_name = identify_device(nonfitbit_hr)
        
        # if device_name and "Apple Watch" in device_name:
        #     print(f"Processando Paciente NonFitBit: {patient_id}")
            
        #     temp_hr = BASE_DIR / f"{patient_id}_apple_hr.csv"
        #     temp_st = BASE_DIR / f"{patient_id}_apple_st.csv"
            
        #     process_applewatch_data(hr_file_nonfitbit, st_file_nonfitbit, temp_hr, temp_st)
            
        #     # Executa NightSignal com os dois arquivos
        #     cmd = ["python3", "nightsignal.py", "--device=AppleWatch", 
        #            f"--restinghr={temp_hr}", f"--steps={temp_st}"]
        #     subprocess.run(cmd, capture_output=True, text=True)

        # # definindo arquivos de saída temporários para o formato Apple Watch
        # hr_out = patient_dir_temp / f"{patient_id}_hr_apple.csv"
        # st_out = patient_dir_temp / f"{patient_id}_st_apple.csv"

        # # processando cada paciente
        # process_applewatch_data(hr_file_nonfitbit, st_file_nonfitbit, hr_out, st_out)

        # # executando algoritmo
        # command = [
        #     "python3", "nightsignal.py",
        #     "--device=AppleWatch",
        #     f"--restinghr={hr_out}",
        #     f"--steps={st_out}"
        # ]
        # subprocess.run(command, capture_output=True, text=True)

    # organizando arquivos gerados
    if os.path.exists("NS-signals.json"):
        shutil.move("NS-signals.json", patient_dir / f"{patient_id}_signals.json")
    
    if os.path.exists("NightSignalResult.pdf"):
        shutil.move("NightSignalResult.pdf", patient_dir / f"{patient_id}_plot.pdf")
        
    # move os arquivos temporários 
    if os.path.exists(fitbit_unified_file):
        shutil.move(fitbit_unified_file, patient_dir_temp / f"{patient_id}_temp.csv")

    
    print(f"Finalizado: {patient_id}\n")

def main():
    """
    Percorre a pasta data/raw procurando subpastas de pacientes.
    Para cada paciente encontrado, executa o processo de unificação dos dados e o algoritmo.
    """
    
    # lista todas as subpastas em data/raw/
    patient_folders = [f for f in DATA_RAW.iterdir() if f.is_dir()]
    
    print(f"Found {len(patient_folders)} patient folders in data/raw/.\n")

    # envia para a função cada pasta de paciente
    for folder in sorted(patient_folders):
        run_patient_process(folder)

if __name__ == "__main__":
    main()