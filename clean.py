import shutil
import os

# Caminho da pasta que você quer deletar
remover1 = 'processing'
remover2 = 'results'
remover3 = 'data/processing'


def limpar_pasta(caminho_pasta):
    for item in os.listdir(caminho_pasta):
        caminho_item = os.path.join(caminho_pasta, item)
        try:
            if os.path.isfile(caminho_item) or os.path.islink(caminho_item):
                os.unlink(caminho_item) # Apaga arquivo ou link
            elif os.path.isdir(caminho_item):
                shutil.rmtree(caminho_item) # Apaga subpasta
        except Exception as e:
            print(f'Falha ao deletar {caminho_item}. Motivo: {e}')
            
if __name__ == "__main__":
    if os.path.exists(remover1):
        limpar_pasta(remover1)
        print(f'Pasta "{remover1}" limpa com sucesso.')
        
    if os.path.exists(remover2):
        limpar_pasta(remover2)
        print(f'Pasta "{remover2}" limpa com sucesso.')
        
    if os.path.exists(remover3):
        limpar_pasta(remover3)
        print(f'Pasta "{remover3}" limpa com sucesso.')
    