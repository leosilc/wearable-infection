import pandas as pd
import numpy as np
from mdatagen.multivariate.mMCAR import mMCAR

data = {'A': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'B': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}
df_teste = pd.DataFrame(data)
y_dummy = np.zeros(len(df_teste))

print("--- Dados Originais ---")
print(df_teste)

# BUG 1 CORRIGIDO: missing_rate deve ser INTEIRO entre 0 e 99
# 0.4 é interpretado como 0% (truncamento interno), use 40
taxa = 10

generator = mMCAR(X=df_teste, y=y_dummy, missing_rate=taxa, seed=42)

# BUG 2 CORRIGIDO: .dataset é só uma cópia de X sem NaN nenhum
# Os NaNs existem APENAS no valor de retorno de .random()
df_faltante = generator.random()  # ← era generator.dataset

print(f"\n--- Dados com MCAR ({taxa}%) ---")
print(df_faltante)

print("\nTotal de valores ausentes encontrados:", df_faltante.isnull().sum().sum())