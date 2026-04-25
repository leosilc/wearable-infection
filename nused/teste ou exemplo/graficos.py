import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Simulação de como seus dados devem estar organizados
# No seu caso, você vai ler os JSONs de resultado e montar este DataFrame
data = {
    'Cenario': ['Original']*84 + ['10% Falha']*84 + ['30% Falha']*84 + ['50% Falha']*84,
    'Dia_Alerta': [
        # Exemplos de valores (Lead Time)
        # Original (mais antecipado, ex: -4, -3)
        *pd.Series([-4, -3, -5, -2, -1]).sample(84, replace=True),
        # 10% Falha (começa a atrasar)
        *pd.Series([-3, -2, -1, 0]).sample(84, replace=True),
        # 30% Falha (atrasa mais)
        *pd.Series([-2, -1, 0, 1]).sample(84, replace=True),
        # 50% Falha (muitos alertas no dia do sintoma ou depois)
        *pd.Series([-1, 0, 1, 2, 3]).sample(84, replace=True)
    ]
}

df_resultados = pd.DataFrame(data)

# 2. Configuração visual do Seaborn
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# 3. Criação do Boxplot
# O 'order' garante que os cenários fiquem na sequência lógica de degradação
plot = sns.boxplot(
    data=df_resultados, 
    x='Cenario', 
    y='Dia_Alerta', 
    palette='vlag',
    order=['Original', '10% Falha', '30% Falha', '50% Falha']
)

# 4. Adicionando uma linha de referência no Dia 0 (Início dos Sintomas)
plt.axhline(0, color='red', linestyle='--', label='Início dos Sintomas')

# 5. Ajuste de labels e título
plt.title('Impacto da Ausência de Dados no Tempo de Antecipação (NightSignal)', fontsize=14)
plt.xlabel('Percentual de Dados Faltantes (Simulado)', fontsize=12)
plt.ylabel('Dia do Alerta (Relativo ao Sintoma)', fontsize=12)
plt.legend()

# Salvar o gráfico para o seu relatório
plt.savefig('boxplot_ic_leonardo.png', dpi=300, bbox_inches='tight')