import pandas as pd 
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt

df_irrigaçao = pd.read_csv("D:/CURSOS/Machine Learning/Desafios/dados_de_irrigacao.csv")
print(df_irrigaçao)

# Análise Exploratoria - EDA #

# Verificar Valores Nulos
print(df_irrigaçao.isna().sum())

# Verificar a estrutura do dataset
print(df_irrigaçao.info())


# calculos estatisticos
media = df_irrigaçao[['Horas de Irrigação', 'Área Irrigada', 'Área Irrigada por Ângulo']].mean()
print("Média:\n", media)

# estatisticas descritivas Total 
print(df_irrigaçao.describe())

# Gráfico de dispersão
# X = Irrigação
# Y = Area Irrigada por angulo

# Gráfico de Dispersão
sns.scatterplot(data=df_irrigaçao, x='Horas de Irrigação', y='Área Irrigada', color='green', s=50)
plt.title("Relação entre Horas de Irrigação e Área Irrigada")
plt.xlabel("Horas de Irrigação")
plt.ylabel("Área Irrigada (m²)")
plt.grid(True)
plt.show()

# Correlações
plt.figure(figsize=(8, 6))
sns.heatmap(df_irrigaçao.corr(method='pearson'), annot=True)
plt.title('Correlação (Pearson)')
plt.show()

# Correlação entre as váriaveis 
# O gráfico mostra uma correlação linear positiva forte quanto mais horas de irrigação, maior a área irrigada.

# Verificar / detectar outliers
sns.boxplot(data=df_irrigaçao, x='Horas de Irrigação')
plt.show()

# Verificar / detectar outliers
sns.boxplot(data=df_irrigaçao, x='Área Irrigada por Ângulo')
plt.show()

## Treinar Modelo - Regressão Linear 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro, kstest, zscore
import statsmodels.api as sm


# Dividir o dataset entre treino e teste
X = df_irrigaçao['Horas de Irrigação'].values.reshape(-1,1)
y = df_irrigaçao['Área Irrigada por Ângulo'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciar o modelo
reg_model = LinearRegression()

# Treinar o Modelo
reg_model.fit(X_train, y_train)

# Equação da reta
coeficiente = reg_model.coef_[0][0]
intercepto = reg_model.intercept_[0]
print(f"A equação da reta é: y = {coeficiente}x + {intercepto}")

# Validação do Modelo

# Predição dos valores com base no conjunto de testes
y_pred = reg_model.predict(X_test)

# Métrica MAE (Mean Absolute Error) e MSE (Mean Squared Error)
mse = ((y_test - y_pred) ** 2).mean()
mae = (abs(y_test - y_pred)).mean()

print(f"MSE: {mse}")
print(f"MAE: {mae}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=range(len(y_test)), y=y_test.flatten(), color='blue', label='Valores Reais')
sns.scatterplot(x=range(len(y_test)), y=y_pred.flatten(), color='red', label='Valores Preditos')
plt.legend()
plt.show()

# Análise dos Résiduos
residuos = y_test - y_pred
residuos_std = zscore(residuos)

sns.scatterplot(x=y_pred.reshape(-1), y=residuos_std.reshape(-1))
plt.axhline(y=0)
plt.show()

# Teste de Normalidade - Shapiro Wilk
# H0 - Segue distribuição normal
# H1 - Não segue distribuição normal
# Se o p-valor > 0.05 não rejeita H0, caso contrário rejeitamos
stat_shapiro, p_valor_shapiro = shapiro(residuos.reshape(-1))
print("Estatística do teste: {} e P-Valor: {}".format(stat_shapiro, p_valor_shapiro))

# Fazer predições com o modelo
horas_exemplo = ([[15]])
area_predita = reg_model.predict(horas_exemplo)
print(f"Para 15 horas de irrigação, a área irrigada por ângulo prevista é: {area_predita[0][0]}")

""" 

Insights Gerais sobre a Análise

A relação entre as horas de irrigação e a área irrigada por ângulo é totalmente linear, quase perfeita
visivelmente na reta obtida.

No desempenho pelas métricas de erro(MSE E MAE) mostram que o modelo linear conseguiu previsões
bem próximas do real.

Na Análise de Residuos não evidencia suficiente para dizer que os residuos fogem da normalidade.

O modelo está apto para ser usado para fazer previsões sobre a área irrigada por ângulo dada 
uma quantidade específica de horas de irrigação.

"""

