from matplotlib import pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt

#importando um dataset
data_frame = pd.read_csv("FuelConsumptionCo2.csv")

print(data_frame.head(10))
print(data_frame.describe)

#selecionando as fetures que vou utilizar durante o treino e test
potencia_dos_motores = data_frame[['ENGINESIZE']]
emissao_co2 = data_frame[["CO2EMISSIONS"]]
print(potencia_dos_motores.head())

#dividindo o dataset em teste e treino
motores_treino, motores_test, co2_treino, co2_test = train_test_split(potencia_dos_motores, emissao_co2, test_size=0.2, random_state=40)

#Criando um modelo de regressão linear  utilizando  a biblioteca scikit-learn
modelo_de_predicao = linear_model.LinearRegression()

#Fazendo o treino do modelo baseado no dataset de treino
modelo_de_predicao.fit(motores_treino,  co2_treino)

print('(A) Intercepto: ', modelo_de_predicao.intercept_)
print('(B) Inclinação: ', modelo_de_predicao.coef_)

#Visualização da reta usando o dataset de treino usando gráficos
plt.scatter(motores_treino, co2_treino, color='blue')
plt.plot(motores_treino, modelo_de_predicao.coef_[0][0]*motores_treino + modelo_de_predicao.intercept_[0], '-r')
plt.ylabel("Emissão de C02")
plt.xlabel("Motores")
plt.title("Gráfico de treino")
plt.show()

#Fazendo a previsão de novos dados usando o dataset de teste
predicao = modelo_de_predicao.predict(motores_test)

#Visualização da reta usando o dataset de teste usando gráficos
plt.scatter(motores_test, co2_test, color='green')
plt.plot(motores_test, modelo_de_predicao.coef_[0][0]*motores_test + modelo_de_predicao.intercept_[0], '-r')
plt.ylabel("Emissão de C02")
plt.xlabel("Motores")
plt.title("Gráfico de teste")
plt.show()

#avaliando o modelo
print("Soma dos Erros ao Quadrado (SSE): %.2f " % np.sum((predicao - co2_test)**2))
print("Erro Quadrático Médio (MSE): %.2f" % mean_squared_error(co2_test, predicao))
print("Erro Médio Absoluto (MAE): %.2f" % mean_absolute_error(co2_test, predicao))
print ("Raiz do Erro Quadrático Médio (RMSE): %.2f " % sqrt(mean_squared_error(co2_test, predicao)))
print("R2-score: %.2f" % r2_score(co2_test, predicao))