from matplotlib import pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt


def dividir_Dados():
    # Importando um dataset
    data_frame = pd.read_csv("FuelConsumptionCo2.csv")

    # Selecionando as features que vou utilizar durante o treino e teste
    potencia_dos_motores = data_frame[['ENGINESIZE']]
    emissao_co2 = data_frame[["CO2EMISSIONS"]]

    # Dividindo o dataset em teste e treino
    return train_test_split(potencia_dos_motores, emissao_co2, test_size=0.2, random_state=40)

def criando_modelo(motores_treino,co2_treino):
    # Criando um modelo de regressão linear utilizando a biblioteca scikit-learn
    modelo_de_predicao = linear_model.LinearRegression()

    # Fazendo o treino do modelo baseado no dataset de treino
    modelo_de_predicao.fit(motores_treino, co2_treino)

    return modelo_de_predicao

def avaliação_de_modelo(modelo_de_predicao, motores_treino, co2_treino, motores_test, co2_test):
    print('(A) Intercepto: ', modelo_de_predicao.intercept_)
    print('(B) Inclinação: ', modelo_de_predicao.coef_)

    # Visualização da reta usando o dataset de treino usando gráficos
    plt.scatter(motores_treino, co2_treino, color='blue')
    plt.plot(motores_treino, modelo_de_predicao.predict(motores_treino), '-r')
    plt.ylabel("Emissão de C02")
    plt.xlabel("Motores")
    plt.title("Gráfico de treino")
    plt.show()

    # Métricas de avaliação
    predicao_treino = modelo_de_predicao.predict(motores_treino)
    predicao_teste = modelo_de_predicao.predict(motores_test)

    r2_treino = r2_score(co2_treino, predicao_treino)
    r2_teste = r2_score(co2_test, predicao_teste)

    mse_treino = mean_squared_error(co2_treino, predicao_treino)
    mse_teste = mean_squared_error(co2_test, predicao_teste)

    mae_treino = mean_absolute_error(co2_treino, predicao_treino)
    mae_teste = mean_absolute_error(co2_test, predicao_teste)

    rmse_treino = sqrt(mse_treino)
    rmse_teste = sqrt(mse_teste)

    print("R² (treino): {:.2f}".format(r2_treino))
    print("R² (teste): {:.2f}".format(r2_teste))

    print("MSE (treino): {:.2f}".format(mse_treino))
    print("MSE (teste): {:.2f}".format(mse_teste))

    print("MAE (treino): {:.2f}".format(mae_treino))
    print("MAE (teste): {:.2f}".format(mae_teste))

    print("RMSE (treino): {:.2f}".format(rmse_treino))
    print("RMSE (teste): {:.2f}".format(rmse_teste))
