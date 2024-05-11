import analise_emissoes_co2
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

# Solicitando ao usuário o tamanho do motor para fazer uma previsão
tamanho_motor = float(input("Insira o tamanho do motor para fazer uma previsão (em litros): "))

# Dividindo os dados
motores_treino, motores_test, co2_treino, co2_test = analise_emissoes_co2.dividir_Dados()

# Criando o modelo
modelo = analise_emissoes_co2.criando_modelo(motores_treino, co2_treino)

# Fazendo a previsão com base no tamanho do motor inserido pelo usuário
predicao_personalizada = modelo.predict([[tamanho_motor]])

print("Para um motor de tamanho %.2f litros, a previsão de emissão de CO2 é de %.2f g/km." % (tamanho_motor, predicao_personalizada[0][0]))

# Avaliando o modelo
analise_emissoes_co2.avaliação_de_modelo(modelo, motores_treino, co2_treino, motores_test, co2_test)

