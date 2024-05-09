from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

#Carregando dados do dataset iris
iris_data= load_iris()
x = iris_data.data
y = iris_data.target

#Dividindo o dataset em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.1, random_state=42)

#Criado modelo de decisão
modelo_decisao = DecisionTreeClassifier()
#Treinando o modelo usando o dataset de treino
modelo_decisao.fit(x_treino, y_treino)

#Testando o modelo treinado com o dataset deteste
modelo_test = modelo_decisao.predict(x_teste)

# Calcular a acurácia do modelo
accuracy = accuracy_score(y_teste, modelo_test)
print("Acurácia do modelo KNN:", accuracy)
