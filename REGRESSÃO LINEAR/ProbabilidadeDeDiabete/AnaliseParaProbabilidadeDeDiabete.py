import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


diabete = load_diabetes()
x = diabete.data
y = diabete.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

modelo_diabet = LinearRegression()
modelo_diabet.fit(X_train, y_train)


mse = mean_squared_error(y_test, modelo_diabet.predict(X_test))
print("Erro quadrático médio:", mse)

plt.scatter(y_test, modelo_diabet.predict(X_test), color='black')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='blue', linewidth=2) 
plt.xlabel('Valor Real')
plt.ylabel('Valor Previsto')
plt.title('Regressão Linear - Diabetes')
plt.show()