# Importando as bibliotecas que irei usar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importando o dataset que sera analisado

dataset = pd.read_csv('C:/Users/allan/OneDrive/Área de Trabalho/Udemy - Curso/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(x[:3])
print(y[:3])

y = y.reshape(len(y), 1)  # Muda a matriz para um formato vertical, para que fique igaul a matriz de x

print(y[:3])

from sklearn.preprocessing import StandardScaler

# É necessario criar dois objetos pois dessa forma ele deixa especificamente para cada variavel, e nao mistura as
# medias, medianas, desvio padrao, etc, das dua variaveis

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

print('Novo formato de X e Y')
print(x[:3])
print(y[:3])

# Treinando o modelo SVR no dataset

from sklearn.svm import SVR

regressor = SVR(kernel='rbf')  # O paramatro de kernel é o melhor para o treinamento do modelo SVR
regressor.fit(x, y)

# Prevendo um novo resultado (salario)

sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))

# Vizualizando em formato de grafico

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='black')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color='red')
plt.title('SVR')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()
