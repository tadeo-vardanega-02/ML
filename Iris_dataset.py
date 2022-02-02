import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 


#Analizamos los datos que tenemos disponibles
dataframe = pd.read_csv('https://raw.githubusercontent.com/Club-datos-FCEN/ClubDeDatos/main/02%20-%20Iris/iris.csv')
print(dataframe)

dataframe.describe()
dataframe.columns

#Distribucion de las especies
print(dataframe.groupby('species').size())

#Graficos para entender mejor nuestros datos 

sns.pairplot(dataframe, hue="species")
plt.show()



#Modelos de Machine learning

X = dataframe[[
     'sepal_length', 'sepal_width', 
     'petal_length', 'petal_width']]

y = dataframe['species']

#Separo los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Modelo de Regresión Logística

algoritmo = LogisticRegression()
algoritmo.fit(X_train, y_train)
print('La precision del clasificador de Regresión Logística en el set de entrenamiento es: {:.2f}'
     .format(algoritmo.score(X_train, y_train)))
Y_pred = algoritmo.predict(X_test)
print(Y_pred)


#Modelo SVC

algoritmo2 = SVC()
algoritmo2.fit(X_train, y_train)
print('La precision del clasificador SVC en el set de entrenamiento es: {:.2f}'
     .format(algoritmo2.score(X_train, y_train)))
Y_pred2 = algoritmo2.predict(X_test)
print(Y_pred2)

#Modelo de Vecinos más Cercanos

algoritmo3 = KNeighborsClassifier(n_neighbors=5)
algoritmo3.fit(X_train, y_train)
print('La precision del clasificador K-NN en el set de entrenamiento es: {:.2f}'
     .format(algoritmo3.score(X_train, y_train)))
Y_pred3 = algoritmo3.predict(X_test)
print(Y_pred3)


#Modelo de Árboles de Decisión 

algoritmo4 = DecisionTreeClassifier()
algoritmo4.fit(X_train, y_train)
print('La precision del clasificador de Árboles de Decisión en el set de entrenamiento es: {:.2f}'
     .format(algoritmo4.score(X_train, y_train)))
Y_pred4 = algoritmo4.predict(X_test)
print(Y_pred4)

