import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


#Traemos los datos que vamos a utilizar
dataframe_test = pd.read_csv('https://raw.githubusercontent.com/Club-datos-FCEN/ClubDeDatos/main/01%20-%20Titanic/titanic_test.csv')

print(dataframe_test)

dataframe_train = pd.read_csv('https://raw.githubusercontent.com/Club-datos-FCEN/ClubDeDatos/main/01%20-%20Titanic/titanic_train.csv')

print(dataframe_train)

#Verifico la cantidad de datos que hay en cada dataset

print(dataframe_test.shape)

print(dataframe_train.shape)


#Verifico el tipo de datos contenida en ambos dataset

print(dataframe_test.info())

print(dataframe_train.info())

#Verifico los datos faltantes de los dataset

print(pd.isnull(dataframe_test).sum())

print(pd.isnull(dataframe_train).sum())

#Estadisticas de los dataset

print(dataframe_test.describe())

print(dataframe_train.describe())

#Cambio los datos de sexos por numeros

dataframe_test['Sex'].replace(['female', 'male'], [0, 1], inplace = True)

dataframe_train['Sex'].replace(['female', 'male'], [0, 1], inplace = True)

#Cambio los datos de embarque por numeros

dataframe_test['Embarked'].replace(['Q', 'S', 'C'], [0, 1, 2], inplace = True)

dataframe_train['Embarked'].replace(['Q', 'S', 'C'], [0, 1, 2], inplace = True)


#Reemplazo los datos de edad por la media de esta columna

print(dataframe_test['Age'].mean())

print(dataframe_train['Age'].mean())

promedio = 30

dataframe_test['Age'] = dataframe_test['Age'].replace(np.nan, promedio)

dataframe_train['Age'] = dataframe_train['Age'].replace(np.nan, promedio)

#Creo varios grupos de acuerdo a los rangos de edad 
#Rangos de edad: 0-8, 9-15, 16-18, 19-25, 26-40, 41-60, 61-100 

rango_edad = [0, 8,  15, 18, 25, 40, 60, 100]

names = ['1', '2', '3', '4', '5', '6', '7']

dataframe_test['Age'] = pd.cut(dataframe_test['Age'], rango_edad, labels=names)

dataframe_train['Age'] = pd.cut(dataframe_train['Age'], rango_edad, labels=names)

#Eliminamos la columna "Cabin" ya que tiene muchos datos perdidos y tambien elimino columnas que no son necesarias

dataframe_test.drop(['Cabin'], axis = 1, inplace = True)

dataframe_train.drop(['Cabin'], axis = 1, inplace = True)

dataframe_test = dataframe_test.drop(['PassengerId', 'Name', 'Ticket'], axis= 1)

dataframe_train = dataframe_train.drop(['Name', 'Ticket'], axis= 1)


#Modelos de Machine learning

#Separo la columna con la información de los sobrevivientes

X = np.array(dataframe_train.drop(['Survived'], 1))

y = np.array(dataframe_train['Survived'])

#Separo los datos en entrenamiento y prueba 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Modelo de Regresión Logística
algoritmo = LogisticRegression()
algoritmo.fit(X_train, y_train)
print('La precision del clasificador de Regresión Logística en el set de entrenamiento es: {:.2f}'
     .format(algoritmo.score(X_train, y_train)))
Y_pred = algoritmo.predict(X_test)



#Modelo SVC

algoritmo2 = SVC()
algoritmo2.fit(X_train, y_train)
print('La precision del clasificador SVC en el set de entrenamiento es: {:.2f}'
     .format(algoritmo2.score(X_train, y_train)))
Y_pred2 = algoritmo2.predict(X_test)


#Modelo de Vecinos más Cercanos

algoritmo3 = KNeighborsClassifier(n_neighbors=3)
algoritmo3.fit(X_train, y_train)
print('La precision del clasificador K-NN en el set de entrenamiento es: {:.2f}'
     .format(algoritmo3.score(X_train, y_train)))
Y_pred3 = algoritmo3.predict(X_test)



#Prediccion utilizando los modelos

ids = dataframe_test['PassengerId']

#Regresión Logística

predicción_logreg = algoritmo.predict(dataframe_test.drop('PassengerId', axis = 1))

out_logreg = pd.DataFrame({'PassengerId':ids, 'Survived': predicción_logreg})

print(out_logreg)


#SVC
predicción_svc = algoritmo2.predict(dataframe_test.drop('PassengerId', axis = 1))

out_svc = pd.DataFrame({'PassengerId':ids, 'Survived': predicción_svc})

print(out_svc)


#Vecinos más Cercanos
predicción_knn = algoritmo3.predict(dataframe_test.drop('PassengerId', axis = 1))

out_knn = pd.DataFrame({'PassengerId':ids, 'Survived': predicción_knn})

print(out_knn)



