import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#Temperaturas

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
kelvin = np.array([233, 263, 273, 281, 288, 295, 311], dtype=float)


#capa = tf.keras.layers.Dense(units=1, input_shape=[1])
#modelo = tf.keras.Sequential([capa])

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])



modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

#Entrenamiento del modelo
print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, kelvin, epochs=1000, verbose=False)
print("Modelo entrenado!")

#Grafico para ver en que momento nuestro modelo deja de aprender
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

#Prediccion usando el modelo
print("Hagamos una predicción!")
resultado = modelo.predict([300.0])
print("El resultado es " + str(resultado) + " Kelvin!")

