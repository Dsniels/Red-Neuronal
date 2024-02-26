from tabnanny import verbose
from unittest import result
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from unratedwriting import typewrite



celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#Capa que conecta cada neurona hacia toda las neuronas de la siguiente capa 
capa = tf.keras.layers.Dense(units = 1, input_shape=[1])
#modelo para trabajar con la capa
modelo = tf.keras.Sequential([capa])
#prepara el modelo para ser entrenado
modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error',
)

typewrite('Entrenando modelo...')
historial = modelo.fit(celsius, fahrenheit, epochs = 1000, verbose = False) #Entrenamiendo del modelo
typewrite('Modelo Entrenado!')

plt.xlabel("#Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])

plt.show()

typewrite('Haciendo Prediccion')
prediction = np.array([100.0])
result = str(modelo.predict(prediction))
typewrite(f'El resultado es {result}° fahrenheit!')
typewrite('Variables internas')
print(capa.get_weights())

""" formula para C° a F°: C*1.8 + 32 """


