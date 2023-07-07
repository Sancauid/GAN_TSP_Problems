import numpy as np
import pandas as pd
from random import randint
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense

# MAE
def mae(targets, predictions):
    return np.mean(np.abs(targets - predictions))

# MSE
def mse (targets, predictions):
    return np.mean(np.square(targets - predictions))

# R-squared
def r_squared(targets, predictions):
    ss_res = np.sum(np.square(np.subtract(targets, predictions)))
    ss_tot = np.sum(np.square(np.subtract(targets, np.mean(targets))))
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

# MAPE (Mean Absolute Percentage Error)
def mape(targets, predictions):
    return np.mean(np.abs((targets - predictions) / targets)) * 100 if np.sum(targets) != 0 else 0

# RMSE (Root Mean Squared Error)
def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(np.subtract(targets, predictions))))

# Explained Variance
def explained_variance(targets, predictions):
    return 1 - np.var(targets - predictions) / np.var(targets) if np.var(targets) != 0 else 0

# Definimos el input shape, el modelo y lo compilamos
def modelo(input_shape):
    
    model = keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        # Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Importamos la información
df = pd.read_csv('data/runtimes.csv', delimiter=';', dtype={'runtime': np.float32})
runtimes = df.drop(columns=["nodes", "instance"])
runtimes = np.log(runtimes)

# Separamos los datos
train_ratio = 0.8
test_ratio = 0.2

split_index1 = int(train_ratio * len(runtimes))
split_index2 = int((train_ratio) * len(runtimes))

train_data = runtimes[:split_index1]
test_data = runtimes[split_index2:]

# Instanciamos el modelo
input_shape = (469, 469, 3)
model = modelo(input_shape)

# Entrenamos al modelo en base al train data
for i in range(len(train_data)):
    print(i)
    img_path = f'data/imgs/tsp_nodes.28.{i}.png'
    img = keras.preprocessing.image.load_img(img_path, target_size=input_shape[:2])
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    runtime = train_data.loc[i]
    model.fit(img_array, [runtime], epochs=10, verbose=0)

# El modelo entrenado hace las predicciones sobre el set de entrenamiento
predictions_train = []
targets_train = []
for i in range(len(train_data)):
    img_path = f'data/imgs/tsp_nodes.28.{i}.png'
    img = keras.preprocessing.image.load_img(img_path, target_size=input_shape[:2])
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    runtime = train_data.loc[i]
    target = runtime
    target = np.array([target])
    prediction = model.predict(img_array)[0][0]
    predictions_train.append(prediction)
    targets_train.append(target)

predictions_train = np.array(predictions_train)
targets_train = np.array(targets_train)
print("RESULTADOS SET DE ENTRENAMIENTO TSP SIN SOLUCIÓN:\n")
print("MAE:", mae(targets_train, predictions_train))
print("MSE:", mse(targets_train, predictions_train))
print("R-squared:", r_squared(targets_train, predictions_train))
print("MAPE:", mape(targets_train, predictions_train))
print("RMSE:", rmse(targets_train, predictions_train))
print("Explained Variance:", explained_variance(targets_train, predictions_train))

# El modelo entrenado hace las predicciones sobre el set de testeo
predictions_test = []
targets_test = []
for i in range(len(test_data)):
    img_path = f'data/imgs/tsp_nodes.28.{split_index2 + i}.png'
    img = keras.preprocessing.image.load_img(img_path, target_size=input_shape[:2])
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    runtime = test_data.loc[split_index2 + i]
    target = runtime
    target = np.array([target])
    prediction = model.predict(img_array)[0][0]
    predictions_test.append(prediction)
    targets_test.append(target)

predictions_test = np.array(predictions_test)
targets_test = np.array(targets_test)
print("RESULTADOS SET DE TESTEO TSP SIN SOLUCIÓN:\n")
print("MAE:", mae(targets_test, predictions_test))
print("MSE:", mse(targets_test, predictions_test))
print("R-squared:", r_squared(targets_test, predictions_test))
print("MAPE:", mape(targets_test, predictions_test))
print("RMSE:", rmse(targets_test, predictions_test))
print("Explained Variance:", explained_variance(targets_test, predictions_test))
