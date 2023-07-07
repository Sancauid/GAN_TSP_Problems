import sys
import numpy as np
from multiprocessing import Pool
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import  Input, Dense, Dropout, LayerNormalization, Flatten, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import normalize
import math


def GNN(runtimes, data):

    adj_matrices = []
    for row in data:
        x, y = row
        distancia = np.hypot(x[:,None] - x, y[:,None] - y).round(1)
        np.fill_diagonal(distancia, 0)
        adj_matrices.append(distancia)

    adj_matrices_array = np.array(adj_matrices)
    adj_matrices_array = np.array([normalize(adj_matrix) for adj_matrix in adj_matrices_array])

    runtimes = runtimes['runtime']
    runtimes = runtimes.to_numpy()
    runtimes = runtimes.reshape(-1, 1)

    # matrices_array_testeo = adj_matrices_array[len(adj_matrices_array) * 0.1:]
    # adj_matrices_array = adj_matrices_array[:len(adj_matrices_array) * 0.9]
    # runtimes_testeo = runtimes[len(runtimes) * 0.1:]
    # runtimes = runtimes[:len(runtimes) * 0.9]

    def gcn_model(num_hidden=1024, num_layers=5, l2_reg=0.001, learning_rate=1e-4):

        # Define input layer
        inputs = Input(shape=(60, 60))

        # Define hidden layers
        x = Dense(num_hidden, activation='gelu', kernel_regularizer=l2(l2_reg))(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        for i in range(num_layers-1):
            x = Dense(num_hidden, activation='gelu', kernel_regularizer=l2(l2_reg))(x)
            x = LayerNormalization()(x)
            x = Dropout(0.2)(x)

        # Reducir a un valor
        x = Flatten()(x)

        # Define output layer
        output_shape = (1)
        outputs = Dense(output_shape, activation='linear')(x)
        outputs = relu(outputs)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compilar Modelo
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model


    # Initialize GCN model
    model = gcn_model()

    # Define early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)

    # Train model
    model.fit(adj_matrices_array, runtimes, validation_split=0.4, epochs=50, batch_size=64, callbacks=[early_stop])

    y_pred = model.predict(adj_matrices_array)

    for numero in range(100):
        print("REAL: ", runtimes[numero], "PREDICCIÓN: ", y_pred[numero])

    # Calculate MAPE
    mape = np.mean(np.abs((runtimes - y_pred) / runtimes)) * 100

    print("MAPE:")
    print(mape)
