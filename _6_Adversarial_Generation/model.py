import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def wasserstein_loss(y_true, y_pred):
    return keras.backend.mean(y_true * y_pred)

def build_generator(latent_dim, num_nodes):
    model = keras.Sequential()
    model.add(layers.Dense(128, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_nodes * (num_nodes - 1), activation='sigmoid'))
    model.add(layers.Reshape((num_nodes, num_nodes - 1)))

    def add_diagonal(x):
        diagonal = tf.zeros_like(x[:, :, 0])
        diagonal = tf.expand_dims(diagonal, axis=-1)

        lower_triangle = tf.concat([diagonal, x], axis=-1)
        upper_triangle = tf.transpose(lower_triangle, perm=[0, 2, 1])

        distance_matrix = lower_triangle + upper_triangle
        distance_matrix = distance_matrix * (1 - tf.eye(tf.shape(x)[1]))
        return distance_matrix

    model.add(layers.Lambda(add_diagonal))
    return model

def build_discriminator(num_nodes):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(num_nodes, num_nodes)))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss=wasserstein_loss, optimizer='adam')
    return model


def build_gan(generator, discriminator):
    discriminator.trainable = True
    discriminator.compile(loss=wasserstein_loss, optimizer=keras.optimizers.RMSprop(lr=0.0005))
    discriminator.trainable = False

    model = keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss=wasserstein_loss, optimizer=keras.optimizers.RMSprop(lr=0.0005))
    return model

def train_gan(generator, discriminator, gan, distances, runtimes, latent_dim, epochs, batch_size, n_critic=5, clip_value=0.01):
    num_samples = distances.shape[0]
    num_nodes = distances.shape[1]

    for epoch in range(epochs):
        for _ in range(n_critic):
            idx = np.random.randint(0, num_samples, batch_size)
            real_instances = distances[idx]
            real_runtimes = runtimes[idx]

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            gen_instances = generator.predict(noise)

            d_loss_real = discriminator.train_on_batch(real_instances, -np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(gen_instances, np.ones((batch_size, 1)))
            d_loss = d_loss_fake - d_loss_real

            for layer in discriminator.layers:
                weights = layer.get_weights()
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                layer.set_weights(weights)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, -np.ones((batch_size, 1)))

        if epoch % 20 == 0:
            print(f"Epoch: {epoch}/{epochs}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")
            print(distances[0])
            print(gen_instances[0])

    print("Resultados finales entrenamiento:")
    print(f"Epoch: {epoch+1}/{epochs}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")
    print(distances[0])
    print(gen_instances[0])
    print("Training completado")
