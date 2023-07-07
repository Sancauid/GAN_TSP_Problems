import numpy as np
from _3_Data_Import.import_data import import_runtimes_graphs
from _6_Adversarial_Generation.model import build_gan, train_gan, build_generator, build_discriminator

runtimes, data = import_runtimes_graphs()

adj_matrices = []
for row in data:
    x, y = row
    distancia = np.hypot(x[:,None] - x, y[:,None] - y).round(1)
    np.fill_diagonal(distancia, 0)
    max_value = np.max(distancia)
    distancia_normalized = distancia / max_value
    adj_matrices.append(distancia_normalized)
    
adj_matrices_array = np.array(adj_matrices)

runtimes = runtimes['runtime']
runtimes = runtimes.to_numpy()
runtimes = runtimes.reshape(-1, 1)

latent_dim = 32
num_nodes = 60
epochs = 20
batch_size = 1048

generator = build_generator(latent_dim, num_nodes)
discriminator = build_discriminator(num_nodes)
gan = build_gan(generator, discriminator)

train_gan(generator, discriminator, gan, adj_matrices_array, runtimes, latent_dim, epochs, batch_size)
