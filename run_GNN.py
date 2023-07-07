import torch
from torch.utils.data import DataLoader
from _3_Data_Import.import_data import import_runtimes_graphs
from _4_Neural_Networks.gnn_graph import GNN

runtimes, data = import_runtimes_graphs()
GNN(runtimes, data)
