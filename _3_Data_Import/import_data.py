import pandas as pd
import numpy as np
import ast

# Importar runtimes y formatear
def import_runtimes():
    df = pd.read_csv('_2_Data_Storage/data/runtimes.csv', delimiter=';', dtype={'runtime': np.float32})
    runtimes = df.drop(columns=["nodes", "instance"])
    
    # Calcular percentiles
    p1 = np.quantile(runtimes, 0.42)
    p2 = np.quantile(runtimes, 0.99)
    
    # Filtrar
    runtimes_filtered = runtimes[(runtimes >= p1) & (runtimes <= p2)]
    runtimes_filtered = np.log(runtimes_filtered)
    runtimes_filtered = runtimes_filtered.dropna(axis=0)

    runtimes_filtered = runtimes_filtered[: int(len(runtimes_filtered) * 0.2) ]
        
    # Utilizar para importar diccionario   
    # distancia = ast.literal_eval(distancia)
    return runtimes_filtered

def import_runtimes_graphs():
    graphs = pd.read_csv('_2_Data_Storage/data/graphs.csv')
    graphs["X"] = graphs["X"].apply(lambda x: x.split(";", 1)[1])
    graphs["X"] = graphs["X"].apply(lambda x: x.split(";", 1)[1])

    graphs["X"] = graphs["X"].apply(lambda x: [int(node) for node in x.split(";")])
    graphs["Y"] = graphs["Y"].apply(lambda x: [int(node) for node in x.split(";")])

    X = np.array([np.array(xi) for xi in graphs["X"]])
    Y = np.array([np.array(yi) for yi in graphs["Y"]])
    data = np.stack((X, Y), axis=1)

    filtered_runtimes = import_runtimes()
    data_final = []
    contador = 0
    for indice in filtered_runtimes.index:
        data_final.append(data[indice])

    return filtered_runtimes, data_final