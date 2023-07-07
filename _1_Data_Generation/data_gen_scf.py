from gurobipy import Model, GRB, quicksum # type: ignore
from random import randint
import numpy as np
import matplotlib.pyplot as plt

# Generar Nodos
n = 20
nodos = [ i for i in range(n) ]
samples = 100

def plot(x_values, y_values, runtime):
    # Plot sin Resultado
    plt.figure(figsize=(5,5))
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.axis('off')

    plt.scatter(x=x_values, y=y_values, color='black', zorder=1)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    img_name = f"tsp_nodes.{n}.{instancia}.png"
    plt.savefig(f"_2_Data_Storage/data_parche/imgs2/{img_name}", bbox_inches='tight') 


def sol_plot(x_values, y_values, arcos_activos, runtime):
    # Plot con Resultado
    plt.figure(figsize=(5,5))
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.axis('off')

    plt.scatter(x=x_values, y=y_values, color='black', zorder=1)

    for i, j in arcos_activos:
        plt.plot([x_values[i], x_values[j]], [y_values[i], y_values[j]], color='black', alpha=0.4, zorder=0)
    
    img_name = f"tsp_solution.{n}.{instancia}.png"
    plt.savefig(f"_2_Data_Storage/data_parche/imgs2/{img_name}", bbox_inches='tight')

def scf(arcos, distancia):

    # Generar el modelo
    model = Model()

    # Se instancian variables de decision
    x = model.addVars(nodos, nodos, vtype=GRB.BINARY, name='x')
    y = model.addVars(nodos, nodos, lb=0, vtype=GRB.CONTINUOUS, name='y')

    # Llamamos a update
    model.update()

    # Restricciones
    model.addConstrs(quicksum(x[i,j] for j in nodos if j != i) == 1 for i in nodos)
    model.addConstrs(quicksum(x[i,j] for i in nodos if i != j) == 1 for j in nodos)

    # Subtours
    model.addConstrs(quicksum(y[i,j] for j in nodos if j != i) - quicksum(y[j,i] for j in nodos if j != 0 and j != i) == 1 for i in nodos if i != 0)
    model.addConstrs(y[i,j] <= (n - 1) * x[i, j] for i in nodos if i != 0 for j in nodos if j != i)

    # Funcion Objetivo y optimizar el problema
    objetivo = quicksum(distancia[n] * x[n] for n in arcos)
    model.setObjective(objetivo, GRB.MINIMIZE)
    model.optimize()

    runtime = model.Runtime
    arcos_activos = [[i, j] for i in nodos for j in nodos if x[i, j].x > 0]

    return runtime, arcos_activos

for instancia in range(samples):

    # En que instancia vamos
    print("Van " + str(instancia) + " instancias")

    # Generar Arcos
    arcos = [ (i, j) for i in nodos for j in nodos if i != j ]

    # Generación de Coordenadas
    X = np.random.randint(101, size=n)
    Y = np.random.randint(101, size=n)

    # Distancia entre Nodos    
    distancia = {(i, j): round(np.hypot(X[i] - X[j], Y[i] - Y[j]), 1) if i != j else 0 for i in nodos for j in nodos}
    values_list = [list(distancia[i, j] for j in nodos) for i in nodos]
    

    runtime, arcos_activos = scf(arcos, distancia)

    plot(X, Y, runtime)
    sol_plot(X, Y, arcos_activos, runtime)

    with open('_2_Data_Storage/data_parche/runtimes.csv', mode='a') as file:
        file.write(f'{runtime}\n')

    with open('_2_Data_Storage/data_parche/graphs.csv', mode='a') as file:
        file.write(str(values_list) + '\n')
