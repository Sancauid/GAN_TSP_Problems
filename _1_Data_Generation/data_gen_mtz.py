from gurobipy import Model, GRB, quicksum # type: ignore
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import math

# Generar Nodos
n = 30
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
    plt.savefig(f"_2_Data_Storage/imgs/{img_name}", bbox_inches='tight') 


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
    plt.savefig(f"_2_Data_Storage/imgs/{img_name}", bbox_inches='tight')


for instancia in range(samples):

    # En que instancia vamos
    print("Van " + str(instancia) + " instancias")

    # Generar Arcos
    arcos = [ (i, j) for i in nodos for j in nodos if i != j ]

    # Generación de Coordenadas
    X = np.random.randint(101, size=n)
    Y = np.random.randint(101, size=n)

    # Distancia entre Nodos
    distancia = { (i, j): np.hypot(X[i] - X[j], Y[i] - Y [j]) for i in nodos for j in nodos if i != j }

    # Generar el modelo
    model = Model()
    model.setParam(GRB.Param.TimeLimit, 60)

    # Se instancian variables de decision
    x = model.addVars(arcos, vtype = GRB.BINARY, name='x')

    # Se instancian variables adicionales
    u = model.addVars(nodos, lb=0.0, ub=(n-1), vtype=GRB.CONTINUOUS, name='u')

    # Llamamos a update
    model.update()

    # Restricciones
    model.addConstrs(quicksum(x[i, j] for j in nodos if j != i) == 1 for i in nodos)
    model.addConstrs(quicksum(x[i, j] for i in nodos if i != j) == 1 for j in nodos)

    # Agregamos las restricciones de MTZ2
    model.addConstrs(u[i] - u[j] + n * x[i, j] <= n - 1 for i, j in arcos if i != 0 and j != 0)
    model.addConstrs(u[i] >= 0 for i in nodos)
    model.addConstrs(u[i] <= n - 1 for i in nodos)

    # Funcion Objetivo y optimizar el problema
    objetivo = quicksum(distancia[n] * x[n] for n in arcos)
    model.setObjective(objetivo, GRB.MINIMIZE)
    model.optimize()

    runtime = model.Runtime
    arcos_activos = [ i for i in arcos if x[i].x == 1 ]

    plot(X, Y, runtime)
    sol_plot(X, Y, arcos_activos, runtime)

    with open('_2_Data_Storage/data_parche/runtimes.csv', mode='a') as file:
        file.write(f'{runtime}\n')

    combined_list = ','.join([';'.join(map(str, X)), ';'.join(map(str, Y))])
    with open('_2_Data_Storage/data_parche/graphs.csv', mode='a') as file:
        file.write(combined_list + "\n")
