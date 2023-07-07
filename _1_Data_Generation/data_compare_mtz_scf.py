from gurobipy import Model, GRB, quicksum # type: ignore
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import math

# Generar Nodos
n = 25
nodos = [ i for i in range(n) ]
samples = 100
suma_runtimes_mtz = 0
suma_runtimes_scf = 0
iguales = True


def mtz_2(arcos, distancia):
    # Generar el modelo
    model = Model()

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

    valor_optimo = model.objVal

    model.dispose()

    return runtime, arcos_activos, valor_optimo

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

    valor_optimo = model.objVal

    model.dispose()

    return runtime, arcos_activos, valor_optimo


for instancia in range(samples):

    # En que instancia vamos
    print("Van " + str(instancia) + " instancias")

    # Generar Arcos
    arcos = [ (i, j) for i in nodos for j in nodos if i != j ]

    # Generación de Coordenadas
    X = np.random.randint(101, size=n)
    Y = np.random.randint(101, size=n)

    # Distancia entre Nodos
    distancia = {(i, j): round(np.hypot(X[i] - X[j], Y[i] - Y[j]), 1) for i in nodos for j in nodos if i != j}

    runtime1, arcos_activos1, valor_optimo1 = mtz_2(arcos, distancia)
    runtime2, arcos_activos2, valor_optimo2 = scf(arcos, distancia)

    suma_runtimes_mtz += runtime1
    suma_runtimes_scf += runtime2

    print(valor_optimo1)
    print(valor_optimo2)

    print(valor_optimo1 == valor_optimo2)

    if valor_optimo1 != valor_optimo2:
        iguales = False

    # with open('_2_Data_Storage/data_parche/runtimes.csv', mode='a') as file:
    #     file.write(f'{runtime1}, {runtime2}\n')


#with open('_2_Data_Storage/data_parche/runtimes.csv', mode='a') as file:
#        file.write(f'{suma_runtimes_mtz}, {suma_runtimes_scf}\n')


print("TODAS LAS INSTANCIAS TIENEN EL MISMO V.O.")
print(iguales)