from gurobipy import Model, GRB, quicksum # type: ignore
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import math

# Generar Nodos
n = 25
nodos = [ i for i in range(n) ]
samples = 50
repeticiones = 10

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

    runtime_minimo = float('inf')
    runtime_maximo = 0
    runtime_promedio = 0
    runtime_primero = 0
    runtime_ultimo = 0

    for misma_instancia in range(repeticiones):
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

        # Tiempo de ejecución y solución
        runtime = model.Runtime
        arcos_activos = [ i for i in arcos if x[i].x == 1 ]

        if runtime <= runtime_minimo:
            runtime_minimo = runtime
        if runtime >= runtime_maximo:
            runtime_maximo = runtime
        if misma_instancia == 0:
            runtime_primero = runtime
        if misma_instancia == repeticiones - 1:
            runtime_ultimo = runtime
        runtime_promedio += runtime

        # No se utiliza el mismo modelo
        model.dispose()
    
    runtime_promedio = runtime_promedio / repeticiones
    diferencia_promedio_minimo = (runtime_promedio - runtime_minimo) / runtime_promedio * 100
    diferencia_promedio_maximo = (runtime_maximo - runtime_promedio) / runtime_maximo * 100

    with open('_2_Data_Storage/same_instance_runtimes.csv', mode='a') as file:
        file.write(f'{runtime_primero},{runtime_ultimo},{runtime_promedio},{diferencia_promedio_minimo},{diferencia_promedio_maximo}\n')




