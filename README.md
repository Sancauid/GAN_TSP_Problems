# GAN_TSP_Problems

## Generación adversarial de problemas del vendedor viajero complejos para Gurobi 

En el siguiente repositorio se encuentra el código generado para esta IPre. Para correr cualquiera de las dos redes neuronales, se deben utilizar los archivos run_GNN y run_GAN para la GNN y GAN respectivamente. Estos mostrarán prints en pantalla para poder ver el progeso de entrenamiento de la red neuronal, aparte de las instancias generadas por la GAN.

## _1_Data_Generation

En estos archivos se encuentran los generadores de instancias del TSP en Gurobi, para utilizar se debe tener descargada la libería Gurobi. Al acceder al archivo data_gen_scf, se pueden modificar la cantidad de nodos e instancias que se deciden crear al correr el código, estas serán posteriormente resueltas utilizando Single Commodity Flow y guardadas en _2_Data_Storage.

## _2_Data_Storage

Acá se guardan las instancias creadas, junto a un visualizador de la información generada. Para visualizar las instancias se puede correr graph.py y para obtener estadísticas de ellas se puede correr stats.py

## _3_Data_Import

Esta carpeta está encargada de importar las instancias almacenadas y es luego importado por ambas redes para ser utilizadas.

## _4_Neural_Networks

Aquí se encuentra implementada la GNN o Graph Neural Network, esta puede ser corrida en el archivo run_GNN.py que se encuentra en el directorio raíz.

## _5_Results_GNN

Acá están los resultados de la GNN, junto a los hiperparámetros utilizados para cada uno de los resultados.

## _6_Adversarial_Generation

En esta carpeta está el modelo de la GAN, que puede ser corrido con el archivo run_GAN.py, en el directorio raíz

## _7_Results_GAN

En el archivo txt se encuentran los resultados obtenidos finalmente con la GAN.



