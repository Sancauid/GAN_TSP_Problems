import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('_2_Data_Storage/data/runtimes.csv', delimiter=';', dtype={'runtime': np.float32})
data = df.drop(columns=["nodes", "instance"])


# calculate the summary statistics
mean = np.mean(data)
median = np.median(data)
minimum = np.min(data)
maximum = np.max(data)
rango = np.ptp(data)
std_deviation = np.std(data)

# print the summary statistics
print("Mean: ", mean)
print("Median: ", median)
print("Minimum: ", minimum)
print("Maximum: ", maximum)
print("Range: ", rango)
print("Standard Deviation: ", std_deviation)


for numero in range(0, 101):
    percentile = np.percentile(data, numero)
    print("Percentile {}: {}".format(numero, percentile))