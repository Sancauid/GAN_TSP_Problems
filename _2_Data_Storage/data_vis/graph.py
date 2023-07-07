import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('_2_Data_Storage/data/runtimes.csv', delimiter=';', dtype={'runtime': np.float32})
runtimes = df.drop(columns=["nodes", "instance"])

# Create a box plot of the runtimes
plt.boxplot(runtimes["runtime"])
# Set the y label
plt.ylabel('Runtime')
# Show the plot
plt.show()


# Create a violin plot of the runtimes
plt.violinplot(runtimes["runtime"])
# Set the y label
plt.ylabel('Runtime')
# Show the plot
plt.show()