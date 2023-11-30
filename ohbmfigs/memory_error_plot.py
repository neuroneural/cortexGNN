import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from datalh
datalh = {
    "Memory (GiB)": [5.27, 2.18, 2.95, 3.72, 4.48, 5.24, 6.01],
    "Mean_HausDorff_Distance": [0.8173626012139238, 0.7873215395214104, 0.727640451787874, 0.6917642264185416,
                                0.6608719310933958, 0.6544264991489462, 0.649583875901824],
    "Std_HausDorff_Distance": [0.07652121328331668, 0.07086618406315326, 0.07189662065855185, 0.06385394218189502,
                               0.05627032247982908, 0.06410625158794886, 0.06572587136659902],
    "Model": ["PialNN", "PialGCN", "PialGCN", "PialGCN", "PialGCN", "PialGCN", "PialGCN"],
    "Layers": ["  0 Layers", "  2 Layers", "  3 Layers", "  4 Layers", " 5 Layers", "  6 Layers", "  7 Layers"]
}

df = pd.DataFrame(datalh)

# Creating the scatter plot with error bars
plt.figure(figsize=(10, 10))
ax = sns.scatterplot(data=df, x='Memory (GiB)', y='Mean_HausDorff_Distance', hue='Model', style='Model', s=100)

# Adding error bars
for i in range(len(df)):
    ax.errorbar(x=df['Memory (GiB)'][i], y=df['Mean_HausDorff_Distance'][i], 
                yerr=df['Std_HausDorff_Distance'][i], fmt='none', c='black', capsize=5)

# Annotating each point with the corresponding layer text
for i in range(len(df)):
    plt.text(df['Memory (GiB)'][i], df['Mean_HausDorff_Distance'][i], df['Layers'][i], 
             horizontalalignment='left', size='medium', color='black', weight='semibold')

# Ideal model annotation
ax.annotate('Ideal model', xy=(2, .57), xytext=(.3, .15),
            textcoords='axes fraction', arrowprops=dict(facecolor='green', shrink=0.05),
            fontsize=16, ha='right')

# Adding labels and title
plt.xlabel('Training Memory (GiB)', fontsize=16)
plt.ylabel('Hausdorff Distance', fontsize=16)
plt.title('Mean Hausdorff Distance with Memory Usage by Model (LH)', fontsize=20)
plt.legend(fontsize='large')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig("memory_error.png", format='png')
plt.savefig("memory_error.svg", format='svg')
plt.close()
