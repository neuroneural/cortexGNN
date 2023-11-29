import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Model": ["PialNN", "PialGCN", "PialGCN", "PialGCN", "PialGCN", "PialGCN", "PialGCN"],
    "Memory (GiB)": [5.27, 2.18, 2.95, 3.72, 4.48, 5.24, 6.01],
    "GNN Layers": [0, 2, 3, 4, 5, 6, 7]
}

df = pd.DataFrame(data)

# Creating a seaborn bar plot
plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=df, x='GNN Layers', y='Memory (GiB)', hue='Model', palette='pastel')

# Annotating each bar with the corresponding model name
for index, row in df.iterrows():
    bar = barplot.patches[index]
    height = bar.get_height()
    barplot.annotate(row['Model'], 
                     (bar.get_x() + bar.get_width() / 2., height / 2), 
                     ha='center', va='center', 
                     xytext=(0, 0), textcoords='offset points',
                     rotation=90, fontsize=10)

# Adding an arrow for 'Lower is better'
plt.annotate('Lower is better', xy=(0.5, 1), xytext=(0.5, 2),
             arrowprops=dict(facecolor='green', shrink=0.05, headlength=10, headwidth=15),
             ha='center', fontsize=12)

# Adding labels and title with increased font size
plt.xlabel('GNN Layers',fontsize=22)
plt.ylabel('Memory (GiB)',fontsize=22)
plt.title('Memory Usage by Model and GNN Layers During Training', fontsize=22)
plt.legend(fontsize='large')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig("train_memory.png", format='png')
plt.savefig("train_memory.svg", format='svg')
plt.close()
