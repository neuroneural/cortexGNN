import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math

# Data
data = {
    "Model": ["PialNN", "CortexGCN", "CortexGCN", "CortexGCN", "CortexGAT", "CortexGAT", "CortexGAT"],
    "Memory (GiB)": [5.4, 0.92, 1.13, 1.33, 1.69, 2.62, 3.55],
    "GNN Layers": [0, 2, 3, 4, 2, 3, 4]
}

df = pd.DataFrame(data)

# Creating a seaborn bar plot
plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=df, x='GNN Layers', y='Memory (GiB)', hue='Model', palette='pastel')

# Fixing the annotation logic
for i, bar in enumerate(barplot.patches):
    height = bar.get_height()
    if not math.isnan(height):
        model_name = df[df['Memory (GiB)'] == float(height)]['Model'].iloc[0]
        text_position = height / 2  # Centered vertically inside the bar
        barplot.annotate(model_name, 
                         (bar.get_x() + bar.get_width() / 2., text_position), 
                         ha='center', va='center', 
                         xytext=(0, 0), textcoords='offset points',
                         rotation=90, fontsize=10)  # Increase fontsize as needed

# Adjust x-axis limits to reduce white space between groups
x_min, x_max = barplot.get_xlim()
barplot.set_xlim(x_min - 0.5, x_max + 0.5)

plt.annotate('Lower is better', xy=(0.5, 1), xytext=(0.5, 2),
             arrowprops=dict(facecolor='green', shrink=0.05, headlength=10, headwidth=15),
             ha='center', fontsize=12)
# Adding labels and title with increased font size
plt.xlabel('GNN Layers',fontsize=22)
plt.ylabel('Memory (GiB)',fontsize=22)
plt.title('Memory Usage by Model and GNN Layers During Training', fontsize=22)  # Increase fontsize as needed
plt.legend(fontsize='large')  # You can specify 'small', 'medium', 'large', '
plt.xticks(fontsize=12)  # Change font size for x-axis tick labels
plt.yticks(fontsize=12)  # Change font size for y-axis tick labels

plt.savefig("train_memory.png", format='png')
plt.close()
