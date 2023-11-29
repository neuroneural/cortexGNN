import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np




# Assuming the CSV file is named 'evaluation_stats_ohbm.csv' and is located in the current directory
csv_file = 'evaluation_stats_ohbm.csv'

# Reading the data from the CSV file
df = pd.read_csv(csv_file)

# Adding a combined column for Model and Layers
df["Model_Layers"] = df["Model"] + ", " + df["Layers"].astype(str)

# Extract layers as a separate numeric column for sorting
df['Layers_num'] = df['Layers']

# Pivot the DataFrame for mean values
pivot_mean_df = df.pivot_table(index="Model_Layers", columns="Hemisphere", values="HD_Mean", aggfunc='mean').reset_index()

# Pivot the DataFrame for standard deviation values
pivot_std_df = df.pivot_table(index="Model_Layers", columns="Hemisphere", values="HD_Std", aggfunc='mean').reset_index()

# Melt the DataFrames
melted_mean_df = pivot_mean_df.melt(id_vars="Model_Layers", var_name="Hemisphere", value_name="HD_Mean")
melted_std_df = pivot_std_df.melt(id_vars="Model_Layers", var_name="Hemisphere", value_name="HD_Std")

# Adding a 'Layers' column to the melted DataFrames
melted_mean_df['Layers'] = melted_mean_df['Model_Layers'].str.extract('(\d+)', expand=False).astype(int)
melted_std_df['Layers'] = melted_std_df['Model_Layers'].str.extract('(\d+)', expand=False).astype(int)

# Sort by Layers
sorted_mean_df = melted_mean_df.sort_values(by='Layers')
sorted_std_df = melted_std_df.sort_values(by='Layers')

# Inferno colormap for the bars
inferno = plt.cm.get_cmap("inferno", 12)  # Divide the colormap into 12 parts
left_color = inferno(2)  # 1st quartile color
very_dark_right_color = inferno(0.75)  # A color near the end of the colormap

# Font size settings
axis_label_font_size = 16
title_font_size = 20
legend_font_size = 16
axis_ticks_font_size = 16  # Adjust this for axis ticks

# Plotting
plt.figure(figsize=(15, 8))
ax = sns.barplot(x="Model_Layers", y="HD_Mean", hue="Hemisphere", data=sorted_mean_df, palette=[left_color, very_dark_right_color])

# Adding error bars
for i in range(len(sorted_mean_df)):
    mean_row = sorted_mean_df.iloc[i]
    std_row = sorted_std_df[(sorted_std_df['Model_Layers'] == mean_row['Model_Layers']) & (sorted_std_df['Hemisphere'] == mean_row['Hemisphere'])].iloc[0]
    #print(std_row)
    if i%2 == 0:
        x_pos = i/2.0-0.203125
    else:
        x_pos = i/2.0-0.303125
    ax.errorbar(x=x_pos, y=mean_row['HD_Mean'], yerr=std_row['HD_Std'], fmt='none', c='black', capsize=5)

arrowprops = dict(arrowstyle="->", color="green")

plt.xticks(rotation=45, ha="right")


# Annotation with a green down arrow and text
plt.annotate('Lower is better', xy=(0.7, 0.6), xycoords='axes fraction',
             xytext=(0.7, .8), textcoords='axes fraction',
             arrowprops=dict(facecolor='green', shrink=0.05,headwidth=40),
             ha='center', fontsize=20, va='bottom')


plt.xlabel("Model, GNNLayers", fontsize=axis_label_font_size)
plt.ylabel("HD Mean", fontsize=axis_label_font_size)
ax.tick_params(axis='both', which='major', labelsize=axis_ticks_font_size)  # Set font size for axis ticks
plt.title("Quality of Mesh predictions on Test set (Hausdorff Distance)", fontsize=title_font_size)
plt.legend(fontsize=legend_font_size)
plt.ylim(0.5, .9)  # Setting y-axis limits
plt.tight_layout()
plt.savefig('evalplot.png')
