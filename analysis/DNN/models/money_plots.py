import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


z_size = 12.
x_size = 3.
y_size = 3.

target_file = "/home/alma1/GNN/Deepset/28oct_DNN/TowardPIDwithGranularCalorimeters/results/DNN/accuracyTable.csv"
summaryTable = pd.read_csv(target_file, sep=",")

summaryTable['Name'] = summaryTable['Name'].str.replace('^pp_', '', regex=True)
summaryTable.rename(columns={'Name': 'segmentation'}, inplace=True)
summaryTable = summaryTable[summaryTable["accuracy"] != -1]



summaryTable[["segx", "segy", "segz"]] = summaryTable["segmentation"].str.extract(r"(\d+)_(\d+)_(\d+)")
summaryTable["segx"] = summaryTable["segx"].astype(float)
summaryTable["segy"] = summaryTable["segy"].astype(float)
summaryTable["segz"] = summaryTable["segz"].astype(float)

filtered_segz = summaryTable[summaryTable["segz"] == 100]

area_XY = []
accuracy_XY = []
error_XY_lower = []
error_XY_upper = []
if (len(filtered_segz)):
    area_XY = ((100/filtered_segz["segx"]) * (100/filtered_segz["segy"])*x_size*y_size).tolist()
    accuracy_XY = filtered_segz["accuracy"].tolist()
    error_XY_min = filtered_segz["minVal"].tolist()
    error_XY_max = filtered_segz["maxVal"].tolist()
    
    error_XY_lower = [accuracy - error_min for accuracy, error_min in zip(accuracy_XY, error_XY_min)]
    error_XY_upper = [error_max - accuracy for accuracy, error_max in zip(accuracy_XY, error_XY_max)]
    print(error_XY_lower)
    print(error_XY_upper)
    
    area_XY, accuracy_XY, error_XY_lower, error_XY_upper = zip(*sorted(zip(area_XY, accuracy_XY, error_XY_lower, error_XY_upper), key=lambda x: x[0]))

filtered_segx = summaryTable[summaryTable["segx"] == 100]

delta_Z = []
accuracy_Z = []
error_Z_lower = []
error_Z_upper = []
if (len(filtered_segx)):
    
    delta_Z = ((100/filtered_segx["segz"])*z_size).tolist()
    accuracy_Z = filtered_segx["accuracy"].tolist()
    error_Z_min = filtered_segx["minVal"].tolist()
    error_Z_max = filtered_segx["maxVal"].tolist()

    error_Z_lower = [accuracy - error_min for accuracy, error_min in zip(accuracy_Z, error_Z_min)]
    error_Z_upper = [error_max - accuracy for accuracy, error_max in zip(accuracy_Z, error_Z_max)]
    

    delta_Z, accuracy_Z, error_Z_lower, error_Z_upper = zip(*sorted(zip(delta_Z, accuracy_Z, error_Z_lower, error_Z_upper), key=lambda x: x[0]))


volume_XYZ = ((100/summaryTable["segx"]) * (100/summaryTable["segy"])*(100/summaryTable["segz"])*x_size*y_size*z_size).tolist()
accuracy_XYZ = summaryTable["accuracy"].tolist()
error_XYZ_min = summaryTable["minVal"].tolist()
error_XYZ_max = summaryTable["maxVal"].tolist()

error_XYZ_lower = [accuracy - error_min for accuracy, error_min in zip(accuracy_XYZ, error_XYZ_min)]
error_XYZ_upper = [error_max - accuracy for accuracy, error_max in zip(accuracy_XYZ, error_XYZ_max)]
print(error_XYZ_lower)
print(error_XYZ_upper)

volume_XYZ, accuracy_XYZ, error_XYZ_lower, error_XYZ_upper = zip(*sorted(zip(volume_XYZ, accuracy_XYZ, error_XYZ_lower, error_XYZ_upper), key=lambda x: x[0]))

baseline_file = "/home/alma1/GNN/Deepset/28oct_DNN/TowardPIDwithGranularCalorimeters/results/DNN/baseline.tsv"
baselineTable = pd.read_csv(baseline_file, sep="\t")
baselineTable['accuracy']=baselineTable['accuracy']*100.
baselineTable['minVal']=baselineTable['minVal']*100.
baselineTable['maxVal']=baselineTable['maxVal']*100.
base = baselineTable['accuracy'].iloc[0].astype(float)
base_min = baselineTable['minVal'].iloc[0].astype(float)
base_max = baselineTable['maxVal'].iloc[0].astype(float)


# Create the figure with 2 rows and 2 columns
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey=False)

# Define the color for the 1-sigma region
sigma_color = 'red'

# Plot 1: Accuracy vs Area_XY
axes[0, 0].errorbar(
    area_XY, accuracy_XY, 
    yerr=(error_XY_lower, error_XY_upper), 
    fmt='s--', capsize=5, label='Accuracy'
)
axes[0, 0].axhspan(
    base_min, base_max, color=sigma_color, alpha=0.3, label=r'Baseline $\sigma=1$'
)
axes[0, 0].axhline(y=base, color='red', linestyle='--', label='Baseline')  # Add baseline
axes[0, 0].set_xlabel(r'$\Delta_{XY} \,\, [\mathrm{mm}^2]$', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].grid(True, linestyle='--', alpha=0.7)
axes[0, 0].legend()

# Plot 2: Accuracy vs Delta_Z
axes[0, 1].errorbar(
    delta_Z, accuracy_Z, 
    yerr=(error_Z_lower, error_Z_upper), 
    fmt='s--', capsize=5, label='Accuracy', color='orange'
)
axes[0, 1].axhspan(
    base_min, base_max, color=sigma_color, alpha=0.3, label=r'Baseline $\sigma=1$'
)
axes[0, 1].axhline(y=base, color='red', linestyle='--', label='Baseline')  # Add baseline
axes[0, 1].set_xlabel(r'$\Delta_{Z} \,\, [\mathrm{mm}]$', fontsize=12)
axes[0, 1].grid(True, linestyle='--', alpha=0.7)
axes[0, 1].legend()

# Plot 3: Accuracy vs Volume_XYZ
axes[1, 0].errorbar(
    volume_XYZ, accuracy_XYZ, 
    yerr=(error_XYZ_lower, error_XYZ_upper), 
    fmt='s--', capsize=5, label='Accuracy', color='green'
)
axes[1, 0].axhspan(
    base_min, base_max, color=sigma_color, alpha=0.3, label=r'Baseline $\sigma=1$'
)
axes[1, 0].axhline(y=base, color='red', linestyle='--', label='Baseline')  # Add baseline
axes[1, 0].set_xlabel(r'$\Delta_{XYZ} \,\, [\mathrm{mm}^3]$', fontsize=12)
axes[1, 0].set_ylabel('Accuracy', fontsize=12)
axes[1, 0].grid(True, linestyle='--', alpha=0.7)
axes[1, 0].legend()


# Plot 4: Matrix of Accuracy (seg_xy vs seg_z)
# Create an accuracy matrix with seg_x as x and seg_z as y
seg_xy_values = np.array([100,50,25,10])
seg_z_values = np.array([100,50,25,10])

accuracy_matrix = np.zeros((len(seg_z_values), len(seg_xy_values)))
print(f"segz: {seg_z_values}")
print(f"segx: {seg_xy_values}")
print(f"summaryTable: {summaryTable}")
summaryTable['accuracy'] = summaryTable['accuracy']*.01
for i, seg_x in enumerate(seg_xy_values):
    for j, seg_z in enumerate(seg_z_values):
        # Find the average accuracy for each combination of seg_x and seg_z
        subset = summaryTable[(summaryTable["segx"] == seg_x) & (summaryTable["segz"] == seg_z)]
        print(f"subset: {subset}")
        if (len(subset)):
            accuracy_matrix[j, i] = subset["accuracy"].mean()
        else:
            accuracy_matrix[j, i] = 0.

labels_x = ((100)/(seg_xy_values))*((100)/(seg_xy_values))*x_size*y_size
labels_y = 100/seg_z_values*z_size

heatmap_kwargs = {
    'annot': True,
    'fmt': ".2f",
    'cmap': 'Blues',
    'vmin': 0.60,
    'vmax': 0.63, 
    'annot_kws': {"size": 16},
    'cbar': True 
}

import seaborn as sns
print(f"accuracy_matrix: {accuracy_matrix}")      
sns.heatmap(
    accuracy_matrix, 
    xticklabels=labels_x, 
    yticklabels=labels_y, 
    ax=axes[1, 1],
    **heatmap_kwargs
)

axes[1, 1].set_xlabel(r'$\Delta_{XY} \,\, [\mathrm{mm}^2]$', fontsize=12)
axes[1, 1].set_ylabel(r'$\Delta_{Z} \,\, [\mathrm{mm}]$', fontsize=12)
axes[1, 1].set_title('Accuracy Matrix', fontsize=12)

plt.tight_layout()
plt.savefig(f'/home/alma1/GNN/Deepset/28oct_DNN/TowardPIDwithGranularCalorimeters/results/DNN/moneyplot.png')

# plt.show()
