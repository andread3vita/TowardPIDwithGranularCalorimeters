import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.patches as mpatches
import sys

z_size = 12.
x_size = 3.
y_size = 3.

analysis_type = sys.argv[1] # ppi = proton, pion , pik = pion, kaon , pk = proton, kaon , ppik = proton, pion, kaon

folder_path = "../../results/xgboost/"
if analysis_type == "ppi":
    folder_path += "proton_pion"
elif analysis_type == "pik":
    folder_path += "pion_kaon"
elif analysis_type == "pk":
    folder_path += "proton_kaon"
elif analysis_type == "ppik":
    folder_path += "proton_pion_kaon"

target_file = f"{folder_path}/accuracyTable.tsv"
summaryTable = pd.read_csv(target_file, sep="\t")
summaryTable = summaryTable[summaryTable["accuracy"] != -1]


summaryTable[["segx", "segy", "segz"]] = summaryTable["segmentation"].str.extract(r"(\d+)_(\d+)_(\d+)")
summaryTable["segx"] = summaryTable["segx"].astype(float)
summaryTable["segy"] = summaryTable["segy"].astype(float)
summaryTable["segz"] = summaryTable["segz"].astype(float)

segZ_accuracy = (
    summaryTable.sort_values(by=["segz", "segx"])  
    .groupby("segz")["accuracy"]         
    .apply(list)                         
    .to_dict()                           
)

segZ_minAcc = (
    summaryTable.sort_values(by=["segz", "segx"])  
    .groupby("segz")["minVal"]         
    .apply(list)                         
    .to_dict()                           
)

segZ_maxAcc = (
    summaryTable.sort_values(by=["segz", "segx"])  
    .groupby("segz")["maxVal"]         
    .apply(list)                         
    .to_dict()                           
)

segXY_accuracy = (
    summaryTable.sort_values(by=["segx", "segz"])  
    .groupby("segx")["accuracy"]         
    .apply(list)                         
    .to_dict()                          
)

segXY_minAcc = (
    summaryTable.sort_values(by=["segx", "segz"])  
    .groupby("segx")["minVal"]         
    .apply(list)                         
    .to_dict()                           
)

segXY_maxAcc = (
    summaryTable.sort_values(by=["segx", "segz"])  
    .groupby("segx")["maxVal"]         
    .apply(list)                         
    .to_dict()                           
)

volume_XYZ = ((100/summaryTable["segx"]) * (100/summaryTable["segy"])*(100/summaryTable["segz"])*x_size*y_size*z_size).tolist()
accuracy_XYZ = summaryTable["accuracy"].tolist()
error_XYZ_min = summaryTable["minVal"].tolist()
error_XYZ_max = summaryTable["maxVal"].tolist()

error_XYZ_lower = [accuracy - error_min for accuracy, error_min in zip(accuracy_XYZ, error_XYZ_min)]
error_XYZ_upper = [error_max - accuracy for accuracy, error_max in zip(accuracy_XYZ, error_XYZ_max)]

volume_XYZ, accuracy_XYZ, error_XYZ_lower, error_XYZ_upper = zip(*sorted(zip(volume_XYZ, accuracy_XYZ, error_XYZ_lower, error_XYZ_upper), key=lambda x: x[0]))


baseline_file = f"{folder_path}/baseline.tsv"
baselineTable = pd.read_csv(baseline_file, sep="\t")

base = baselineTable['accuracy'].iloc[0].astype(float)
base_min = baselineTable['minVal'].iloc[0].astype(float)
base_max = baselineTable['maxVal'].iloc[0].astype(float)

# Define the color for the 1-sigma region
sigma_color = 'red'

# Create the figure with 2 rows and 2 columns
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey=False)

# Plot 1: Accuracy vs Area_XY
for segz in segZ_accuracy.keys():
    # Get the accuracy, minVal, and maxVal for the current segx
    accuracy = segZ_accuracy[segz]
    min_val = segZ_minAcc[segz]
    max_val = segZ_maxAcc[segz]
    
    # Calculate the error (difference between accuracy and min/max)
    error = [accuracy - min_v for accuracy, min_v in zip(accuracy, min_val)]
    error_upper = [max_v - accuracy for max_v, accuracy in zip(max_val, accuracy)]

    areaXY = [100 / segx * 100 / segx * x_size * y_size for segx in [10.0, 25.0, 50.0, 100.0]]
    
    # Plot the curve with error bars
    axes[0, 0].errorbar(
        areaXY,                     
        accuracy, 
        yerr=[error, error_upper],  # Lower and upper error
        label=f'segZ = {segz}',     # Legend for each segx
        fmt='--o',                  # Line style and marker
        capsize=5                   # Add caps to the error bars
    )
    
axes[0, 0].axhspan(
    base_min, base_max, color=sigma_color, alpha=0.3, label=r'Baseline $\sigma=1$'
)


if analysis_type == "ppi":
    axes[0, 0].set_title(r'Accuracy $p/\pi$ as a Function of Cell Cross-Section Area') 
elif analysis_type == "pik":
    axes[0, 0].set_title(r'Accuracy $\pi/\K$ as a Function of Cell Cross-Section Area') 
elif analysis_type == "pk":
    axes[0, 0].set_title(r'Accuracy $p/\K$ as a Function of Cell Cross-Section Area') 

axes[0, 0].axhline(y=base, color='red', linestyle='--', label='Baseline')  # Add baseline
axes[0, 0].set_xlabel(r'$\Delta_{XY} \,\, [\mathrm{mm}^2]$', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].legend()  # Show the legend
axes[0, 0].grid(True, linestyle='--', alpha=0.7)
axes[0, 0].set_xscale('log')

# Plot 2: Accuracy vs Delta_Z
for segx in segXY_accuracy.keys():
    # Get the accuracy, minVal, and maxVal for the current segx
    accuracy = segXY_accuracy[segx]
    min_val = segXY_minAcc[segx]
    max_val = segXY_maxAcc[segx]
    
    # Calculate the error (difference between accuracy and min/max)
    error = [accuracy - min_v for accuracy, min_v in zip(accuracy, min_val)]
    error_upper = [max_v - accuracy for max_v, accuracy in zip(max_val, accuracy)]

    # Calculate deltaZ (deltaZ = 100 / segz * z_size)
    deltaZ = [100 / segz * z_size for segz in [10.0, 25.0, 50.0, 100.0]] 

    # Plot the curve with error bars
    axes[0, 1].errorbar(
        deltaZ,  # Use deltaZ for the x-axis
        accuracy, 
        yerr=[error, error_upper],  # Lower and upper error
        label=f'segXY = {segx}',     # Legend for each segx
        fmt='--o',                  # Line style and marker
        capsize=5                   # Add caps to the error bars
    )

axes[0, 1].axhspan(
    base_min, base_max, color=sigma_color, alpha=0.3, label=r'Baseline $\sigma=1$'
)

if analysis_type == "ppi":
    axes[0, 1].set_title(r'Accuracy $p/\pi$ as a Function of Longitudinal Segmentation') 
elif analysis_type == "pik":
    axes[0, 1].set_title(r'Accuracy $\pi/\K$ as a Function of Longitudinal Segmentation') 
elif analysis_type == "pk":
    axes[0, 1].set_title(r'Accuracy $p/\K$ as a Function of Longitudinal Segmentation') 
    
axes[0, 1].axhline(y=base, color='red', linestyle='--', label='Baseline') 
axes[0, 1].set_xlabel(r'$\Delta_{Z} \,\, [\mathrm{mm}]$', fontsize=12)
axes[0, 1].grid(True, linestyle='--', alpha=0.7)
axes[0, 1].legend()  # Show the legend
axes[0, 1].set_xscale('log')

# Plot 3: Accuracy vs Volume_XYZ
axes[1, 0].errorbar(
    volume_XYZ, accuracy_XYZ, 
    yerr=(error_XYZ_lower, error_XYZ_upper), 
    fmt='s--', capsize=5, label='Accuracy', color='green'
)
axes[1, 0].axhspan(
    base_min, base_max, color=sigma_color, alpha=0.3, label=r'Baseline $\sigma=1$'
)

if analysis_type == "ppi":
    axes[1, 0].set_title(r'Accuracy $p/\pi$ as a Function of Cell Volume') 
elif analysis_type == "pik":
    axes[1, 0].set_title(r'Accuracy $\pi/\K$ as a Function of Cell Volume') 
elif analysis_type == "pk":
    axes[1, 0].set_title(r'Accuracy $p/\K$ as a Function of Cell Volume') 
    
axes[1, 0].axhline(y=base, color='red', linestyle='--', label='Baseline') 
axes[1, 0].set_xlabel(r'$\Delta_{XYZ} \,\, [\mathrm{mm}^3]$', fontsize=12)
axes[1, 0].set_ylabel('Accuracy', fontsize=12)
axes[1, 0].grid(True, linestyle='--', alpha=0.7)
axes[1, 0].legend()
axes[1, 0].set_xscale('log')


# Plot 4: Matrix of Accuracy (seg_xy vs seg_z)
# Create an accuracy matrix with seg_x as x and seg_z as y
seg_xy_values = np.array([100,50,25,10])
seg_z_values = np.array([100,50,25,10])

accuracy_matrix = np.zeros((len(seg_z_values), len(seg_xy_values)))

for i, seg_x in enumerate(seg_xy_values):
    for j, seg_z in enumerate(seg_z_values):
        # Find the average accuracy for each combination of seg_x and seg_z
        subset = summaryTable[(summaryTable["segx"] == seg_x) & (summaryTable["segz"] == seg_z)]
        
        if (len(subset)):
            accuracy_matrix[j, i] = subset["accuracy"].mean()
        else:
            accuracy_matrix[j, i] = 0.

labels_x = ((100)/(seg_xy_values))*((100)/(seg_xy_values))*x_size*y_size
labels_y = 100/seg_z_values*z_size

heatmap_kwargs = {
    'annot': True,
    'fmt': ".3f",
    'cmap': 'Blues',
    'vmin': 0.60,
    'vmax': 0.63, 
    'annot_kws': {"size": 16},
    'cbar': True 
}

import seaborn as sns

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

if analysis_type == "ppi":
    axes[1, 1].set_title(r'Accuracy $p/\pi$ as a Function of Segmentation') 
elif analysis_type == "pik":
    axes[1, 1].set_title(r'Accuracy $\pi/\K$ as a Function of Segmentation') 
elif analysis_type == "pk":
    axes[1, 1].set_title(r'Accuracy $p/\K$ as a Function of Segmentation') 

plt.tight_layout()
plt.savefig(f'{folder_path}/moneyplot.png')














# area_XY = []
# accuracy_XY = []
# error_XY_lower = []
# error_XY_upper = []
# if (len(filtered_segx)):
#     area_XY = ((100/filtered_segx["segx"]) * (100/filtered_segx["segy"])*x_size*y_size).tolist()
#     accuracy_XY = filtered_segx["accuracy"].tolist()
#     error_XY_min = filtered_segx["minVal"].tolist()
#     error_XY_max = filtered_segx["maxVal"].tolist()
    
#     error_XY_lower = [accuracy - error_min for accuracy, error_min in zip(accuracy_XY, error_XY_min)]
#     error_XY_upper = [error_max - accuracy for accuracy, error_max in zip(accuracy_XY, error_XY_max)]
    
#     area_XY, accuracy_XY, error_XY_lower, error_XY_upper = zip(*sorted(zip(area_XY, accuracy_XY, error_XY_lower, error_XY_upper), key=lambda x: x[0]))

# delta_Z = []
# accuracy_Z = []
# error_Z_lower = []
# error_Z_upper = []
# if (len(filtered_segz)):
    
#     delta_Z = ((100/filtered_segz["segz"])*z_size).tolist()
#     accuracy_Z = filtered_segz["accuracy"].tolist()
#     error_Z_min = filtered_segz["minVal"].tolist()
#     error_Z_max = filtered_segz["maxVal"].tolist()

#     error_Z_lower = [accuracy - error_min for accuracy, error_min in zip(accuracy_Z, error_Z_min)]
#     error_Z_upper = [error_max - accuracy for accuracy, error_max in zip(accuracy_Z, error_Z_max)]

#     delta_Z, accuracy_Z, error_Z_lower, error_Z_upper = zip(*sorted(zip(delta_Z, accuracy_Z, error_Z_lower, error_Z_upper), key=lambda x: x[0]))


# volume_XYZ = ((100/summaryTable["segx"]) * (100/summaryTable["segy"])*(100/summaryTable["segz"])*x_size*y_size*z_size).tolist()
# accuracy_XYZ = summaryTable["accuracy"].tolist()
# error_XYZ_min = summaryTable["minVal"].tolist()
# error_XYZ_max = summaryTable["maxVal"].tolist()

# error_XYZ_lower = [accuracy - error_min for accuracy, error_min in zip(accuracy_XYZ, error_XYZ_min)]
# error_XYZ_upper = [error_max - accuracy for accuracy, error_max in zip(accuracy_XYZ, error_XYZ_max)]

# volume_XYZ, accuracy_XYZ, error_XYZ_lower, error_XYZ_upper = zip(*sorted(zip(volume_XYZ, accuracy_XYZ, error_XYZ_lower, error_XYZ_upper), key=lambda x: x[0]))

# baseline_file = f"{folder_path}/baseline.tsv"
# baselineTable = pd.read_csv(baseline_file, sep="\t")

# base = baselineTable['accuracy'].iloc[0].astype(float)
# base_min = baselineTable['minVal'].iloc[0].astype(float)
# base_max = baselineTable['maxVal'].iloc[0].astype(float)

# # Create the figure with 2 rows and 2 columns
# fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey=False)

# # Define the color for the 1-sigma region
# sigma_color = 'red'

# # Plot 1: Accuracy vs Area_XY
# for x, y, yerr_lower, yerr_upper, color in zip(area_XY, accuracy_XY, error_XY_lower, error_XY_upper, filtered_segx['color']):
#     axes[0, 0].errorbar(
#         x, y,
#         yerr=[[yerr_lower], [yerr_upper]],
#         fmt='o--', color=color, capsize=5, alpha=0.8
#     )
# axes[0, 0].axhspan(
#     base_min, base_max, color=sigma_color, alpha=0.3, label=r'Baseline $\sigma=1$'
# )
# axes[0, 0].axhline(y=base, color='red', linestyle='--', label='Baseline')  # Add baseline
# axes[0, 0].set_xlabel(r'$\Delta_{XY} \,\, [\mathrm{mm}^2]$', fontsize=12)
# axes[0, 0].set_ylabel('Accuracy', fontsize=12)
# axes[0, 0].grid(True, linestyle='--', alpha=0.7)

# # Plot 2: Accuracy vs Delta_Z
# for x, y, yerr_lower, yerr_upper, color in zip(delta_Z, accuracy_Z, error_Z_lower, error_Z_upper, filtered_segz['color']):
#     axes[0, 1].errorbar(
#         x, y,
#         yerr=[[yerr_lower], [yerr_upper]],
#         fmt='o--', color=color, capsize=5, alpha=0.8
#     )
# axes[0, 1].axhspan(
#     base_min, base_max, color=sigma_color, alpha=0.3, label=r'Baseline $\sigma=1$'
# )
# axes[0, 1].axhline(y=base, color='red', linestyle='--', label='Baseline')  # Add baseline
# axes[0, 1].set_xlabel(r'$\Delta_{Z} \,\, [\mathrm{mm}]$', fontsize=12)
# axes[0, 1].grid(True, linestyle='--', alpha=0.7)

# handles_xy = [
#     mpatches.Patch(color=color, label=label)
#     for label, color in color_map.items() if "Z" in label
# ]
# axes[0, 0].legend(handles=handles_xy, title='Z segmentation', loc='best')

# handles_z = [
#     mpatches.Patch(color=color, label=label)
#     for label, color in color_map.items() if "XY" in label
# ]
# axes[0, 1].legend(handles=handles_z, title='XY Segmentation', loc='best')

# # Plot 3: Accuracy vs Volume_XYZ
# axes[1, 0].errorbar(
#     volume_XYZ, accuracy_XYZ, 
#     yerr=(error_XYZ_lower, error_XYZ_upper), 
#     fmt='s--', capsize=5, label='Accuracy', color='green'
# )
# axes[1, 0].axhspan(
#     base_min, base_max, color=sigma_color, alpha=0.3, label=r'Baseline $\sigma=1$'
# )
# axes[1, 0].axhline(y=base, color='red', linestyle='--', label='Baseline')  # Add baseline
# axes[1, 0].set_xlabel(r'$\Delta_{XYZ} \,\, [\mathrm{mm}^3]$', fontsize=12)
# axes[1, 0].set_ylabel('Accuracy', fontsize=12)
# axes[1, 0].grid(True, linestyle='--', alpha=0.7)
# axes[1, 0].legend()


# # Plot 4: Matrix of Accuracy (seg_xy vs seg_z)
# # Create an accuracy matrix with seg_x as x and seg_z as y
# seg_xy_values = np.array([100,50,25,10])
# seg_z_values = np.array([100,50,25,10])

# accuracy_matrix = np.zeros((len(seg_z_values), len(seg_xy_values)))

# for i, seg_x in enumerate(seg_xy_values):
#     for j, seg_z in enumerate(seg_z_values):
#         # Find the average accuracy for each combination of seg_x and seg_z
#         subset = summaryTable[(summaryTable["segx"] == seg_x) & (summaryTable["segz"] == seg_z)]
        
#         if (len(subset)):
#             accuracy_matrix[j, i] = subset["accuracy"].mean()
#         else:
#             accuracy_matrix[j, i] = 0.

# labels_x = ((100)/(seg_xy_values))*((100)/(seg_xy_values))*x_size*y_size
# labels_y = 100/seg_z_values*z_size

# heatmap_kwargs = {
#     'annot': True,
#     'fmt': ".3f",
#     'cmap': 'Blues',
#     'vmin': 0.61,
#     'vmax': 0.63, 
#     'annot_kws': {"size": 16},
#     'cbar': True 
# }

# import seaborn as sns

# sns.heatmap(
#     accuracy_matrix, 
#     xticklabels=labels_x, 
#     yticklabels=labels_y, 
#     ax=axes[1, 1],
#     **heatmap_kwargs
# )

# axes[1, 1].set_xlabel(r'$\Delta_{XY} \,\, [\mathrm{mm}^2]$', fontsize=12)
# axes[1, 1].set_ylabel(r'$\Delta_{Z} \,\, [\mathrm{mm}]$', fontsize=12)
# axes[1, 1].set_title('Accuracy Matrix', fontsize=12)

# plt.tight_layout()
# plt.savefig(f'{folder_path}/moneyplot.png')

# plt.show()

