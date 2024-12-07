import matplotlib.pyplot as plt
import numpy as np

area_XY = [9., 36., 144., 900.]  
accuracy_XY = [0.620, 0.620, 0.612, 0.615]  
error_XY_min = [0.616, 0.616, 0.608, 0.611]  
error_XY_max = [0.624, 0.624, 0.616, 0.619]  

delta_Z = [12., 24., 48., 120.]  
accuracy_Z = [0.620, 0.613, 0.615, 0.614]  
error_Z_min = [0.616, 0.609, 0.611, 0.610]  
error_Z_max = [0.624, 0.617, 0.619, 0.618]  

error_XY_lower = [accuracy - error_min for accuracy, error_min in zip(accuracy_XY, error_XY_min)]
error_XY_upper = [error_max - accuracy for accuracy, error_max in zip(accuracy_XY, error_XY_max)]

error_Z_lower = [accuracy - error_min for accuracy, error_min in zip(accuracy_Z, error_Z_min)]
error_Z_upper = [error_max - accuracy for accuracy, error_max in zip(accuracy_Z, error_Z_max)]


fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

axes[0].errorbar(
    area_XY, accuracy_XY, 
    yerr=(error_XY_lower, error_XY_upper), 
    fmt='s--', capsize=5, label='Accuracy'
)
axes[0].set_xlabel(r'$\Delta_{XY} \,\, [\mathrm{mm}^2]$', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)

axes[1].errorbar(
    delta_Z, accuracy_Z, 
    yerr=(error_Z_lower, error_Z_upper), 
    fmt='s--', capsize=5, label='Accuracy', color='orange'
)
axes[1].set_xlabel(r'$\Delta_{Z} \,\, [\mathrm{mm}]$', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(f'../../results/xgboost/moneyplot.png')

plt.show()
