# Standard libraries
import sys
import os
from filelock import FileLock

# Libraries for data manipulation
import numpy as np
import pandas as pd
import itertools

# Libraries for machine learning
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,  # Evaluate classification accuracy
    auc,             # Compute the area under the ROC curve
    confusion_matrix,  # Analyze classification errors
    roc_auc_score,    # Compute the AUC for binary classification
    roc_curve         # Generate data for the ROC curve
)
from sklearn.model_selection import GridSearchCV, train_test_split  # Model selection and parameter tuning
from sklearn.utils import shuffle  # Shuffle datasets for randomness

# Libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

# Libraries for statistical analysis
from scipy.stats import beta

# Set a random seed for reproducibility
seed = 42

#########################
######### INPUT #########
#########################

primary_folder = '../../dataset/'

seg_x = 100
seg_y = 100
seg_z = 100

file_path = f'{primary_folder}results_{seg_x}_{seg_y}_{seg_z}/final_combined.tsv'
data = pd.read_csv(file_path, sep="\t")

# Split features (X) and target (y)
X = data.iloc[:, :-1]
X = X[["TotalEnergy","weightedTime","time0"]]
y = data.iloc[:, -1]

# Shuffle the data
X_balanced, y_balanced = shuffle(X, y, random_state=seed)

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=seed, stratify=y_balanced
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
)

###############################
######### GRID SEARCH #########
###############################

# Parameters to optimize
param_grid = {
    'max_depth': [5, 10],                       # Tree depth
    'learning_rate': [0.01, 0.05, 0.1],         # Learning rate
    'n_estimators': [50, 100, 200, 500],        # Number of trees
    'subsample': [0.6, 1.0],                    # Row sampling
    'colsample_bytree': [0.6, 1.0],             # Column sampling
    'gamma': [0, 1, 5],                         # Penalty for split
    'min_child_weight': [1, 3, 5],              # Minimum weight required for a child node
    'reg_alpha': [0.01, 0.1, 1],                # L1 regularization (lasso)
    'reg_lambda': [1, 1.5, 2],                  # L2 regularization (ridge)
}

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(random_state=seed)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='balanced_accuracy',    # metric (balanced_accuracy or roc_auc)
    cv=3,                           # 3-fold cross-validation
    verbose=3,                      # Display details during execution
    n_jobs=-1                       # Use all available cores
)

# Run GridSearch
print("Running GridSearch on hyperparameters...")
grid_search.fit(X_train, y_train)

# Best hyperparameters found
best_params = grid_search.best_params_
print(f"Best hyperparameters found: {best_params}")

############################
######### TRAINING #########
############################

# Creating the DMatrix (required for xgb.train)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# List to monitor training metrics
evals = [(dtrain, 'train'), (dval, 'validation')]

n_estimators = best_params.pop('n_estimators', None)
best_params['objective'] = 'binary:logistic'
best_params['eval_metric'] = 'logloss'

# Dictionary to store the results for each epoch
evals_result = {}

# Training the model with tracking of metrics
best_model = xgb.train(
    params=best_params, 
    dtrain=dtrain, 
    num_boost_round=n_estimators,  # Number of trees (iterations)
    evals=evals,          # Training and validation sets
    evals_result=evals_result,  # Storing the results for each epoch
    verbose_eval=True     # Print information for each epoch
)


########################
######### TEST #########
########################

# Predictions on the test set
y_pred_proba = best_model.predict(dtest)
y_pred_binary = (y_pred_proba > 0.5).astype(int)  # Convert to binary labels


###########################
######### METRICS #########
###########################

###### ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Sigma
N_x = sum(y_test == 0)
N_y = sum(y_test == 1)

sigma_fpr = np.sqrt(fpr * (1 - fpr) / N_x)
sigma_tpr = np.sqrt(tpr * (1 - tpr) / N_y)

# Plot ROC curve
plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})', color='blue',linewidth=1)
plt.fill_between(
    fpr,
    tpr - sigma_tpr,
    tpr + sigma_tpr,
    color='blue',
    alpha=0.5,
    label='1-sigma region (TPR)'
)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guess')
plt.xlabel('Proton Positive Rate')
plt.ylabel('Pion Positive Rate')
plt.title('ROC Curve with 1-Sigma Uncertainty Region')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.savefig(f'../../results/xgboost/baseline/roc_curve_with_uncertainty.png')

plt.show()

###### FEATURE IMPORTANCE
plt.figure(figsize=(10, 8))

xgb.plot_importance(best_model, importance_type='gain', max_num_features=20, title="Feature Importance", height=0.8)
plt.tight_layout()
plt.savefig(f'../../results/xgboost/baseline/feature_importance_best_model.png')

plt.close()


##### CONFIDENCE DISTRIBUTION

def create_pairs(arr):
    pairs = []
    for el in arr:
        if el > 0.5:
            pairs.append([1 - el, el])
        else:
            pairs.append([el, 1 - el])
    return pairs

def split_by_position(pairs, y_test):
    
    array_1 = [] 
    array_2 = [] 
    
    for i, pair in enumerate(pairs):
        
        max_idx = np.argmax(pair)
        max_el = max(pair)
        
        if y_test[i] == max_idx:
            array_1.append(max_el)
        else:            
            array_2.append(max_el)
    
    return np.array(array_1), np.array(array_2)


y_pred_proba_pairs = create_pairs(y_pred_proba)
true_class_probs, false_class_probs = split_by_position(y_pred_proba_pairs, y_test.values)

# Plotting the distributions
plt.figure(figsize=(8, 6))

plt.hist(true_class_probs, bins=50, alpha=0.5, color='blue', label='Right Prediction')
plt.hist(false_class_probs, bins=50, alpha=0.5, color='red', label='Wrong Prediction')
plt.xlabel('Predicted Probability')
plt.ylabel('Counts')
plt.title('Distribution of Predicted Probabilities for True vs False Class')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'../../results/xgboost/baseline/predicted_probability_distribution.png')

plt.close()

##### TRAINING AND VALIDATION LOSSES

# Plotting the loss for each epoch (training and validation)
epochs = len(evals_result['train']['logloss'])

plt.figure(figsize=(12, 6))

plt.plot(range(epochs), evals_result['train']['logloss'], label='Training Loss')
plt.plot(range(epochs), evals_result['validation']['logloss'], label='Validation Loss')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'../../results/xgboost/baseline/training_validation_loss.png')

plt.close()

##### CONFUSION MATRIX (WITH AND WITHOUT NC)

def createNC_matrix(y_test,y_pred_binary, y_pred_prob, thr):
    
    mask = (y_pred_prob > thr) | (y_pred_prob < (1 - thr))
    pred_with_nd = np.where(mask, y_pred_binary, 'NC')
    
    cm_withNC = np.zeros((2, 3), dtype=int)  # 2x3 matrix: 2 true labels (0, 1) and 3 predicted labels (0, 1, NC)

    for true, pred in zip(y_test, pred_with_nd):
        if true == 0:
            if pred == "0":
                cm_withNC[0, 0] += 1  # True 0, Pred 0
            elif pred == "1":
                cm_withNC[0, 1] += 1  # True 0, Pred 1
            elif pred == 'NC':
                cm_withNC[0, 2] += 1  # True 0, Pred ND
        elif true == 1:
            if pred == "0":
                cm_withNC[1, 0] += 1  # True 1, Pred 0
            elif pred == "1":
                cm_withNC[1, 1] += 1  # True 1, Pred 1
            elif pred == 'NC':
                cm_withNC[1, 2] += 1  # True 1, Pred ND
               
    return cm_withNC

def optimal_matrix(y_test, y_pred_prob,y_pred_binary):
    
    acc_array = []
    eff_array = []
    
    for i in range(100):
        
        thr_test = 0.01 + 0.01*i
        cm = createNC_matrix(y_test, y_pred_binary, y_pred_prob,thr_test)
        
        try:
            denominator = cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0]
            if denominator == 0:
                acc = 0
            else:
                acc = (cm[0, 0] + cm[1, 1]) / denominator
        except (ZeroDivisionError, RuntimeWarning):
            acc = 0

        try:
            eff = (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1]) / np.sum(cm)
        except (ZeroDivisionError, RuntimeWarning):
            eff = 0
        
        acc_array.append(acc)
        eff_array.append(eff)
    
    acc_array = np.array(acc_array)
    eff_array = np.array(eff_array)
    thr = 0.01 + np.arange(100) * 0.01
    
    acc_interp = interp1d(thr, acc_array, kind='linear', fill_value="extrapolate")
    eff_interp = interp1d(thr, eff_array, kind='linear', fill_value="extrapolate")
    thresholds = np.linspace(thr.min(), thr.max(), 1000)
    differences = acc_interp(thresholds) - eff_interp(thresholds)
    intersection_idx = np.where(np.diff(np.sign(differences)))[0][0]
    intersection_thr = thresholds[intersection_idx]
    intersection_value = acc_interp(intersection_thr)
    
    cm_acc = createNC_matrix(y_test, y_pred_binary, y_pred_prob,intersection_thr)
    
    return cm_acc, intersection_thr, intersection_value, acc_array, eff_array


cm_standard = confusion_matrix(y_test, y_pred_binary, labels=[0, 1])
acc_standard = accuracy_score(y_pred_binary,y_test)

# CLOPPER-PEARSON INTERVAL

#acc = (n_tn + n_tp)/n = n_t / n
    # n_tn = cm[0,0]
    # m_tp = cm[1,1]
    # n = np.sum(cm)

alpha = 0.32
n_t = cm_standard[0,0] + cm_standard[1,1]
n = np.sum(cm_standard)

lower_bound = beta.ppf(alpha / 2, n_t, n - n_t + 1)
upper_bound = beta.ppf(1 - alpha / 2, n_t + 1, n - n_t)


cm_acc, thr_acc, acc_withNC, acc_array, eff_array = optimal_matrix(y_test, y_pred_proba,y_pred_binary)

pred_labels_standard = ['0', '1'] 
pred_labels_with_nc = ['0', '1', 'NC'] 
true_labels = ['0', '1']

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

heatmap_kwargs = {
    'annot': True,
    'fmt': "d",
    'cmap': 'Blues',
    'vmin': 0,
    'vmax': 7500, 
    'annot_kws': {"size": 16},
    'cbar': False 
}
sns.heatmap(
    cm_standard, 
    xticklabels=pred_labels_standard, 
    yticklabels=true_labels, 
    ax=axes[0],
    **heatmap_kwargs
)
axes[0].set_title(f"Standard Confusion Matrix\nAccuracy: {acc_standard:.3f}, 68% CI [{lower_bound:.3f}, {upper_bound:.3f}]", fontsize=18)
axes[0].set_xlabel("Predicted", fontsize=14)
axes[0].set_ylabel("True", fontsize=14)
sns.heatmap(
    cm_acc, 
    xticklabels=pred_labels_with_nc, 
    yticklabels=true_labels, 
    ax=axes[1],
    **heatmap_kwargs
)
axes[1].set_title(f"Confusion Matrix with NC\nThreshold: {thr_acc:.3f}, Accuracy: {acc_withNC:.3f}", fontsize=18)
axes[1].set_xlabel("Predicted", fontsize=14)
axes[1].set_ylabel("True", fontsize=14)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(axes[0].collections[0], cax=cbar_ax)
for ax in axes:
    ax.tick_params(labelsize=14) 
    ax.title.set_fontsize(18)   
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)

plt.savefig(f'../../results/xgboost/baseline/confusion_matrices.png')

plt.show()


##### ACCURACY AND EFFICIENCY

thr = 0.01 + np.arange(100) * 0.01

acc_interp = interp1d(thr, acc_array, kind='linear', fill_value="extrapolate")
eff_interp = interp1d(thr, eff_array, kind='linear', fill_value="extrapolate")
thresholds = np.linspace(thr.min(), thr.max(), 1000)
differences = acc_interp(thresholds) - eff_interp(thresholds)
intersection_idx = np.where(np.diff(np.sign(differences)))[0][0]
intersection_thr = thresholds[intersection_idx]
intersection_value = acc_interp(intersection_thr)

plt.figure(figsize=(8, 6), dpi=100)

plt.plot(thr, acc_array, label="Accuracy", color="blue", linestyle="-")
plt.plot(thr, eff_array, label="Efficiency", color="green", linestyle="-")
plt.axvline(intersection_thr, color="red", linestyle=":", label=f"Intersection at {intersection_thr:.2f}")
plt.annotate(
    f"({intersection_thr:.2f}, {intersection_value:.2f})",
    xy=(intersection_thr, intersection_value),
    xytext=(intersection_thr + 0.02, intersection_value + 0.1),
    arrowprops=dict(facecolor='red', shrink=0.05),
    fontsize=10,
    color="red"
)
plt.title("Accuracy and Efficiency vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'../../results/xgboost/baseline/accuracy_efficiency.png')

plt.show()

target_file = "../../results/xgboost/baseline.tsv"
data = {
        "segmentation": ['baseline'],
        "accuracy": [-1],
        "minVal": [-1],
        "maxVal": [-1],
        "threshold": [-1],
        "tradeOffValue": [-1],
    }

summaryTable = pd.DataFrame(data)
os.makedirs(os.path.dirname(target_file), exist_ok=True)

segmentation = 'baseline'
summaryTable["accuracy"] = summaryTable["accuracy"].astype(float)
summaryTable["minVal"] = summaryTable["minVal"].astype(float)
summaryTable["maxVal"] = summaryTable["maxVal"].astype(float)
summaryTable["threshold"] = summaryTable["threshold"].astype(float)
summaryTable["tradeOffValue"] = summaryTable["tradeOffValue"].astype(float)
        
summaryTable.loc[summaryTable["segmentation"] == segmentation, "accuracy"] = acc_standard
summaryTable.loc[summaryTable["segmentation"] == segmentation, "minVal"] = lower_bound
summaryTable.loc[summaryTable["segmentation"] == segmentation, "maxVal"] = upper_bound

summaryTable.loc[summaryTable["segmentation"] == segmentation, "threshold"] = intersection_thr
summaryTable.loc[summaryTable["segmentation"] == segmentation, "tradeOffValue"] = intersection_value

summaryTable.to_csv(target_file, sep="\t", index=False)