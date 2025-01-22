# Standard libraries
import sys
import os

# Libraries for data manipulation
import numpy as np
import pandas as pd

# Libraries for machine learning
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,  # Evaluate classification accuracy
    auc,             # Compute the area under the ROC curve
    confusion_matrix,  # Analyze classification errors
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

seg_x = sys.argv[1]
seg_y = sys.argv[2]
seg_z = sys.argv[3]

time_type = sys.argv[4]

analysis_type = sys.argv[5] # ppi = proton, pion , pik = pion, kaon , pk = proton, kaon , ppik = proton, pion, kaon

results_folder = "../../results/xgboost/"
if analysis_type == "ppi":
    results_folder += "proton_pion"
elif analysis_type == "pik":
    results_folder += "pion_kaon"
elif analysis_type == "pk":
    results_folder += "proton_kaon"
elif analysis_type == "ppik":
    results_folder += "proton_pion_kaon"
    
os.makedirs(f"{results_folder}", exist_ok=True) 

# create Summary Table# Create Summary Table
target_file = f"{results_folder}/accuracyTable.tsv"
segmentation = f"{seg_x}_{seg_y}_{seg_z}"

if not os.path.exists(target_file):

    headers = ["segmentation", "accuracy", "minVal", "maxVal"]
    with open(target_file, "w") as file:
        file.write("\t".join(headers) + "\n")
else:
    with open(target_file, "r") as file:
        lines = file.readlines()

    updated_lines = [line for line in lines if line.strip() != segmentation]

    if len(updated_lines) != len(lines):
        print("The selected segmentation is already stored in the accuracyTable. Please remove that line to continue.")
        sys.exit(1)

# import data
primary_folder = '../../dataset/'

folder_path = f"{results_folder}/{seg_x}_{seg_y}_{seg_z}"
os.makedirs(f"{folder_path}", exist_ok=True) 

file_path = ""
if time_type == "d":
    file_path = f'{primary_folder}results_{seg_x}_{seg_y}_{seg_z}/final_combined_digi.tsv'
else:
    file_path = f'{primary_folder}results_{seg_x}_{seg_y}_{seg_z}/final_combined_smear.tsv'
    
data = pd.read_csv(file_path, sep="\t")

if time_type == "d":
    data = data.drop(columns=["time0"])

if analysis_type == "ppi":
    data = data[data['Class'].isin([0, 1])]
    
elif analysis_type == "pik":
    data = data[data['Class'].isin([1, 2])]
    data.loc[data['Class'] == 1, 'Class'] = 0
    data.loc[data['Class'] == 2, 'Class'] = 1
    
elif analysis_type == "pk":
    data = data[data['Class'].isin([0, 2])]
    data.loc[data['Class'] == 2, 'Class'] = 1
  
# Split features (X) and target (y)
X = data.iloc[:, :-1]
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

# # Parameters to optimize
# param_grid = {
#     'max_depth': [5, 10],                       # Tree depth
#     'learning_rate': [0.001, 0.01],             # Learning rate
#     'n_estimators': [1000],                     # Number of trees
#     'subsample': [0.6, 1.0],                    # Row sampling
#     'colsample_bytree': [0.6, 1.0],             # Column sampling
#     'gamma': [0, 1, 5],                         # Penalty for split
#     'min_child_weight': [1, 3, 5],              # Minimum weight required for a child node
#     'reg_alpha': [0.1],                         # L1 regularization (lasso)
#     'reg_lambda': [1, 1.5],                     # L2 regularization (ridge)
# }

# # Initialize XGBoost model
# xgb_model = xgb.XGBClassifier(random_state=seed)

# # GridSearchCV
# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=param_grid,
#     scoring='balanced_accuracy',    # metric (balanced_accuracy or roc_auc)
#     cv=3,                           # 3-fold cross-validation
#     verbose=3,                      # Display details during execution
#     n_jobs=5                       # Use 20 available cores
# )

# # Run GridSearch
# print("Running GridSearch on hyperparameters...")
# grid_search.fit(X_train, y_train)


# # Best hyperparameters found
# best_params = grid_search.best_params_
# print(f"Best hyperparameters found: {best_params}")


best_params = {}
if analysis_type == "ppi":
    best_params = {
        'colsample_bytree': 0.6,                        
        'learning_rate': 0.001,         
        'max_depth': 10,               
        'min_child_weight': 1,         
        'n_estimators': 4500,           
        'reg_alpha': 0.1,              
        'reg_lambda': 2,             
        'subsample': 1.0               
    }
    
elif analysis_type == "pik":
    
    best_params = {
        'colsample_bytree': 0.6,                        
        'learning_rate': 0.001,         
        'max_depth': 7,               
        'min_child_weight': 1,         
        'n_estimators': 3500,           
        'reg_alpha': 0.1,              
        'reg_lambda': 2,             
        'subsample': 1.0               
    }
    
elif analysis_type == "pk":
    
    best_params = {
        'colsample_bytree': 0.6,                        
        'learning_rate': 0.001,         
        'max_depth': 10,               
        'min_child_weight': 1,         
        'n_estimators': 4000,           
        'reg_alpha': 0.01,              
        'reg_lambda': 2,             
        'subsample': 1.0               
    }
    
elif analysis_type == "ppik":
    print("Not yet implemented")
    sys.exit(1)
    


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

if not analysis_type == "ppik":
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'logloss'
else:
    best_params['objective'] = 'multi:softprob'
    best_params['eval_metric'] = 'mlogloss'

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


if not analysis_type == "ppik":
    
    ########################
    ######### TEST #########
    ########################


    # Predictions on the test set
    y_pred_proba = best_model.predict(dtest)
    y_pred_binary = (y_pred_proba > 0.5).astype(int)  # Convert to binary labels


    ###########################
    ######### METRICS #########
    ###########################

    # legend:
    #   if ppi -> p = 0 and pi = 1
    #   if pk  -> p = 0 and k = 1
    #   if pik -> pi = 0 and k = 1
    
    
    ###### ROC

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Sigma
    N_x = sum(y_test == 0)
    N_y = sum(y_test == 1)

    sigma_fpr = np.sqrt(fpr * (1 - fpr) / N_x)
    sigma_tpr = np.sqrt(tpr * (1 - tpr) / N_y)

    x_label = ""
    y_label = ""
    
    if analysis_type == "ppi":
        
        x_label = "proton"
        y_label = "pion"
    
    elif analysis_type == "pik":
        
        x_label = "pion"
        y_label = "kaon"
        
    elif analysis_type == "pk":
             
        x_label = "proton"
        y_label = "kaon"
    
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
    plt.xlabel(f'{x_label} Positive Rate')
    plt.ylabel(f'{y_label} Positive Rate')
    plt.title('ROC Curve with 1-Sigma Uncertainty Region')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{folder_path}/roc_curve_with_uncertainty.png')

    plt.show()

    ###### FEATURE IMPORTANCE

    plt.figure(figsize=(10, 8))

    xgb.plot_importance(best_model, importance_type='gain', max_num_features=20, title="Feature Importance", height=0.8)
    plt.tight_layout()
    plt.savefig(f'{folder_path}/feature_importance_best_model.png')

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
    plt.savefig(f'{folder_path}/predicted_probability_distribution.png')

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
    plt.savefig(f'{folder_path}/training_validation_loss.png')

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

        acc_array_1 = []
        eff_array_1 = []

        acc_array_0 = []
        eff_array_0 = []
        
        for i in range(100):
            
            thr_test = 0.01 + 0.01*i
            cm = createNC_matrix(y_test, y_pred_binary, y_pred_prob,thr_test)
            
            denominator = cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0]
            denominator_0 = cm[0, 0] + cm[0, 1]
            denominator_1 = cm[1, 0] + cm[1, 1]

            if denominator == 0:
                acc = 0
            else:
                acc = (cm[0, 0] + cm[1, 1]) / denominator

            if denominator_0 == 0:
                acc_0 = 0
                
            else:
                acc_0 = (cm[0, 0]) / denominator_0

            if denominator_1 == 0:
                acc_1 = 0
            else:
                acc_1 = (cm[1, 1]) / denominator_1
            
            
            eff = (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1]) / np.sum(cm)
            eff_0 = (cm[0, 0] + cm[0, 1]) / (cm[0, 0] + cm[0, 1]+ cm[0, 2])
            eff_1 = (cm[1, 0] + cm[1, 1]) / (cm[1, 0] + cm[1, 1]+ cm[1, 2])

            acc_array.append(acc)
            eff_array.append(eff)

            acc_array_1.append(acc_1)
            eff_array_1.append(eff_1)

            acc_array_0.append(acc_0)
            eff_array_0.append(eff_0)


        thr = 0.01 + np.arange(100) * 0.01

        #total info
        acc_array = np.array(acc_array)
        eff_array = np.array(eff_array)
        
        acc_interp = interp1d(thr, acc_array, kind='linear', fill_value="extrapolate")
        eff_interp = interp1d(thr, eff_array, kind='linear', fill_value="extrapolate")
        thresholds = np.linspace(thr.min(), thr.max(), 1000)
        differences = acc_interp(thresholds) - eff_interp(thresholds)
        intersection_idx = np.where(np.diff(np.sign(differences)))[0][0]
        intersection_thr = thresholds[intersection_idx]
        intersection_value = acc_interp(intersection_thr)
        
        cm_acc = createNC_matrix(y_test, y_pred_binary, y_pred_prob,intersection_thr)

        # 0 info
        acc_array_0 = np.array(acc_array_0)
        eff_array_0 = np.array(eff_array_0)

        # 1 info
        acc_array_1 = np.array(acc_array_1)
        eff_array_1 = np.array(eff_array_1)
        
        
        return cm_acc, \
                intersection_thr, intersection_value, acc_array, eff_array, \
                acc_array_1, eff_array_1, \
                acc_array_0, eff_array_0
                
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


    cm_acc, \
        thr_acc, acc_withNC, acc_array, eff_array,\
        acc_array_1, eff_array_1, \
        acc_array_0, eff_array_0 = optimal_matrix(y_test, y_pred_proba,y_pred_binary)

    pred_labels_standard = [f'{x_label}', f'{y_label}'] 
    pred_labels_with_nc = [f'{x_label}', f'{y_label}', 'NC'] 
    true_labels = [f'{x_label}', f'{y_label}'] 

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

    plt.savefig(f'{folder_path}/confusion_matrices.png')

    plt.show()


    ##### ACCURACY AND EFFICIENCY

    thr = 0.01 + np.arange(100) * 0.01

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # General plot
    axes[0].plot(thr, acc_array, label="Accuracy", color="blue", linestyle="-")
    axes[0].plot(thr, eff_array, label="Efficiency", color="green", linestyle="-")
    axes[0].set_title("Accuracy and Efficiency vs Threshold (General)")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Value")
    axes[0].grid(True)
    axes[0].legend()

    # 0 plot
    axes[1].plot(thr, acc_array_0, label=f"Accuracy ({x_label})", color="blue", linestyle="-")
    axes[1].plot(thr, eff_array_0, label=f"Efficiency ({x_label})", color="green", linestyle="-")
    axes[1].set_title(f"Accuracy and Efficiency vs Threshold ({x_label})")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Value")
    axes[1].grid(True)
    axes[1].legend()

    # 1 plot
    axes[2].plot(thr, acc_array_1, label=f"Accuracy ({y_label})", color="blue", linestyle="-")
    axes[2].plot(thr, eff_array_1, label=f"Efficiency ({y_label})", color="green", linestyle="-")
    axes[2].set_title(f"Accuracy and Efficiency vs Threshold ({y_label})")
    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("Value")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'{folder_path}/accuracy_efficiency.png')
    plt.show()


    ##########################
    ######### OUTPUT #########
    ##########################

    def append_to_file(file_path, row):
        with open(file_path, "a") as file:
            file.write(row + "\n")


    append_to_file(target_file, f"{segmentation}\t{acc_standard}\t{lower_bound}\t{upper_bound}")
    print(f"{segmentation}\t{acc_standard}\t{lower_bound}\t{upper_bound}")

else:
    
    print( "Not Implemented" )
    # ########################
    # ######### TEST #########
    # ########################
    
    # # Predictions on the test set
    # y_pred_proba = best_model.predict(dtest)
    # y_pred = np.argmax(y_pred_proba, axis=1)
    
    # ###########################
    # ######### METRICS #########
    # ###########################

    # # legend:
    # #   if p  -> 0
    # #   if pi -> 1
    # #   if k  -> 2
    
    # ###### FEATURE IMPORTANCE

    # plt.figure(figsize=(10, 8))

    # xgb.plot_importance(best_model, importance_type='gain', max_num_features=20, title="Feature Importance", height=0.8)
    # plt.tight_layout()
    # plt.savefig(f'{folder_path}/feature_importance_best_model.png')

    # plt.close()
    
    # ##### CONFIDENCE DISTRIBUTION

    # def split_by_position(probs, y_test):
        
    #     array_1 = [] 
    #     array_2 = [] 
        
    #     for i, prob in enumerate(probs):
            
    #         max_idx = np.argmax(prob)
    #         max_el = max(prob)
            
    #         if y_test[i] == max_idx:
    #             array_1.append(max_el)
    #         else:            
    #             array_2.append(1 - max_el)
        
    #     return np.array(array_1), np.array(array_2)


    # true_class_probs, false_class_probs = split_by_position(y_pred_proba, y_test.values)

    # # Plotting the distributions
    # plt.figure(figsize=(8, 6))

    # plt.hist(true_class_probs, bins=50, alpha=0.5, color='blue', label='Right Prediction')
    # plt.hist(false_class_probs, bins=50, alpha=0.5, color='red', label='Wrong Prediction')
    # plt.xlabel('Predicted Probability')
    # plt.ylabel('Counts')
    # plt.title('Distribution of Predicted Probabilities for True vs False Class')
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f'{folder_path}/predicted_probability_distribution.png')

    # plt.close()

    # ##### TRAINING AND VALIDATION LOSSES

    # # Plotting the loss for each epoch (training and validation)
    # epochs = len(evals_result['train']['logloss'])

    # plt.figure(figsize=(12, 6))

    # plt.plot(range(epochs), evals_result['train']['logloss'], label='Training Loss')
    # plt.plot(range(epochs), evals_result['validation']['logloss'], label='Validation Loss')
    # plt.xlabel('Boosting Round')
    # plt.ylabel('Log Loss')
    # plt.title('Training and Validation Loss per Epoch')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f'{folder_path}/training_validation_loss.png')

    # plt.close()
    
    # ##### CONFUSION MATRIX (WITH AND WITHOUT NC)

    # def createNC_matrix(y_test, y_pred_prob,y_pred, thr):
        
    #     max_probs = np.max(y_pred_prob, axis=1)
    #     mask = (max_probs > thr) | (max_probs < (1 - thr))
    #     pred_with_nd = np.where(mask, y_pred, 'NC')
        
    #     cm_withNC = np.zeros((3, 4), dtype=int)  # 3x4 matrix: 3 true labels (0, 1, 2) and43 predicted labels (0, 1,2 ,NC)

    #     for true, pred in zip(y_test, pred_with_nd):
            
    #         if true == 0:
    #             if pred == "0":
    #                 cm_withNC[0, 0] += 1  # True 0, Pred 0
    #             elif pred == "1":
    #                 cm_withNC[0, 1] += 1  # True 0, Pred 1
    #             elif pred == "2":
    #                 cm_withNC[0, 2] += 1  # True 0, Pred 2
    #             elif pred == 'NC':
    #                 cm_withNC[0, 3] += 1  # True 0, Pred ND
    #         elif true == 1:
    #             if pred == "0":
    #                 cm_withNC[1, 0] += 1  # True 1, Pred 0
    #             elif pred == "1":
    #                 cm_withNC[1, 1] += 1  # True 1, Pred 1
    #             elif pred == "2":
    #                 cm_withNC[1, 2] += 1  # True 1, Pred 2
    #             elif pred == 'NC':
    #                 cm_withNC[1, 3] += 1  # True 1, Pred ND
                
    #     return cm_withNC

    # def optimal_matrix(y_test, y_pred_prob,y_pred_binary):
        
    #     acc_array = []
    #     eff_array = []
        
    #     acc_array_0 = []
    #     eff_array_0 = []

    #     acc_array_1 = []
    #     eff_array_1 = []
        
    #     acc_array_2 = []
    #     eff_array_2 = []

        
    #     for i in range(100):
            
    #         thr_test = 0.01 + 0.01*i
    #         cm = createNC_matrix(y_test, y_pred_binary, y_pred_prob,thr_test)
            
    #         denominator = cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0] + cm[2,2] + cm[1,2] + cm[0,2] + cm[2,0] + cm[2,1]
    #         denominator_0 = cm[0, 0] + cm[0, 1] + cm[0,2]
    #         denominator_1 = cm[1, 0] + cm[1, 1] + cm[1,2]
    #         denominator_2 = cm[2, 0] + cm[2, 1] + cm[2,2]

    #         if denominator == 0:
    #             acc = 0
    #         else:
    #             acc = (cm[0, 0] + cm[1, 1] +cm[2,2]) / denominator

    #         if denominator_0 == 0:
    #             acc_0 = 0   
    #         else:
    #             acc_0 = (cm[0, 0]) / denominator_0

    #         if denominator_1 == 0:
    #             acc_1 = 0
    #         else:
    #             acc_1 = (cm[1, 1]) / denominator_1
                
    #         if denominator_2 == 0:
    #             acc_2 = 0
    #         else:
    #             acc_2 = (cm[2, 2]) / denominator_2
            
            
    #         eff = (denominator) / np.sum(cm)
    #         eff_0 = (denominator_0) / (cm[0, 0] + cm[0, 1]+ cm[0, 2]+cm[0,3])
    #         eff_1 = (denominator_1) / (cm[1, 0] + cm[1, 1]+ cm[1, 2]+cm[1,3])
    #         eff_2 = (denominator_2) / (cm[2, 0] + cm[2, 1]+ cm[2, 2]+cm[2,3])

    #         acc_array.append(acc)
    #         eff_array.append(eff)

    #         acc_array_0.append(acc_0)
    #         eff_array_0.append(eff_0)
            
    #         acc_array_1.append(acc_1)
    #         eff_array_1.append(eff_1)
            
    #         acc_array_2.append(acc_2)
    #         eff_array_2.append(eff_2)

            
    #     thr = 0.01 + np.arange(100) * 0.01

    #     #total info
    #     acc_array = np.array(acc_array)
    #     eff_array = np.array(eff_array)
        
    #     acc_interp = interp1d(thr, acc_array, kind='linear', fill_value="extrapolate")
    #     eff_interp = interp1d(thr, eff_array, kind='linear', fill_value="extrapolate")
    #     thresholds = np.linspace(thr.min(), thr.max(), 1000)
    #     differences = acc_interp(thresholds) - eff_interp(thresholds)
    #     intersection_idx = np.where(np.diff(np.sign(differences)))[0][0]
    #     intersection_thr = thresholds[intersection_idx]
    #     intersection_value = acc_interp(intersection_thr)
        
    #     cm_acc = createNC_matrix(y_test, y_pred_binary, y_pred_prob,intersection_thr)

    #     # 0 info
    #     acc_array_0 = np.array(acc_array_0)
    #     eff_array_0 = np.array(eff_array_0)

    #     # 1 info
    #     acc_array_1 = np.array(acc_array_1)
    #     eff_array_1 = np.array(eff_array_1)
        
    #     # 2 info
    #     acc_array_2 = np.array(acc_array_2)
    #     eff_array_2 = np.array(eff_array_2)
        
        
    #     return cm_acc, \
    #             intersection_thr, intersection_value, acc_array, eff_array, \
    #             acc_array_0, eff_array_0, \
    #             acc_array_1, eff_array_1, \
    #             acc_array_2, eff_array_2
                    
                
    # cm_standard = confusion_matrix(y_test, y_pred, labels=[0, 1])
    # acc_standard = accuracy_score(y_pred,y_test)
    
    
    # # CLOPPER-PEARSON INTERVAL

    # #acc = (n_tn + n_tp)/n = n_t / n
    #     # n_tn = cm[0,0]
    #     # m_tp = cm[1,1]
    #     # n = np.sum(cm)

    # alpha = 0.32
    # n_t = cm_standard[0,0] + cm_standard[1,1] + cm_standard[2,2]
    # n = np.sum(cm_standard)

    # lower_bound = beta.ppf(alpha / 2, n_t, n - n_t + 1)
    # upper_bound = beta.ppf(1 - alpha / 2, n_t + 1, n - n_t)


    # cm_acc, \
    #     thr_acc, acc_withNC, acc_array, eff_array,\
    #     acc_array_0, eff_array_0, \
    #     acc_array_1, eff_array_1, \
    #     acc_array_2, eff_array_2 = optimal_matrix(y_test, y_pred_proba,y_pred)

    # pred_labels_standard = ['proton', 'pion','kaon'] 
    # pred_labels_with_nc = ['proton', 'pion','kaon', 'NC'] 
    # true_labels = ['proton', 'pion','kaon']

    # fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # heatmap_kwargs = {
    #     'annot': True,
    #     'fmt': "d",
    #     'cmap': 'Blues',
    #     'vmin': 0,
    #     'vmax': 7500, 
    #     'annot_kws': {"size": 16},
    #     'cbar': False 
    # }
    # sns.heatmap(
    #     cm_standard, 
    #     xticklabels=pred_labels_standard, 
    #     yticklabels=true_labels, 
    #     ax=axes[0],
    #     **heatmap_kwargs
    # )
    # axes[0].set_title(f"Standard Confusion Matrix\nAccuracy: {acc_standard:.3f}, 68% CI [{lower_bound:.3f}, {upper_bound:.3f}]", fontsize=18)
    # axes[0].set_xlabel("Predicted", fontsize=14)
    # axes[0].set_ylabel("True", fontsize=14)
    # sns.heatmap(
    #     cm_acc, 
    #     xticklabels=pred_labels_with_nc, 
    #     yticklabels=true_labels, 
    #     ax=axes[1],
    #     **heatmap_kwargs
    # )
    # axes[1].set_title(f"Confusion Matrix with NC\nThreshold: {thr_acc:.3f}, Accuracy: {acc_withNC:.3f}", fontsize=18)
    # axes[1].set_xlabel("Predicted", fontsize=14)
    # axes[1].set_ylabel("True", fontsize=14)
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    # fig.colorbar(axes[0].collections[0], cax=cbar_ax)
    # for ax in axes:
    #     ax.tick_params(labelsize=14) 
    #     ax.title.set_fontsize(18)   
    #     ax.xaxis.label.set_fontsize(14)
    #     ax.yaxis.label.set_fontsize(14)

    # plt.savefig(f'{folder_path}/confusion_matrices.png')

    # plt.show()
    
    # ##### ACCURACY AND EFFICIENCY

    # thr = 0.01 + np.arange(100) * 0.01

    # # Interpolation for general data
    # acc_interp_0 = interp1d(thr, acc_array_0, kind='linear', fill_value="extrapolate")
    # acc_interp_1 = interp1d(thr, acc_array_1, kind='linear', fill_value="extrapolate")
    # thresholds = np.linspace(thr.min(), thr.max(), 1000)
    # differences = acc_interp_0(thresholds) - acc_interp_1(thresholds)
    # intersection_idx = np.where(np.diff(np.sign(differences)))[0][0]
    # intersection_thr = thresholds[intersection_idx]
    # intersection_value = acc_interp_0(intersection_thr)

    # # # Interpolation for 0s
    # # acc_interp_0 = interp1d(thr, acc_array_0, kind='linear', fill_value="extrapolate")
    # # eff_interp_0 = interp1d(thr, eff_array_0, kind='linear', fill_value="extrapolate")
    # # thresholds = np.linspace(thr.min(), thr.max(), 1000)
    # # differences_0 = acc_interp_0(thresholds) - eff_interp_0(thresholds)
    # # intersection_idx_0 = np.where(np.diff(np.sign(differences_0)))[0][0]
    # # intersection_thr_0 = thresholds[intersection_idx_0]
    # # intersection_value_0 = acc_interp_0(intersection_thr_0)

    # # # Interpolation for 1s
    # # acc_interp_1 = interp1d(thr, acc_array_1, kind='linear', fill_value="extrapolate")
    # # eff_interp_1 = interp1d(thr, eff_array_1, kind='linear', fill_value="extrapolate")
    # # thresholds = np.linspace(thr.min(), thr.max(), 1000)
    # # differences_1 = acc_interp_1(thresholds) - eff_interp_1(thresholds)
    # # intersection_idx_1 = np.where(np.diff(np.sign(differences_1)))[0][0]
    # # intersection_thr_1 = thresholds[intersection_idx_1]
    # # intersection_value_1 = acc_interp_1(intersection_thr_1)
    
    # # # Interpolation for 1s
    # # acc_interp_2 = interp1d(thr, acc_array_2, kind='linear', fill_value="extrapolate")
    # # eff_interp_2 = interp1d(thr, eff_array_2, kind='linear', fill_value="extrapolate")
    # # thresholds = np.linspace(thr.min(), thr.max(), 1000)
    # # differences_2 = acc_interp_2(thresholds) - eff_interp_2(thresholds)
    # # intersection_idx_2 = np.where(np.diff(np.sign(differences_2)))[0][0]
    # # intersection_thr_2 = thresholds[intersection_idx_2]
    # # intersection_value_2 = acc_interp_1(intersection_thr_2)

    # # Plotting
    # fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # # General plot
    # axes[0].plot(thr, acc_array, label="Accuracy", color="blue", linestyle="-")
    # axes[0].plot(thr, eff_array, label="Efficiency", color="green", linestyle="-")
    # axes[0].axvline(intersection_thr, color="red", linestyle=":", label=f"Intersection at {intersection_thr:.2f}")
    
    # axes[0].set_title("Accuracy and Efficiency vs Threshold (General)")
    # axes[0].set_xlabel("Threshold")
    # axes[0].set_ylabel("Value")
    # axes[0].grid(True)
    # axes[0].legend()

    # # proton plot
    # axes[1].plot(thr, acc_array_0, label="Accuracy (proton)", color="blue", linestyle="-")
    # axes[1].plot(thr, eff_array_0, label="Efficiency (proton)", color="green", linestyle="-")
    # axes[1].axvline(intersection_thr, color="red", linestyle=":", label=f"Intersection at {intersection_thr:.2f}")
    
    # axes[1].set_title("Accuracy and Efficiency vs Threshold (proton)")
    # axes[1].set_xlabel("Threshold")
    # axes[1].set_ylabel("Value")
    # axes[1].grid(True)
    # axes[1].legend()

    # # pion plot
    # axes[2].plot(thr, acc_array_1, label="Accuracy (pion)", color="blue", linestyle="-")
    # axes[2].plot(thr, eff_array_1, label="Efficiency (pion)", color="green", linestyle="-")
    # axes[2].axvline(intersection_thr, color="red", linestyle=":", label=f"Intersection at {intersection_thr:.2f}")
    
    # axes[2].set_title("Accuracy and Efficiency vs Threshold (pion)")
    # axes[2].set_xlabel("Threshold")
    # axes[2].set_ylabel("Value")
    # axes[2].grid(True)
    # axes[2].legend()
    
    # # kaon plot
    # axes[3].plot(thr, acc_array_2, label="Accuracy (kaon)", color="blue", linestyle="-")
    # axes[3].plot(thr, eff_array_2, label="Efficiency (kaon)", color="green", linestyle="-")
    # axes[3].axvline(intersection_thr, color="red", linestyle=":", label=f"Intersection at {intersection_thr:.2f}")
    
    # axes[3].set_title("Accuracy and Efficiency vs Threshold (kaon))")
    # axes[3].set_xlabel("Threshold")
    # axes[3].set_ylabel("Value")
    # axes[3].grid(True)
    # axes[3].legend()

    # plt.tight_layout()
    # plt.savefig(f'{folder_path}/accuracy_efficiency.png')
    # plt.show()
    
    # ##########################
    # ######### OUTPUT #########
    # ##########################

    # def append_to_file(file_path, row):
    #     with open(file_path, "a") as file:
    #         file.write(row + "\n")


    # append_to_file(target_file, f"{segmentation}\t{acc_standard}\t{lower_bound}\t{upper_bound}\t{intersection_thr}\t{intersection_value}")
    # print(f"{segmentation}\t{acc_standard}\t{lower_bound}\t{upper_bound}\t{intersection_thr}\t{intersection_value}")
