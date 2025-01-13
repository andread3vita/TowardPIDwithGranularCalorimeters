# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from simpleDNN import SimpleDNN
from torch.nn import functional as F
import os
from scipy.stats import beta
from tqdm.auto import tqdm

tqdm(disable=True)

#data_set name with accuracy dict
dict_acc={}
dict_ci={}

class CustomDataset(Dataset):
    def __init__(self, csv_file:str, features:list):
        # Load the data
        self.data = pd.read_csv(csv_file)
        self.data['class_label'] = np.argmax(self.data[['label_proton', 'label_pion']].values, axis=1)
        self.data = self.data.drop(['label_proton', 'label_pion'], axis=1)
        self.data = self.data.dropna()
        self.X = self.data[features].values
        # self.X = StandardScaler().fit_transform(self.X)
        self.X = QuantileTransformer().fit_transform(self.X)
        self.y = self.data['class_label'].values 
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return one sample of data
        X_sample = torch.tensor(self.X[idx], dtype=torch.float32)
        y_sample = torch.tensor(self.y[idx], dtype=torch.long)
        return X_sample, y_sample

# Step 2: Use DataLoader

def create_dataloader(csv_file, batch_size=32, shuffle=True,num=1,train=0.6,valid=0.2,test=0.2,features=[]):
    dataset = CustomDataset(csv_file, features=features)
   
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    train=int(train*len(dataset))
    valid=int(valid*len(dataset))
    test=int(test*len(dataset))
    print(f"Train: {train}, Valid: {valid}, Test: {test}")
    indices0 = np.where(dataset.y==0)[0]
    indices1 = np.where(dataset.y==1)[0]
    train0_indices,valid0_indices,test0_indices = indices0[:int(train/2)], indices0[int(train/2):int(train/2)+int(valid/2)],indices0[int(train/2)+int(valid/2):]
    train1_indices,valid1_indices,test1_indices = indices1[:int(train/2)], indices1[int(train/2):int(train/2)+int(valid/2)],indices1[int(train/2)+int(valid/2):]
    train_indices = np.concatenate([train0_indices,train1_indices])
    valid_indices = np.concatenate([valid0_indices,valid1_indices])
    test_indices = np.concatenate([test0_indices,test1_indices])
    # train_indices,valid_indices,test_indices = indices[:train], indices[train:valid+train],indices[train+valid:]
    print(f"Train: {len(train_indices)}, Valid: {len(valid_indices)}, Test: {len(test_indices)}")   
    train_loader=DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_indices),num_workers=num)
    valid_loader=DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(valid_indices),num_workers=num)
    test_loader=DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_indices),num_workers=num)
    return train_loader,valid_loader,test_loader


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device



# from tqdm import tqdm

def train(model, device, train_loader, optimizer):
    
    train_loss_ep = 0.
    
    model.train()
    with tqdm.tqdm(train_loader, ascii=True,disable=True) as tq:
        for data, target in tq:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # loss = F.nll_loss(output, target)
            loss= F.cross_entropy(output, target) # It is not recommended to use cross entropy loss for Binary classification.
            loss.backward()
            optimizer.step()  
            train_loss_ep += loss.item() * data.size(0)
        
    return train_loss_ep

def test(model, device, valid_loader1):
    
    test_loss_ep = 0.
    model.eval()
    with tqdm.tqdm(valid_loader1, ascii=True,disable=True) as tq:
        for data, target in tq:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # loss = F.nll_loss(output, target)
            loss= F.cross_entropy(output, target)
            
            test_loss_ep += loss.item() * data.size(0)
        
    return test_loss_ep
        

#directory = "/home/alma1/GNN/Deepset/28oct_DNN/csv/"
directory = "/home/alma1/GNN/Deepset/28oct_DNN/dataset/Merged/"
for root,_,files in os.walk(directory):
    length=len(files)
    for filename in files:
        if filename.endswith(".csv"):
            csv_file = os.path.join(root, filename)
            
            length-=1
            
            #f lis a list of features to be used in the model (you can add or reove time0 here)
            # for case in ["with_time", "without_time"]:
            name=csv_file.split("/")[-1].split(".")[0]
            f=['E_LR_geom', 'Asymmetry_w', 'E_LR', 'Asymmetry', 'numPeaks', 'E1', 'R1',
                'TotalEnergy', 'NumberOfUniqueCells', 'MaxEnergyInCell',
                'SecondMaxEnergyInCell', 'distanceFirstSecondMaxEnergy', 'RatioEcell',
                'DeltaEcell_secondMax', 'EfractionCell', 'weightedTime', 'radius',
                'radialSigma', 'length', 'longitudinalSigma', 'radius_plain',
                'radialSigma_plain', 'length_plain', 'longitudinalSigma_plain',
                'Z_vertex', 'VertexTime', 'PostVertexEnergyFraction',
                'numCellBeforeVertex', 'DeltaT', 'TotalEnergyCloseToVertex',
                'EnergyFractionCloseToVertex', 'MaxEnergyCloseVertex',
                'VarianceAtVertex', 'distanceMaxFromVertex',
                'centralTowerFraction_cell', 'time0', 'speed']
            
# without numofuniquecells and totalenergy
            # if case=="with_time":
            #     f=["time0"]+f
            # else:
            #     f=f

            # if 'time0' in f:
            #     name=name+"_time"
            # else:
            #     name=name+"_notime"
            
            print(name,"reamaing files:",length)
            train_loader,valid_loader,test_loader = create_dataloader(csv_file, batch_size=256,num=28,features=f)
            for X_batch, y_batch in train_loader:
                print(X_batch, len(y_batch))
                print(X_batch.shape)
                print(y_batch.shape)
                input_size = X_batch.shape[1]
                break
            hidden_layers = [96,32,16,4]
            model = SimpleDNN(input_size, hidden_layers) # I also tried to use weights=weights in the model but it did not work
            model.to(device)
            torch.save(model.state_dict(), f"/home/alma1/GNN/Deepset/28oct_DNN/TowardPIDwithGranularCalorimeters/results/DNN/models/PID_hcal_{name}.pt")

            

            # %%
            def eval_model(model, data_loader):
                model.eval() # Set model to eval mode
                true_preds, num_preds = 0., 0.
                
                with torch.no_grad(): # Deactivate gradients for the following code
                    for data_inputs, data_labels in data_loader:
                        
                    
                        data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
                        preds = model(data_inputs)
                        
                    
                        pred_labels = torch.argmax(preds, dim=-1) # Binarize predictions to 0 and 10
                        
                        # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
                        true_preds += (pred_labels == data_labels).sum()
                        num_preds += data_labels.shape[0]
                        
                acc = true_preds / num_preds
                # print(f"Accuracy of the model: {100.0*acc:4.2f}%")
                return 100.0*acc.item()
            def Eval_model(model, data_loader):
                model.eval() # Set model to eval mode
                true_preds, num_preds = 0., 0.
                empty_tensor = torch.tensor([]).to(device)
                with torch.no_grad(): # Deactivate gradients for the following code
                    for data_inputs, data_labels in data_loader:

                        data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
                        preds = model(data_inputs)
                        preds=F.softmax(preds,dim=1)
                        
                        pred_labels = torch.argmax(preds, dim=-1) # Binarize predictions to 0 and 1
                        predicted_confidence = preds.gather(1, pred_labels.view(-1, 1)).squeeze()  # Shape: (128,)
                        # print(f"probabilities: {preds.shape} labels: {pred_labels.shape} true labels: {data_labels.shape}")
                        result_tensor = torch.cat([
                        preds,             # Probabilities after softmax (128, 2)
                        pred_labels.unsqueeze(1).float(),  # Predicted labels (128, 1)
                        data_labels.unsqueeze(1).float(),       # True labels (128, 1)
                        predicted_confidence.unsqueeze(1)       # Prediction confidence (128, 1)
                        ], dim=1)


                        # print(result_tensor.shape)
                        same_elements = (result_tensor[:, 2] == result_tensor[:, 3])

                        # Count how many times the elements are the same
                        count_same = same_elements.sum().item()
                        true_preds += count_same
                        num_preds +=len(result_tensor)
                        empty_tensor = torch.cat([empty_tensor,result_tensor],dim=0)
                        
                acc = true_preds / num_preds
                print(f"Accuracy of the model: {100.0*acc:4.2f}%")
                return empty_tensor, f"{100.0*acc:4.2f}%"

            # %%
            import pandas as pd
            import tqdm
            n_epochs = 180
            loss_data = []

            valid_loss_min = np.inf  # set initial "min" to infinity

            for epoch in range(n_epochs):
                model_test = SimpleDNN(input_size, hidden_layers)
                model_test.to(device)
                model_test.load_state_dict(torch.load(f"/home/alma1/GNN/Deepset/28oct_DNN/TowardPIDwithGranularCalorimeters/results/DNN/models/PID_hcal_{name}.pt",weights_only=True))  
            
                weight_decay=0.001
                lr=0.0009
                lr=lr/(1+weight_decay*(epoch)**2.5)
                if epoch%10==0:
                    lr=0.00005
                # print(f" LR: {np.round(lr,6)}")
                optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)  
                # optimizer = optim.AdamW(model.parameters(), lr=lr)  
                # optimizer = optim.SGD(model.parameters(), lr=lr,weight_decay=weight_decay,momentum=0.9)  #SGD optimizer is pretty slow compared to Adam
                train_loss = train(model, device, train_loader, optimizer)
                valid_loss = test(model, device, valid_loader)
                
                
                train_loss = train_loss / len(train_loader.sampler)
                valid_loss = valid_loss / len(valid_loader.sampler)

            

                loss_data.append({'Epoch': epoch + 1, 'Training Loss': train_loss, 'Validation Loss': valid_loss,"Training_Accuracy":eval_model(model, train_loader),"Validation_Accuracy": eval_model(model_test, test_loader)})

                print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                    epoch+1, 
                    train_loss,
                    valid_loss
                    ))
                
                # save model if validation loss has decreased
                if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                    torch.save(model.state_dict(), f"/home/alma1/GNN/Deepset/28oct_DNN/TowardPIDwithGranularCalorimeters/results/DNN/models/PID_hcal_{name}.pt")
                    valid_loss_min = valid_loss
                loss_history_df = pd.DataFrame(loss_data)
                loss_history_df.to_csv('loss_history.csv', index=False)

            # %%
            model_test = SimpleDNN(input_size, hidden_layers)
            model_test.to(device)
            model_test.load_state_dict(torch.load(f"/home/alma1/GNN/Deepset/28oct_DNN/TowardPIDwithGranularCalorimeters/results/DNN/models/PID_hcal_{name}.pt",weights_only=True))


            # saving accuracy in a
            # %%
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            from scipy.stats import beta

            def plot_confusion_matrix(model, data_loader):
                model.eval() # Set model to eval mode
                true_labels = []
                pred_labels = []
                
                with torch.no_grad(): # Deactivate gradients for the following code
                    for data_inputs, data_labels in data_loader:

                        data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
                        preds = model(data_inputs)
                        pred_labels.extend(torch.argmax(preds, dim=-1).cpu().numpy()) # Binarize predictions to 0 and 1
                        true_labels.extend(data_labels.cpu().numpy())
                        
                cm = confusion_matrix(true_labels, pred_labels)
                plt.figure(figsize=(10, 8))
                colors = ["#cce5ff", "#004c99"]  # Light blue to dark blue
                cmap = LinearSegmentedColormap.from_list("Custom Blue", colors)
                sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                        xticklabels=["Proton", "Pion"], yticklabels=["Proton", "Pion"],
                        annot_kws={"size": 18, "weight": "bold"})  # Set annot font size and weight
                TP = cm[0, 0]
                TN = cm[1, 1]
                FP = cm[1, 0]
                FN = cm[0, 1]
                alpha = 0.32  # 95% confidence interval
                n_total = TP + TN + FP + FN
                n_correct = TP + TN

            # Calculate accuracy
                accuracy = n_correct / n_total
                lower_bound = beta.ppf(alpha / 2, n_correct, n_total - n_correct + 1)
                upper_bound = beta.ppf(1 - alpha / 2, n_correct + 1, n_total - n_correct)
                # Print results
                print(f"Accuracy: {accuracy*100:.2f}")
                print(f"95% Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")

                # Add title to the plot
                title = f"Accuracy: {accuracy*100:.2f}%, 68% CI: [{lower_bound*100:.2f}%, {upper_bound*100:.2f}%]"
                
                print(f"cm values: {cm}")
                plt.xticks(size=14, weight='bold')
                plt.yticks(size=14, weight='bold')
                plt.title('Confusion Matrix', size=15, weight='bold')
                plt.xlabel('Predicted', size=14, weight='bold')
                plt.ylabel('True', size=14, weight='bold')
                plt.title(title,weight='bold',size=15)
                
                plt.savefig(f'/home/alma1/GNN/Deepset/28oct_DNN/DNN_plots/confusion_matrix_{name}.png')
                plt.savefig(f'/home/alma1/GNN/Deepset/28oct_DNN/DNN_plots/confusion_matrix_{name}.pdf')
                return cm
                

            cm=plot_confusion_matrix(model_test, test_loader)
            
            dict_acc[name]=eval_model(model_test, test_loader)
            dff=pd.DataFrame(dict_acc.items(),columns=["Name","Accuracy"])
            dff.to_csv("accuracy.csv",index=False)
            
            ## Confidence interval
            TP = cm[0,0]
            TN = cm[1,1]
            FP = cm[1,0]
            FN = cm[0,1]
            alpha = 0.32  # 95% confidence interval

            # Total predictions
            n_total = TP + TN + FP + FN
            n_correct = TP + TN

            # Calculate accuracy
            accuracy = n_correct / n_total

            # Clopper-Pearson interval
            lower_bound = beta.ppf(alpha / 2, n_correct, n_total - n_correct + 1)
            upper_bound = beta.ppf(1 - alpha / 2, n_correct + 1, n_total - n_correct)
            
            dict_ci[name]= lower_bound, upper_bound
            dff=pd.DataFrame(dict_ci.items(),columns=["Name","CI"])
            dff.to_csv("CI.csv",index=False)


            # %%
            import pandas as pd
            import matplotlib.pyplot as plt

            # Load the CSV file
            df = pd.read_csv('loss_history.csv')
            df=df.drop(index=0).reset_index(drop=True)
            print(df.head())

            # Set up the figure and axis
            fig, ax1 = plt.subplots(figsize=(10, 6))
            # plt.style.use('dark_background')

            # Plot Training and Validation Loss on the left y-axis
            ax1.plot(df['Epoch'], df['Training Loss'], label='Training Loss', color='red', alpha=0.9, linewidth=3)
            ax1.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss', color='cyan', alpha=0.9, linewidth=3)
            ax1.set_xlabel('Epoch', fontsize=18)
            ax1.set_ylabel('Loss', fontsize=18, color='black')
            ax1.tick_params(axis='y', labelcolor='black',labelsize=16)
            ax1.tick_params(axis='x', labelcolor="black",labelsize=16)
            ax1.set_ylim(0.38, 0.7)  # Set y-axis limits for Loss

            # Create a second y-axis for accuracy on the right side
            ax2 = ax1.twinx()
            #ax2.plot(df['Epoch'], df['Training Accuracy'], label='Training Accuracy', color='orange', alpha=0.7, linestyle='--', linewidth=2)
            ax2.plot(df['Epoch'], df['Validation_Accuracy'], label='Validation Accuracy', color='lime', alpha=0.8, linestyle='--', linewidth=2)
            ax2.plot(df['Epoch'], df['Training_Accuracy'], label='Training Accuracy', color='orange', alpha=0.7, linestyle='--', linewidth=2)
            ax2.set_ylabel('Accuracy (%)', fontsize=18, color='black')
            ax2.tick_params(axis='y', labelcolor='black',labelsize=16)
            ax2.set_ylim(50, 100)  # Set y-axis limits for Accuracy in %

            # Combine legends for both y-axes
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper right')

            ax1.grid(visible=True,which="both",linestyle="--",linewidth=0.5)


            plt.savefig(f"/home/alma1/GNN/Deepset/28oct_DNN/DNN_plots/lr_accuracy_{name}.pdf")
            plt.savefig(f"/home/alma1/GNN/Deepset/28oct_DNN/DNN_plots/lr_accuracy_{name}.png")
            
            data_tensor, acc = Eval_model(model_test, test_loader)
            pion_pred=data_tensor[:,1].cpu().numpy()
            true_labels = data_tensor[:, 3].cpu().numpy()       # True labels
            predicted_labels = data_tensor[:, 2].cpu().numpy()  # Predicted labels
            confidence = data_tensor[:, 4].cpu().numpy()        # Confidence in percentage

            import matplotlib.pyplot as plt
            # Plotting
            plt.figure(figsize=(10, 6))

            # Scatter plot for each class
            conf_median = np.median(confidence[true_labels == predicted_labels])
            plt.hist(confidence[true_labels == predicted_labels]*100., bins=50, color='blue', alpha=0.5, label='True_prediction',density=True,histtype='stepfilled')
            plt.axvline(x=conf_median*100, color='blue', linestyle='--', label=f'Median Confidence = {conf_median*100:.2f}%')
            plt.xlabel("Confidence (%)",fontsize=12)
            plt.ylabel("True Predictions",fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(f"Accuracy of the model:{acc} ",fontsize=12)
            plt.title("Confidence for Binary Classification")
            plt.legend()
            plt.savefig(f"/home/alma1/GNN/Deepset/28oct_DNN/DNN_plots/confidence_true_{name}.pdf")
            
            plt.figure(figsize=(10, 6))

            plt.hist(confidence[true_labels != predicted_labels]*100., bins=50, color='red', alpha=0.5, label='False_prediction',density=True,histtype='stepfilled')
            conf_median = np.median(confidence[true_labels != predicted_labels])
            plt.axvline(x=conf_median*100, color='red', linestyle='--', label=f'Median Confidence = {conf_median*100:.2f}%')
            plt.xlabel("Confidence (%)",fontsize=12)
            plt.ylabel("False Predictions",fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(f"Accuracy of the model:{acc} ",fontsize=12)
            plt.title("Confidence for Binary Classification")
            plt.legend()
            plt.savefig(f"/home/alma1/GNN/Deepset/28oct_DNN/DNN_plots/confidence_false_{name}.pdf")
            
            #ROC curve
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, auc

            

            y_true =true_labels      # True labels
            y_scores = pion_pred  # Predicted scores


            # Calculate false positive rate, true positive rate, and thresholds
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)

            N_x = sum(y_true == 0)
            N_y = sum(y_true == 1)

            sigma_fpr = np.sqrt(fpr * (1 - fpr) / N_x)
            sigma_tpr = np.sqrt(tpr * (1 - tpr) / N_y)
            # Calculate AUC (Area Under the Curve)
            roc_auc = auc(fpr, tpr)

            # Plot the ROC curve
            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.fill_between(
                    fpr,
                    tpr - 5*sigma_tpr,
                    tpr + 5*sigma_tpr,
                    color='blue',
                    alpha=0.25,
                    label='1-sigma region (PiPR) [5x]'
                )
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Guess')  # Diagonal line
            plt.xlabel('Proton Positive Rate',fontsize=12)
            plt.ylabel('Pion Positive Rate',fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc='lower right')
            plt.grid()
            plt.savefig(f"/home/alma1/GNN/Deepset/28oct_DNN/DNN_plots/roc_curve_{name}.pdf")
            plt.savefig(f"/home/alma1/GNN/Deepset/28oct_DNN/DNN_plots/roc_curve_{name}.png")
            
            #Accuracy vs Efficiency analysis
            
            from scipy.interpolate import interp1d
            from scipy.optimize import fsolve
            proton_indices = np.where(true_labels == 0)[0]
            pion_indices = np.where(true_labels == 1)[0]

            proton_predictions = predicted_labels[proton_indices]
            pion_predictions = predicted_labels[pion_indices]

            true_predictions = (true_labels == predicted_labels)
            print(predicted_labels) 
            # Initialize lists to store accuracy and efficiency
            thresholds = np.linspace(0, 1, 100)
            accuracies = []
            efficiencies = []

            efficiencies_proton = []
            accuracies_proton = []
            efficiencies_pion = []
            accuracies_pion = []

            # Loop over thresholds
            for threshold in thresholds:
                classified = confidence > threshold  # Predictions above the threshold
                classified_proton = confidence[true_labels==0] > threshold
                classified_pion = confidence[true_labels==1] > threshold
                num_classified = classified.sum()
                num_true = true_predictions[classified].sum() if num_classified > 0 else 0

                num_proton_classified = classified_proton.sum()
                #num_proton = (predicted_labels[classified] == 0).sum() if num_proton_classified > 0 else 0 
                num_proton = ((predicted_labels ==0) & (true_labels==0) & classified).sum() if num_proton_classified > 0 else 0
                num_pion_classified = classified_pion.sum()    
                num_pion = ((predicted_labels ==1) & (true_labels==1) & classified).sum() if num_pion_classified > 0 else 0

                #print(f"num_classified: {num_classified} num_true: {num_true} num_proton_classified: {num_proton_classified} num_proton: {num_proton} num_pion_classified: {num_pion_classified} num_pion: {num_pion}")
                
                efficiency = num_classified / len(confidence)

                efficiency_proton = num_proton_classified / len(proton_indices) if len(proton_indices) > 0 else 0
                accuracy = num_true / num_classified if num_classified > 0 else 0
                accuracy_proton = num_proton / num_proton_classified if num_proton_classified > 0 else 0

                efficiency_pion = num_pion_classified / len(pion_indices) if len(pion_indices) > 0 else 0
                accuracy_pion = num_pion / num_pion_classified if num_pion_classified > 0 else 0
                # accuracy_pion = (pion_predictions == 1).sum() / num_pion if num_pion > 0 else 0
                #print(f"Threshold: {threshold:.2f} - Classified: {num_classified} - True: {num_true} - Efficiency: {efficiency:.2f} - Accuracy: {accuracy:.2f}")

                efficiencies.append(efficiency)
                accuracies.append(accuracy)
                accuracies_proton.append(accuracy_proton)
                efficiencies_proton.append(efficiency_proton)
                accuracies_pion.append(accuracy_pion)
                efficiencies_pion.append(efficiency_pion)

            accuracies = np.array(accuracies)
            efficiencies = np.array(efficiencies)

            accuracies_proton = np.array(accuracies_proton)
            efficiencies_proton = np.array(efficiencies_proton)

            accuracies_pion = np.array(accuracies_pion)
            efficiencies_pion = np.array(efficiencies_pion)



            x=thresholds
            f_acc = interp1d(x, accuracies, kind='linear', fill_value="extrapolate")
            f_eff = interp1d(x, efficiencies, kind='linear', fill_value="extrapolate")

            f_acc_proton = interp1d(x, accuracies_proton, kind='linear', fill_value="extrapolate")
            f_eff_proton = interp1d(x, efficiencies_proton, kind='linear', fill_value="extrapolate")
            f_acc_pion = interp1d(x, accuracies_pion, kind='linear', fill_value="extrapolate")


            # Function to find the difference between interpolated values
            def diff(x):
                return f_acc(x) - f_eff(x)
            def diff_proton(x):
                return f_acc_proton(x) - f_eff_proton(x)
            def diff_pion(x):
                return f_acc_pion(x) - f_eff(x)
            # Find intersection points
            x_common = fsolve(diff, x)  # Use x as initial guesses
            x_common = np.unique(x_common[(x_common >= x[0]) & (x_common <= x[-1])])  # Filter valid points
            y_common = f_acc(x_common)  # Get y values from either function

            x_common_proton = fsolve(diff_proton, x)  # Use x as initial guesses
            x_common_proton = np.unique(x_common_proton[(x_common_proton >= x[0]) & (x_common_proton <= x[-1])])  # Filter valid points
            y_common_proton = f_acc_proton(x_common_proton)  # Get y values from either function

            x_common_pion = fsolve(diff_pion, x)  # Use x as initial guesses
            x_common_pion = np.unique(x_common_pion[(x_common_pion >= x[0]) & (x_common_pion <= x[-1])])  # Filter valid points
            y_common_pion = f_acc_pion(x_common_pion)  # Get y values from either function


            fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

            # General Accuracy and Efficiency Plot
            axes[0].plot(x*100, accuracies*100, label=f'Accuracies={round(y_common[0]*100,2)}%', marker='o', markersize=2)
            axes[0].plot(x*100, efficiencies*100, label=f'Efficiencies={round(y_common[0]*100,2)}%', marker='o', markersize=2)
            axes[0].scatter(x_common[0]*100, y_common[0]*100, color='red', label='Intersections', marker='o', s=100)
            if len(x_common) > 0:
                first_x, first_y = x_common[0]*100, y_common[0]*100
                axes[0].annotate(f"({first_x:.3f}, {first_y:.3f})",
                                (first_x, first_y),
                                textcoords="offset points",
                                xytext=(10, -10),
                                arrowprops=dict(arrowstyle="->", color='gray'))
            axes[0].legend()
            axes[0].set_xlabel('Threshold (%)', fontsize=12)
            axes[0].set_ylabel('Value (%)', fontsize=12)
            axes[0].set_title('Accuracy and Efficiency Tradeoff (General)', fontsize=14)
            axes[0].grid()
            axes[0].tick_params(axis='both', labelsize=12)

            # Proton Accuracy and Efficiency Plot
            axes[1].plot(x*100, accuracies_proton*100, label=f'Proton Accuracies={round(y_common_proton[0]*100,2)}%', marker='o', markersize=2)
            axes[1].plot(x*100, efficiencies*100, label=f'Efficiencies={round(y_common_proton[0]*100,2)}%', marker='o', markersize=2)
            axes[1].scatter(x_common_proton[0]*100, y_common_proton[0]*100, color='red', label='Intersections', marker='o', s=100)
            if len(x_common_proton) > 0:
                first_x, first_y = x_common_proton[0]*100, y_common_proton[0]*100
                axes[1].annotate(f"({first_x:.3f}, {first_y:.3f})",
                                (first_x, first_y),
                                textcoords="offset points",
                                xytext=(10, -10),
                                arrowprops=dict(arrowstyle="->", color='gray'))
            axes[1].legend()
            axes[1].set_xlabel('Threshold (%)', fontsize=12)
            axes[1].set_ylabel('Value (%)', fontsize=12)
            axes[1].set_title('Accuracy and Efficiency Tradeoff (Proton)', fontsize=14)
            axes[1].grid()
            axes[1].tick_params(axis='both', labelsize=12)

            # Pion Accuracy and Efficiency Plot
            axes[2].plot(x*100, accuracies_pion*100, label=f'Pion Accuracies={round(y_common_pion[0]*100,2)}%', marker='o', markersize=2)
            axes[2].plot(x*100, efficiencies*100, label=f'Efficiencies={round(y_common_pion[0]*100,2)}%', marker='o', markersize=2)
            axes[2].scatter(x_common_pion[0]*100, y_common_pion[0]*100, color='red', label='Intersections', marker='o', s=100)
            if len(x_common_pion) > 0:
                first_x, first_y = x_common_pion[0]*100, y_common_pion[0]*100
                axes[2].annotate(f"({first_x:.3f}, {first_y:.3f})",
                                (first_x, first_y),
                                textcoords="offset points",
                                xytext=(10, -10),
                                arrowprops=dict(arrowstyle="->", color='gray'))
            axes[2].legend()
            axes[2].set_xlabel('Threshold (%)', fontsize=12)
            axes[2].set_ylabel('Value (%)', fontsize=12)
            axes[2].set_title('Accuracy and Efficiency Tradeoff (Pion)', fontsize=14)
            axes[2].grid()
            axes[2].tick_params(axis='both', labelsize=12)

            
            plt.savefig(f"/home/alma1/GNN/Deepset/28oct_DNN/DNN_plots/accuracy_vs_efficiency_{name}.pdf")
                
                

            


