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
from tqdm.auto import tqdm

tqdm(disable=True)

#data_set name with accuracy dict
dict_acc={}

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
        

directory = "/home/alma1/GNN/Deepset/28oct_DNN/csv/"

for root,_,files in os.walk(directory):
    length=len(files)
    for filename in files:
        if filename.endswith(".csv"):
            csv_file = os.path.join(root, filename)
            
            length-=1
            
            #f lis a list of features to be used in the model (you can add or reove time0 here)
            for case in ["with_time", "without_time"]:
                name=csv_file.split("/")[-1].split(".")[0]
                f=['speed',  
            'AsymmetryY',
            'centralTowerFraction_cell',
            'AsymmetryX',
            'VarianceAtVertex',
            'weightedTime',
            'AsymmetryX_plain',
            'EfractionCell',
            'AsymmetryY_plain',
            'TotalEnergy',
            'numCellBeforeVertex',
            'radialSigma',
            'Asymmetry',
            'RatioEcell',
            'PostVertexEnergyFraction',
            'Asymmetry_plain',
            'longitudinalSigma',
            'longitudinalSigma_plain',
            'radius',
            'DeltaEcell_secondMax',
            'NumberOfUniqueCells',
            'TotalEnergyCloseToVertex',
            'radialSigma_plain',
            'length',
            'theta2',
            'Z_vertex',
            'd2',
            'MaxEnergyInCell',
            'MaxEnergyCloseVertex',
            'E2',
            'distanceMaxFromVertex',
            'EnergyFractionCloseToVertex',
            'radius_plain',
            'E1',
            'length_plain',
            'SecondMaxEnergyInCell',
            'R1',
            'VertexTime',
            'Aplanarity',
            'R2',
            'distanceFirstSecondMaxEnergy',
            'DeltaT',
            'E3',
            'theta3',
            'd3',
            'R3',
            'NumPeaks']
                if case=="with_time":
                    f=["time0"]+f
                else:
                    f=f

                if 'time0' in f:
                    name=name+"_time"
                else:
                    name=name+"_notime"
                
                print(name,"reamaing files:",length)
                train_loader,valid_loader,test_loader = create_dataloader(csv_file, batch_size=256,num=28,features=f)
                for X_batch, y_batch in train_loader:
                    print(X_batch, len(y_batch))
                    print(X_batch.shape)
                    print(y_batch.shape)
                    input_size = X_batch.shape[1]
                    break
                hidden_layers = [96,32,16]
                model = SimpleDNN(input_size, hidden_layers) # I also tried to use weights=weights in the model but it did not work
                model.to(device)
                torch.save(model.state_dict(), "model_hcal.pt")

                

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
                n_epochs = 5
                loss_data = []

                valid_loss_min = np.inf  # set initial "min" to infinity

                for epoch in range(n_epochs):
                    model_test = SimpleDNN(input_size, hidden_layers)
                    model_test.to(device)
                    model_test.load_state_dict(torch.load('model_hcal.pt',weights_only=True))  
                
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
                        torch.save(model.state_dict(), 'model_hcal.pt')
                        valid_loss_min = valid_loss
                    loss_history_df = pd.DataFrame(loss_data)
                    loss_history_df.to_csv('loss_history.csv', index=False)

                # %%
                model_test = SimpleDNN(input_size, hidden_layers)
                model_test.to(device)
                model_test.load_state_dict(torch.load('model_hcal.pt',weights_only=True))


                

                # %%
                # eval_model(model_test, test_loader)
                dict_acc[name]=eval_model(model_test, test_loader)
                dff=pd.DataFrame(dict_acc.items(),columns=["Name","Accuracy"])
                dff.to_csv("accuracy.csv",index=False)
                

                # %%
                from sklearn.metrics import confusion_matrix
                import seaborn as sns
                import matplotlib.pyplot as plt
                from matplotlib.colors import LinearSegmentedColormap

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
                    
                    print(f"cm values: {cm}")
                    plt.xticks(size=14, weight='bold')
                    plt.yticks(size=14, weight='bold')
                    plt.title('Confusion Matrix', size=15, weight='bold')
                    plt.xlabel('Predicted', size=14, weight='bold')
                    plt.ylabel('True', size=14, weight='bold')
                    plt.title('Confusion Matrix',weight='bold',size=15)
                    
                    plt.savefig(f'/home/alma1/GNN/Deepset/28oct_DNN/DNN_plots/confusion_matrix_{name}.png')
                    plt.savefig(f'/home/alma1/GNN/Deepset/28oct_DNN/DNN_plots/confusion_matrix_{name}.pdf')
                    

                plot_confusion_matrix(model_test, test_loader)


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
                plt.savefig(f"/home/alma1/GNN/Deepset/28oct_DNN/DNN_plots/confidence_{name}.pdf")

                plt.hist(confidence[true_labels != predicted_labels]*100., bins=50, color='red', alpha=0.5, label='False_prediction',density=True,histtype='stepfilled')
                plt.axvline(x=conf_median*100, color='red', linestyle='--', label=f'Median Confidence = {conf_median*100:.2f}%')
                plt.xlabel("Confidence (%)",fontsize=12)
                plt.ylabel("False Predictions",fontsize=12)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(f"Accuracy of the model:{acc} ",fontsize=12)
                plt.title("Confidence for Binary Classification")
                plt.legend()
                plt.savefig(f"/home/alma1/GNN/Deepset/28oct_DNN/DNN_plots/confidence_{name}.pdf")