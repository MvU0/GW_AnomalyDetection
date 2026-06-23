# Import necessary packages
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as GAT_DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
import random


# Necessary for GAT
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.nn import BatchNorm, InstanceNorm, LayerNorm, GraphNorm, PairNorm, DiffGroupNorm
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_scatter import scatter
from torch_geometric.utils import softmax as graph_softmax

import optuna
from pathlib import Path     # Needed for checking paths for saving plots


# Maybe necessary
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Just for checking
import time
import sys
from prettytable import PrettyTable

#=========================================================================#
# CODE PACKAGE USABLE FOR CHECKING IF AUXILIARY CHANNEL DATASET #
# CONTAINS ENOUGH INFORMATION FOR CLASS SEPARATION (USING CNN/MLP or GAT) #
#=========================================================================#

"""
CNN/MLP:
    Mandatory input :
        data - shape (N_images, channels, 'time'-steps) - The data that needs to be classified - MUST be Numpy array or Pandas DataFrame
        labels - shape (N_images,) - The corresponding labels to the data - Labels need to be integers from 0 to n_classes-1
        model_type - Either 'CNN', 'MLP' - The type of model used to classify glitches

    Optional:
        balance_classes - boolean 'True'/'False' - Default is 'True', makes train/val/test dataset all have (approximately) the same amount of each class
        n_epochs - int - Default is 10, amount of epochs training will take
        lr - float - Learning rate, default is 1e-3
        train_frac - float - Fraction of data to use in training, default is 0.7
        val_frac- float - Fraction of data to use for validation, default is 0.1
        patience - int - Number of epochs where val-loss may not improve before stopping the run 
        seed - int - seed for randomness, default is 15

    Non-changable parameters:
        optimizer - Default is 'Adam'
        batch_size - Default is 32, could change in this code

    Returns:
        Accuracy
        (True_labels, Predicted_labels)
        (train_loss, val_loss)
        lr_list
        val_acc_list
        
    Outputs:
        Running the code prints:
            The amount of samples in train/val/test sets.
            Training loss per epoch.
            Validation loss and validation accuracy per epoch.
            Test accuracy

GAT:
    Mandatory input :
        data - shape (N_images, channels, 'time'-steps) - The data that needs to be classified - MUST be Numpy array or Pandas DataFrame
        labels - shape (N_images,) - The corresponding labels to the data - Labels need to be integers from 0 to n_classes-1
    
    Optional input:
        balance_classes - boolean 'True'/'False' - Default is 'True', makes train/val/test dataset all have (approximately) the same amount of each class
        n_epochs - int - Default is 100, amount of epochs training will take
        lr - float - Learning rate, default is 1e-3
        train_frac - float - Fraction of data to use in training, default is 0.7
        val_frac - float - Fraction of data to use for validation, default is 0.1 
        topk - int - Default is 10 - amount of incoming edges per channel
        similarity_type - str - Default is 'cosine' - other option 'pearson' - defines which measure to use for similarity
        mode - str - Default is 'mean' - other options 'single' and 'concat' - defines how to construct similarity ;
                            'mean': take mean over all images and then calculate similarity between channels
                            'concat': concatenate all images along time axis and then calculate similarity between channels
                            'single': calculate similarities separately for each image - get unique edges for each image
        seed - int - Default is 15 - used for random splitting of data
        optim - 'str' - Default is Adam, which optimizer to use in training
        save_imgs - boolean - Default is False, determines if loss plot and confusion matrix need to be saved
        save_path - str - can be full or relative path to folder for images and model - default is './saveables'
        num_layers - int - Integer which says how many GAT layers to use - Each layer goes from: dimension_in --> dimension_in//2 - Default is 1
        dropout - float - Value to use for dropout in model - Default is 0.5,
        patience - int - Amount of times in a row the validation loss is allowed to not reach a new minimum before cancelling the training - Default is 100
        weight_decay - float - Value for weight decay - Default is 5e-4
        pool_type - string - Type of pooling to apply to reduce graph dimension - Default is 'mean' other options: 'max', 'att', 
        T_max - int - Period (in # epochs) of cosine annealing of learning rate - Default is 100
        eta_min - float - Minimum learning rate in cosine annealing - default is 1e-5
        batch_size - Default is 64 for train/val, 1 for test

    Non-changable parameters:
        --

    Returns:
        Accuracy
        (True_labels, Predicted_labels)
        (train_loss, val_loss)
        lr
        (vall_acc_list, train_acc_list)
        importance_metrics = (edge_indices_list, att_weights_list, feature_gates_list, node_gates_list, lin_weight) -- feature_gate_list,node_gates_list are None unless attention pooling is used

    Outputs:
        Running the code prints:
            The amount of samples in train/val/test sets.
            Training loss per epoch.
            Validation loss and validation accuracy per epoch.
            Test accuracy
        Saves:
            Model with lowest validation loss
            Plot of test and validation loss
            Confusion matrix
"""




#-------------------------------------------------------------------#
# First define all necessary classes and functions for all methods  #
# These are all functions/classes to be called in the main function #
#-------------------------------------------------------------------#
def seed_everything(seed=15):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic CUDA behavior

    torch.backends.cudnn.deterministic = False

    torch.backends.cudnn.benchmark = False

def class_balance(data, labels):
    """  
    Function to ballance classes

    Args:
        data - shape (N_classes, channels, time_steps)
        labels - shape (N_classes,)

    Returns balanced_data and balanced_labels
    """
    # Check num_classes and amount in smallest class
    unique, counts = np.unique(labels, return_counts = True) 
    
    num_classes = len(unique)

    least_common_class = np.argmin(counts) 
    least_common_count = counts[least_common_class]

    # Generate one list/array containing exaclty num_classes * least_common_count indices which point to exactly least_common_class images per class
    selected_indices_all = []
    for class_label in unique:
        # Indices of this class in the original data
        class_indices = np.where(np.array(labels) == class_label)[0]

        # Randomly choose least_common_count indices
        selected_indices = np.random.choice(class_indices, size = least_common_count, replace = False)
        selected_indices_all.append(selected_indices)

    selected_indices_all = np.concatenate(selected_indices_all)
    balanced_data = data[selected_indices_all]
    balanced_labels = labels[selected_indices_all]

    return balanced_data, balanced_labels

def train_val_test(data, labels, train_frac, val_frac, model_type, balance_classes = True, seed = 15):
    """
    Function to split data into training/validation/test sets,
    and put into correct dataloaders using custom (but quite standard) dataset structure

    Args:
        data: Data to be used, can be lists, numpy arrays, scipy-sparse matrices or pandas dataframes
        labels: labels corresponding to the data, can be a list or a 1D-array
        train_frac: fraction of full data to be used as training data - default is 0.7 (see the main function)
        val_frac: fraction of full data to be used as validation data - default is 0.1 (see the main function)
        balance_classes: boolean for balancing classes - default is True
        seed: seed for random splitting
    
    Returns:
        Train-, validation- and test-dataloaders
    """
    
    if balance_classes:
        data, labels = class_balance(data, labels)
    

    # Step 1 - make train/val/test split
    #   1.1 make train/val+test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels,
        test_size = (1-train_frac),
        random_state = seed,
        stratify = labels
    )
    # Normalize the data n  # UNIQUE FOR THIS MODEL
    mean = np.mean(X_train, axis=(0,2), keepdims=True)
    std = np.std(X_train, axis=(0,2), keepdims=True)
    std[std == 0 ] = 1e-8

    X_train = (X_train - mean)/std
    X_temp = (X_temp - mean)/std


    test_size = (1 - val_frac/(1-train_frac))  # calculate the fractionsize of test data on leftover (temp) part of data 
    #   1.2 - make val / test split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size = test_size,
        random_state = seed,
        stratify = y_temp
    )


    

    # Put everything in torch tensors

    if model_type == 'CNN':
        X_train = torch.tensor(X_train).unsqueeze(1)
        X_val = torch.tensor(X_val).unsqueeze(1)
        X_test = torch.tensor(X_test).unsqueeze(1)
    elif model_type == 'MLP':
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)



    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)
    y_test = torch.tensor(y_test)

    print("Train samples:", X_train.shape[0])
    print("Val samples:", X_val.shape[0])
    print("Test samples:", X_test.shape[0])

    # Step 2 - Create dataloaders
    #   2.1 - Create custom train/val/test datasets
    dataset_train = CustomDataset(X_train, y_train)
    dataset_val = CustomDataset(X_val, y_val)
    dataset_test = CustomDataset(X_test, y_test)

    #   2.2 - Turn into DataLoaders
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

    return dataloader_train, dataloader_val, dataloader_test

class CustomDataset(Dataset):
    """
    Custom dataset for pairing input data with corresponding labels
    
    Dataset wraps feature data 'X' adn target labels 'labels' such that they can be accessed byindex during training or evaluation.
    It implements the required '__len__' and '__getitem__' methodsexpected by PyTorch DataLoader
    
    Args:
        X (array-like or Tensor): Input features with shape (N_images, ...) 
        labels (array-like or Tensor): Target labels with length N

    Returns: 
        tuple: (X[idx], label[idx]) for a given sample index

    """
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]
    
def train(model, scheduler, train_loader, val_loader, num_epochs, optimizer, num_classes, patience, device):
    """  
    Function for training and validation of a CNN/MLP classifier model

    Args:
        model: this is a predefined model either CNN or MLP
        scheduler: predefined learning rate scheduler
        train_loader: dataloader containing training data, must be a PyTorch DataLoader
        val_loader: dataloader containing validation data, must be a PyTorch DataLoader
        num_epochs: number of epochs that the model is trained, default is 10 
        optimizer: which optimizer to use when updating parameters - default is Adam
        num_classes: amount of unique classes in the dataset
        patience: amount of times validation loss can not improve before stopping training
        device: which device to use, either CUDA or CPU, if CUDA it uses GPU so faster

    Return:
        training- and validation loss per epoch
    
    """
    if num_classes == 2:
       criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    lr_list = []

    # Define minimum loss and improve count for premature stopping
    min_loss = 1e+8
    stop_improve_count = 0

    for epoch in range(num_epochs):
        # ---------------
        # TRAINING
        # ---------------
        model.train()
        running_loss = 0.0
        lr_list.append(optimizer.param_groups[0]['lr'])

        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

            if num_classes == 2:
                labels = labels.float()  # ensure float labels for BCE
                out = model(images).view(-1) # view(-1) ensures output has right shape (N,)
                loss = criterion(out, labels)
            
            else:
                labels = labels.long()  # ensure long labels for CE
                out = model(images)
                loss = criterion(out, labels)
            
            loss.backward()  # do backward propagation
            optimizer.step()
            running_loss += loss.item()

        # ---------------
        # VALIDATION
        # ---------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                if num_classes == 2:
                    labels = labels.float()  # ensure float labels for BCE
                    out = model(images).view(-1)  # view(-1) ensures output has right shape (N,)

                    loss = criterion(out, labels)
                    val_loss += loss.item()

                    preds = torch.sigmoid(out) > 0.5  # predicted labels
                    correct += (preds.cpu() == labels.cpu().bool()).sum().item()
                    

                else:
                    labels = labels.long()  # ensure long labels for CE
                    out = model(images)

                    loss = criterion(out, labels)
                    val_loss += loss.item()

                    preds = out.argmax(dim=1)  # predicted labels
                    correct += (preds.cpu() == labels.cpu()).sum().item()

                total += labels.size(0)    
            if val_loss < min_loss:
                torch.save(model.state_dict(), '/data/gravwav/mvuden/anomaly-detection-thesis-mees-van-uden/New_GNN_start/_Dataset_classifier/saveables/CNN/Model/model.pt')
                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1
                if stop_improve_count >= patience:
                    break
            scheduler.step()
            #scheduler.step(val_loss)
            
                

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {running_loss/len(train_loader):.4f} | " 
              f"Val Loss: {val_loss/len(val_loader):.4f} Val Acc: {acc:.2f}%")
        
        val_acc_list.append(acc)
        train_loss_list.append(running_loss/len(train_loader))
        val_loss_list.append(val_loss/len(val_loader))
    

    print("Training finished")
    return(train_loss_list, val_loss_list), lr_list, val_acc_list

def test(model, test_loader, num_classes, device):
    """  
    Function for testing the trained CNN/MLP glitch classifier

    Args:
        model: predefined + pretrained classifying model
        test_loader: dataloader containing test data, must be a PyTorch DataLoader
        num_classes: amount of unique classes in the data
        device: which device to use (GPU(CUDA)/CPU)
    Returns:
        accuracy, true_labels, predicted_labels, probabilities
    
    """
    model.load_state_dict(torch.load('/data/gravwav/mvuden/anomaly-detection-thesis-mees-van-uden/New_GNN_start/_Dataset_classifier/saveables/CNN/Model/model.pt'))  # Load the best model
    model = model.to(device)
    model.eval()  # set model to evaluation mode to prevent weight updates

    preds_list = []
    labels_list = []
    probs_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            if num_classes == 2:
                probs = torch.sigmoid(out).view(-1)  # view(-1) ensures output has right shape (N,)
                preds = (probs > 0.5).long()
            else:
                probs = torch.softmax(out, dim=1)
                preds = out.argmax(dim=1)

            preds_list.append(preds.cpu())
            labels_list.append(labels.cpu())
            probs_list.append(probs.cpu())

    pred_labels = torch.cat(preds_list)
    true_labels = torch.cat(labels_list)
    probs = torch.cat(probs_list)

    accuracy = (pred_labels == true_labels).float().mean().item() * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy, true_labels, pred_labels, probs

#--------------------------------------#
# Defining functions necessary for GAT #
#--------------------------------------#

def construct_adjacancy(data, topk, similarity_type, mode):
    """   
    Function to calculate adjacancy matrix/matrices for images

    Args:
        data: Data to be used
        topk: Amount of incoming edges per node - default is 10
        similarity_type: the measure of similarity between channels - either 'pearson' or 'cosine' - default is cosine
        mode: way of constructing similarity, either construct same adjacancy for all images or a unique adjacancy matrix for each image
            - 'single': construct adjacancy per image
            - 'mean' : take mean over all images and calculate similarity across channels
            - 'concat' : concatenate all images along time axis and calculate similarity across channels

    Returns:
        Edge_indices (as torch tensors)
    
    """

    #==================#
    # helper functions #
    #==================#

    # Set similarity/correlation function
    def compute_similarity(x): # x is data
        if similarity_type == "pearson":
            return abs(np.corrcoef(x))
        else: # cosine
            return cosine_similarity(x)
        
    # Build adjacancy matrix + edge index
    def build_graph(sim): # sim is similarity measure
        num_nodes = sim.shape[0]

        # top-k incomed edges per node (column-wise)
        topk_idx = np.argsort(sim, axis=0)[-topk:, :]

        adj = np.zeros_like(sim, dtype=bool)
        cols = np.arange(num_nodes)
        adj[topk_idx, cols] = True  # Here this topk_idx, cols uses advanced indexing - cols is broadcasted to the same shape as topk_idx

        edge_indices = np.vstack(np.where(adj))

        return torch.tensor(edge_indices)
    
    # Single graph per sample 
    if mode == 'single':
        edge_indices_list = []

        for i in range(len(data)):
            sim = compute_similarity(data[i])
            edge_idx = build_graph(sim)

            edge_indices_list.append(edge_idx)

        return edge_indices_list
    
    if mode == "mean":
        data_agg = np.mean(data, axis=0)
    else: # concat
        data_agg = data.transpose(1,0,2).reshape(
            data.shape[1], data.shape[0]*data.shape[2])
        
    sim = compute_similarity(data_agg)
    
    return build_graph(sim)
        
def gat_dataloader(data, labels, edge_indices, batch_size, shuffle):
    """  
    Function for constructing dataloader, where input data is paired with corresponding edge indices and labels

    Args:
        data: data to be used - shape [num_samples, num_nodes, num_features]
        labels: labels corresponding to data - shape [num_samples,]
        edge_indices: edges that define the graph structure - shape [N_edges, 2]
        batch_size: size of samples used before updating weights
        shuffle: boolean - used to say if dataloader needs to shuffle its data

    Returns:
        Train, validation and test dataloaders
    """
    graphs = [
            Data(
                x=torch.as_tensor(data[i], dtype=torch.float),
                edge_index=edge_indices[i],
                y=torch.as_tensor(labels[i], dtype=torch.long).view(1),
            )
            for i in range(len(data))
        ]

    return GAT_DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)

def train_val_test_gat(data, labels, train_frac, val_frac, balance_classes, topk, similarity_type, batch_size, mode, seed = 15, preprocess=True):
    """
    Function to split data into training/validation/test sets for graph attention network,
    and put into correct dataloaders using custom (but quite standard) dataset structure

    Args:
        data: Data to be used, can be lists, numpy arrays, scipy-sparse matrices or pandas dataframes
        labels: labels corresponding to the data, can be a list or a 1D-array
        train_frac: fraction of full data to be used as training data - default is 0.7 (see the main function)
        val_frac: fraction of full data to be used as validation data - default is 0.1 (see the main function)
        balance_classes: boolean for balancing classes - default is True
        seed: seed for random splitting
        batch_size: batch_size for train/val dataloaders - default is 64
        similarity_type: the measure of similarity between channels - either 'pearson' or 'cosine' - default is cosine
        mode: way of constructing similarity, either construct same adjacancy for all images or a unique adjacancy matrix for each image
            - 'single': construct adjacancy per image
            - 'mean' : take mean over all images and calculate similarity across channels
            - 'concat' : concatenate all images along time axis and calculate similarity across channels
    
    Returns:
        Train-, validation- and test-dataloaders
    """

    if balance_classes == True:
        data, labels = class_balance(data, labels)
    

    # Step 1 - make train/val/test split
    #   1.1 make train/val+test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels,
        test_size = (1-train_frac),
        random_state = seed,
        stratify = labels
    )

    if preprocess:
        # Normalize the data # UNIQUE FOR THIS MODEL
        mean = np.mean(X_train, axis=(0,2), keepdims=True)
        std = np.std(X_train, axis=(0,2), keepdims=True)
        std[std == 0] = 1e-8

        X_train = (X_train - mean)/std
        X_temp = (X_temp - mean)/std
    
    

    test_size = (1 - val_frac/(1-train_frac))  # calculate the fractionsize of test data on leftover (temp) part of data 
    #   1.2 - make val / test split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size = test_size,
        random_state = seed,
        stratify = y_temp
    )

    print("Train samples:", X_train.shape[0])
    print("Val samples:", X_val.shape[0])
    print("Test samples:", X_test.shape[0])

    # Step 2 - Construct edge_indices
    if mode == 'single':
        edge_indices_train = construct_adjacancy(data=X_train, topk=topk, mode=mode, similarity_type=similarity_type) 
        edge_indices_val = construct_adjacancy(data=X_val, topk=topk, mode=mode, similarity_type=similarity_type) 
        edge_indices_test = construct_adjacancy(data=X_test, topk=topk, mode=mode, similarity_type=similarity_type) 
    else: # mean or concat
        edge_index = construct_adjacancy(data=X_train, topk=topk, mode=mode, similarity_type=similarity_type) 
        edge_indices_train = [edge_index.clone() for _ in range(X_train.shape[0])] # Make it such that every image has their own set of edge_indices
        edge_indices_val = [edge_index.clone() for _ in range(X_val.shape[0])]
        edge_indices_test = [edge_index.clone() for _ in range(X_test.shape[0])]

    # Step 3 - Create dataloaders
    dataloader_train = gat_dataloader(X_train, y_train, edge_indices_train, batch_size = batch_size, shuffle=True)
    dataloader_val = gat_dataloader(X_val, y_val, edge_indices_val, batch_size=batch_size, shuffle=True)
    dataloader_test = gat_dataloader(X_test, y_test, edge_indices_test, batch_size=1, shuffle=False)
    

    return dataloader_train, dataloader_val, dataloader_test

def GAT_train(model, scheduler, train_loader, val_loader, num_epochs, optimizer, num_classes, cfg, patience, model_save_path, device, trial):
    """  
    Function for training and validation of a GAT classifier model

    Args:
        model: pre-initialized GAT-model
        scheduler: learning rate scheduler
        train_loader: dataloader containing training data, must be a PyTorch DataLoader
        val_loader: dataloader containing validation data, must be a PyTorch DataLoader
        num_epochs: number of epochs that the model is trained, default is 10 
        optimizer: which optimizer to use when updating parameters - default is Adam
        num_classes: amount of unique classes in the dataset
        cfg: configuration of model, containing n_epochs, topk, lr, train_frac, val_frac, mode, similarity_typ, optimizer
        patience: amount of times val loss is allowed to not reach a new minimum before stopping training, default is 15
        save_path: path to save file
        device: which device to use - either CPU or GPU
        trial: which trial you are one - used for pruning
        

    Return:
        training- and validation loss per epoch
    
    """
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    lr_list = []

    # Define minimum loss and improve count for premature stopping
    min_loss = 1e+8
    stop_improve_count = 0


    for epoch in range(num_epochs):
        # -------------
        # TRAINING
        # -------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        lr_list.append(optimizer.param_groups[0]['lr'])

        for data in train_loader:
            # data.x: [batch_nodes, num_features]
            # data.edge_index: edges for the batched graph
            # data.y: [batch_nodes] (node-level labels)
            data = data.to(device)
            optimizer.zero_grad()

            if num_classes == 2:
                labels = data.y.float() # ensure float labels for BCE
                out = model(data.x, data.edge_index, data.batch).view(-1)
                loss = criterion(out, labels)

                preds = torch.sigmoid(out) > 0.5 # predicted labels
                correct_train += (preds.cpu() == labels.cpu().bool()).sum().item()

            else:
                labels = data.y.long() # ensure long labels for CE
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, labels)

                preds = out.argmax(dim=1) # predicted labels
                correct_train += (preds.cpu() == labels.cpu()).sum().item()
            
            total_train += labels.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ---------------
        # VALIDATION
        # ---------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                if num_classes == 2:
                    labels = data.y.float() # ensure float labels for BCE
                    out = model(data.x, data.edge_index, data.batch).view(-1)

                    loss = criterion(out, labels)
                    val_loss += loss.item()

                    preds = torch.sigmoid(out) > 0.5 # predicted labels
                    correct += (preds.cpu() == labels.cpu().bool()).sum().item()

                else:
                    labels = data.y.long() # ensure long labels for CE
                    out = model(data.x, data.edge_index, data.batch)

                    loss = criterion(out, labels)
                    val_loss += loss.item()

                    preds = out.argmax(dim=1) # predicted labels
                    correct += (preds.cpu() == labels.cpu()).sum().item()

                total += labels.size(0)

            if trial is not None:
                trial.report(val_loss, epoch)

                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            if val_loss < min_loss:
                torch.save(model.state_dict(), model_save_path / f"{cfg['cfg_path']}.pt")
                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1
                if stop_improve_count >= patience:
                    break
            #scheduler.step(val_loss)  
            scheduler.step()

        acc = 100 * correct / total
        acc_train = 100 * correct_train / total_train
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {running_loss/len(train_loader):.4f} Train acc: {acc_train:.2f}% | " 
              f"Val Loss: {val_loss/len(val_loader):.4f} Val Acc: {acc:.2f}%")

        train_acc_list.append(acc_train)
        val_acc_list.append(acc)
        train_loss_list.append(running_loss/len(train_loader))
        val_loss_list.append(val_loss/len(val_loader))

    print("Training finished")
    return(train_loss_list, val_loss_list), lr_list, val_acc_list, train_acc_list

def GAT_test(model, test_loader, num_classes, cfg, model_save_path, device):
    """  
    Function for testing the trained CNN/MLP glitch classifier

    Args:
        model: predefined + pretrained classifying model
        test_loader: dataloader containing test data, must be a PyTorch DataLoader
        num_classes: amount of unique classes in the data
        cfg: configuration of model, containing n_epochs, topk, lr, train_frac, val_frac, mode, similarity_typ, optimizer
        save_path: path to save file
        device: which device to use - either CPU or GPU

    Returns:
        accuracy, true_labels, predicted_labels, probabilities
    
    """

    model.load_state_dict(torch.load(model_save_path / f"{cfg['cfg_path']}.pt", map_location=device))  # Load the best model
    model.to(device)
    model.eval()  # set model to evaluation mode to prevent weight updates

    preds_list = []
    labels_list = []
    
    # Also initialize lists of edge_indices, attn_weights, feature_gate and node_gates
    edge_indx_list = []
    att_weights_list = []
    feature_gates_list = []
    node_gates_list = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out, (edge_indx, att_weights, feature_gates, node_gates, lin_weight) = model(data.x, data.edge_index, data.batch, return_attention=True)
            if num_classes == 2:
                out = out.view(-1)  # view(-1) ensures output has right shape (N,)
                preds = (torch.sigmoid(out)>0.5).long()

            else:
                preds = out.argmax(dim=1)

            preds_list.append(preds.cpu())
            labels_list.append(data.y.cpu())
            
            edge_indx_list.append(edge_indx.cpu())
            att_weights_list.append(att_weights.cpu())
            if feature_gates is not None:
                feature_gates_list.append(feature_gates.cpu())
                node_gates_list.append(node_gates.cpu())

    pred_labels = torch.cat(preds_list)
    true_labels = torch.cat(labels_list)

    accuracy = (pred_labels == true_labels).float().mean().item() * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy, true_labels, pred_labels, (edge_indx_list, att_weights_list, feature_gates_list, node_gates_list, lin_weight)

#------------------------------#
# Defining different models    #
#------------------------------#

class Conv2DNet(nn.Module):
    def __init__(self, image_height, image_width, num_classes, input_channels=1):
        """
        Convolutional neural network class for classifying

        Args:
            input_channels: number of channels in input images (1 for grayscale)
            image_height: height of input images
            image_width: width of input images
            num_classes: number of output classes
        
        Returns:
            Either 1 values per image (for 2 class classification) or n values per image (for n class classification; n > 2)
        """
        super(Conv2DNet, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=(3,2), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,2), padding=(1,1))

        # Pooling layer (2x2 max pooling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

        # Activation
        self.relu = nn.ReLU()

        # Compute flattened size dynamically
        self.flattened_size = self._get_flattened_size(input_channels, image_height, image_width)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, 32)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=0.1)
        # Output layer
        if num_classes == 2:
            self.out = nn.Linear(32, 1) # if 2 classes use binary cross entropy loss - need only one value as output
        else:
            self.out = nn.Linear(32, num_classes) # otherwise use regular cross entropy and need as many values as output as there are unique classes
    
    def _get_flattened_size(self, input_channels, H, W):
        """Forward pass through conv layers with dummy tensor to compute flattened size"""
        x = torch.zeros(1, input_channels, H, W)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x.numel()  # total number of elements after flattening
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        Multilayer perceptron neural network class for classifying

        Args:
            input_dim: flattened size of one image, n_channels*n_time_steps
            num_classes: number of unqiue classes
        
        Returns:
            Either 1 values per image (for 2 class classification) or n values per image (for n class classification; n > 2)
        """
        super().__init__()

        if num_classes == 2:        # Define number of output values such that for 2 classes one can use Binary Cross Entropy loss
            out_num = 1
        else:
            out_num = num_classes
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16, out_num)  # <-- dynamic output
        )
    
    def forward(self, x):
        return self.layers(x)

class AttentionPooling(nn.Module):
    """
        Attention Pooling layer used to either do feature level attention pooling or combined gating (both node level and feature level gating)"
        
        Computes: y = sum_i \alpha_i * (g_i \concat x_i)
        where: 
            g_i = feature gate (per-node, per-feature)
            \alpha_i = node attention weight (per-node scalar)
        Args: 
            gate_nn: Neural net to use when constructing feature-level attention weights
            nn: optional neural net to use when constructing node-level attention weights
        """
    def __init__(self, feature_gate: nn.Module, node_gate: nn.Module = None):
        super().__init__()
        
        self.feature_gate = feature_gate
        self.node_gate = None
        if node_gate is not None:
            self.node_gate = node_gate


    def reset_parameters(self):
        reset(self.feature_gate)
        reset(self.node_gate)

    def forward(self, x, batch) -> torch.Tensor:
        """
        Args:
            x: Node features [N, F]
            batch: Batch vector [N]
        Returns:
            pooled: graph-level embedding [B, F]
            alpha: Node attention weights [N, 1]
            gates: Feature-level weights [N, F]
        
        """
        # --- Feature-level gating ---
        gates = self.feature_gate(x)        # [N, F]
        x_gated = x * gates                 # [N, F]

        # --- Node-level gating ---
        if self.node_gate is not None:
            scores = self.node_gate(gates)      # [N, 1]
            alpha = graph_softmax(scores, batch)             # [N, 1]
            alpha = alpha                      # [N, 1]
            
        # Aggregation
        pooled = scatter(alpha * x_gated, batch, dim=0, reduce='sum')  # [B, F] 

        if self.node_gate is not None:
            return pooled, gates, alpha
        else:
            return pooled, gates
        
    def __repr__(self) -> str:
        return(f'{self.__class__.__name__}('
               f'feature_gate={self.feature_gate}'
               f'node_gate={self.node_gate}')

class GATClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.5, num_layers=1, pool_type='mean', norm_type=BatchNorm, num_groups=2):
        """   
        Graph attention network for classifying

        Args:
            in_channels: amount of nodes/auxiliary channels used
            num_classes: amount of different classes in the dataset
            dropout: value used for dropout layer
            hidden_dim: value used for reducing dimension
            num_layers: int used to set amount of GAT-layers in model
            pool_type: which type op aggregation to use
            norm_type: which normalization type to use - goes into norm_dict
        
        Returns:
            Either 1 values per image (for 2 class classification) or n values per image (for n class classification; n > 2) 
        """
        super().__init__()
    
        norm_dict = {'BatchNorm': BatchNorm, 
                    'InstanceNorm': InstanceNorm, 
                    'LayerNorm': LayerNorm, 
                    'GraphNorm': GraphNorm, 
                    'PairNorm': PairNorm,
                    'GroupNorm': lambda out_dim: DiffGroupNorm(in_channels=out_dim ,groups=num_groups)}


        self.pool_type = pool_type

        self.dropout = dropout
        self.num_layers = num_layers
        # ---- Build GAT-layers dynamically
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        dims = [in_channels] # Track feature sizes

        for i in range(num_layers):
            out_dim = max(1, dims[-1] // 2) # divide by 2 each layer
            self.gat_layers.append(GATConv(dims[-1], out_dim, dropout=dropout))
            self.norm_layers.append(norm_dict[norm_type](out_dim))  # Possible normalization methods: BatchNorm, InstanceNorm, LayerNorm, GraphNorm, PairNorm,
            dims.append(out_dim)

        self.out_dim = dims[-1]

        if pool_type == 'att':
            # Define an attention gate: a small MLP that outputs a score for each node
            self.agg = AttentionPooling(
                feature_gate=nn.Sequential(
                    nn.Linear(self.out_dim, self.out_dim),
                    nn.ReLU(),
                    nn.Linear(self.out_dim, self.out_dim),
                    nn.Sigmoid())
                    ,
                node_gate=nn.Sequential(
                    nn.Linear(self.out_dim, max(1,self.out_dim // 2)),
                    nn.ReLU(),
                    nn.Linear(max(1, self.out_dim // 2), 1))
                    )
        elif pool_type == 'mean':
            self.agg = global_mean_pool
        elif pool_type == 'max':
            self.agg = global_max_pool
        else:
            raise ValueError(f"Invalid pool_type: {pool_type}")

        # ---- Classifier -----
        self.linear = nn.Linear(self.out_dim, 1 if num_classes == 2 else num_classes) # This allows for dynamic output

    def forward(self, x, edge_index, batch, return_attention=False):
        # x: [total_nodes_in_batch, num_features]
        # edge_index: [2, num_edges_in_batch]
        
        # Use this so you can always pass feature_gates/node_gates even if not using attention pooling
        feature_gates = node_gates = att_weights = None 
        
        # GATlayer + normalization + activation + dropout
        for gat, norm in zip(self.gat_layers, self.norm_layers):
            if not return_attention:   # For training data don't need edge_index or attn_weights
                x = gat(x, edge_index)
            else:
                x, (edge_indx, att_weights) = gat(x, edge_index, return_attention_weights=True) # Need in testing edge_index 
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pooling layer
        if self.pool_type == 'att':
            if not return_attention:       # Here only 'save' gates if in test mode - Is not needed during training/validation, only for evaluation at the end
                x, _, _ = self.agg(x,batch)
            else:
                x, feature_gates, node_gates = self.agg(x,batch)
        else:
            x = self.agg(x,batch)
        x = self.linear(x)
        
        # I also want the linear weight matrix in the case of having the timesteps as nodes
        theta = gat.lin.weight
        
        if not return_attention: # In training only return data values of forward pass
            return x
        else:
            return x, (edge_indx, att_weights, feature_gates, node_gates, theta) # For testing/analyzing need also edge_index, att_weights, feature_gates and node_gates
        

#----------------#
# Main functions #
#----------------#

def classifier(data, labels, model_type, balance_classes: bool = True, 
               n_epochs: int = 10, lr: float = 1e-3, train_frac: 
               float = 0.7, val_frac: float = 0.1, patience: int = 25, 
               seed: int = 15):
    """
    General function to train either a MLP or CNN classifier on glitch classes for FD auxiliary data
    
    Args:
        See intro text of this .py file

    Returns:
        Accuracy, (true_labels, predicted_labels)
    """
    model_type = model_type.upper() # Now both lowercase and uppercase letters are possible when choosing which model to use
    
    # Checks to see if input arguments are correct:
    if len(data.shape) != 3:
        raise ValueError(
            f"Input data has wrong shape {data.shape}; expected "
            "(N_images, channels, time_steps)"
        )
    if len(labels.shape) != 1:
        raise ValueError(
            f"Input labels has wrong shape {labels.shape}; expected "
            "(N_images,)"
        )
    if model_type not in ('CNN', 'MLP'):
        raise ValueError(
            f"Model input {model_type} is not available, choose either 'CNN' or 'MLP'"
        )
    if train_frac <= 0 or val_frac <= 0:
        raise ValueError(
            "train_frac and val_frac must both be > 0"
        )
    if train_frac + val_frac >= 1:
        raise ValueError(
            f"train_frac + val_frac should be less than 1, got {train_frac + val_frac}"
        )
    
    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is:", device)
    # Consider amount of unique classes and amount per unique class
    unique, counts = np.unique(labels, return_counts = True) 
    
    num_classes = len(unique)

    least_common_class = np.argmin(counts) 
    least_common_count = counts[least_common_class]

    print(f"Least common class is class {least_common_class}, with {least_common_count} samples.")

    if model_type == 'CNN':
        model = Conv2DNet(image_height=data.shape[1], image_width=data.shape[2], num_classes=num_classes, input_channels=1)
    elif model_type == 'MLP':
        model = MLP(input_dim=data.shape[1]*data.shape[2], num_classes=num_classes)

    count_parameters(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    sys.exit(0)
    model = model.to(device)
    seed_everything(seed=seed)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # define optimizer
    scheduler = CosineAnnealingLR(optimizer, T_max = 40, eta_min = 1e-5)

    # Generate dataloaders
    dataloader_train, dataloader_val, dataloader_test = train_val_test(
        data=data, labels=labels, model_type=model_type, train_frac=train_frac, 
        val_frac=val_frac, balance_classes=balance_classes, seed = seed)

    # Train the model
    (train_loss_list, val_loss_list), lr_list, val_acc_list = train(model=model, scheduler=scheduler, train_loader=dataloader_train, val_loader = dataloader_val, 
                                           num_epochs = n_epochs, optimizer=optimizer, num_classes=num_classes, patience=patience, device=device)
    # Test the model
    accuracy, true_labels, pred_labels, probs = test(model=model, test_loader=dataloader_test, num_classes=num_classes, device=device)

    return accuracy, (true_labels, pred_labels), (train_loss_list, val_loss_list), lr_list, val_acc_list

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def GAT_classifier(data, labels, balance_classes: bool = True, 
               n_epochs: int = 100, lr: float = 1e-3, train_frac: 
               float = 0.7, val_frac: float = 0.1, topk: int = 10, 
               dropout: float = 0.5, weight_decay: float = 5e-4,
               patience: int = 15, similarity_type: str = 'cosine', mode: str = 'mean', 
               num_layers: int = 1, optim: str = 'Adam', trial = None, pool_type = 'mean', preprocess='True',
               seed: int = 100, save_imgs: bool = False, save_path: str = './saveables', norm_type: str = 'BatchNorm',
               T_max: int = 100, eta_min = 1e-5, batch_size: int = 64):
    """
    General function to train a GAT classifier on glitch classes for FD auxiliary data
    
    Args:
        See intro text of this .py file

    Returns:
        Accuracy, (true_labels, predicted_labels)
    """
    # Checks to see if input arguments are correct:
    if len(data.shape) != 3:
        raise ValueError(
            f"Input data has wrong shape {data.shape}; expected "
            "(N_images, channels, time_steps)"
        )
    if len(labels.shape) != 1:
        raise ValueError(
            f"Input labels has wrong shape {labels.shape}; expected "
            "(N_images,)"
        )
    if similarity_type not in ('cosine', 'pearson'):
        raise ValueError(
            f"Similarity type input {similarity_type} is not available, choose either 'cosine' or 'pearson'"
        )
    if mode not in ('mean', 'concat', 'single'):
        raise ValueError(
            f"Mode input {mode} is not available, choose either 'mean', 'concat' or 'single'"
        )
    if train_frac <= 0 or val_frac <= 0:
        raise ValueError(
            "train_frac and val_frac must both be > 0"
        )
    if train_frac + val_frac >= 1:
        raise ValueError(
            f"train_frac + val_frac should be less than 1, got {train_frac + val_frac}"
        )
    

    # Check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is:", device)
    # Define base paths and make directories if necessary
    save_path = Path(save_path)
    images_dir = save_path / "Images"
    model_dir = save_path / "Model"
    loss_dir = images_dir / "Loss"
    cm_dir = images_dir / "Confusion_Matrix"

    loss_dir.mkdir(parents=True, exist_ok=True) # creates all missing parent dirs too
    cm_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check which trial id
    if trial is not None:
        trial_id = trial.number
  

    # Consider amount of unique classes and amount per unique class
    unique, counts = np.unique(labels, return_counts = True) 
    
    num_classes = len(unique)

    least_common_class = np.argmin(counts) 
    least_common_count = counts[least_common_class]

    print(f"Least common class is class {least_common_class}, with {least_common_count} samples.")
    
    # Define configuration:
    cfg = {
        'n_epochs': n_epochs,
        'topk': topk,
        'lr': lr,
        'train_frac': train_frac,
        'val_frac': val_frac,
        'mode': mode,
        'similarity_type': similarity_type,
        'optimizer': optim,
        'cfg_path': f"{pool_type}_{topk}_{mode}_{similarity_type}_{optim}_{patience}_{weight_decay}_{dropout}"
    }
    
    if save_imgs == False:
        if trial is not None:
            cfg['cfg_path'] = f"trial_{trial_id}"

    # Define model + optimizer
    optimizer_dict = {
            'Adam': torch.optim.Adam,
            'RMSprop': torch.optim.RMSprop,
            'AdamW': torch.optim.AdamW,
            'RAdam': torch.optim.RAdam,
            'NAdam': torch.optim.NAdam
    }
    model = GATClassifier(in_channels = data.shape[2], num_classes=num_classes, dropout=dropout, num_layers=num_layers, pool_type=pool_type, norm_type=norm_type)
    count_parameters(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    model = model.to(device) # Put it on the used device

    seed_everything(seed=seed) # Set seed for random processes. Right now behind defining the model so weight initialization is random.

    optimizer = optimizer_dict[optim](model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = CosineAnnealingLR(optimizer, T_max = T_max, eta_min = eta_min)

    dataloader_train, dataloader_val, dataloader_test = train_val_test_gat(
                                data=data, labels=labels, train_frac=train_frac, val_frac=val_frac, 
                                balance_classes=balance_classes, topk=topk, similarity_type=similarity_type, 
                                mode=mode, seed=seed, batch_size=batch_size, preprocess=preprocess
                                )
    
    # Train the model
    (train_loss_list, val_loss_list), lr_list, val_acc_list, train_acc_list = GAT_train(model=model, scheduler=scheduler, train_loader=dataloader_train, 
                                           val_loader=dataloader_val, num_epochs=n_epochs, 
                                           optimizer=optimizer, num_classes=num_classes,
                                           cfg=cfg, model_save_path=model_dir, device=device, 
                                           trial=trial, patience=patience
                                           )
    
    # Test the model
    accuracy, true_labels, pred_labels, importance_metrics = GAT_test(model=model, test_loader = dataloader_test, 
                                                  num_classes=num_classes, cfg = cfg, 
                                                  model_save_path=model_dir, device=device)
    
    # NOTE the subparts of importance_metrics
    edge_indices_list, att_weights_list, feature_gates_list, node_gates_list, lin_weight = importance_metrics

    
    if save_imgs:
        # Plot loss and learning rate across epochs for high performing samples
        fig, ax1=plt.subplots()
        color='tab:red'
        color2='tab:orange'
        color3='tab:blue'
        color4='tab:green'
        ax1.grid()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(train_loss_list, color=color, label='train')
        ax1.plot(val_loss_list, color=color2, label='val')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend()

        ax2=ax1.twinx()
        ax2.set_ylabel('lr', color=color3)
        ax2.plot(lr_list, color=color3)
        ax2.tick_params(axis='y', labelcolor=color3)

        ax3=ax1.twinx()
        ax3.set_ylabel('Val acc', color=color4)
        ax3.plot(val_acc_list, color=color4)
        ax3.tick_params(axis='y', labelcolor=color4)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(loss_dir/f"{cfg['cfg_path']}.png")
        plt.close()

        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="", cmap="Blues", cbar=True,
                xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.savefig(cm_dir/f"{cfg['cfg_path']}.png")
        plt.close()


    return accuracy, (true_labels, pred_labels), (train_loss_list, val_loss_list), lr_list, (val_acc_list, train_acc_list), importance_metrics,




