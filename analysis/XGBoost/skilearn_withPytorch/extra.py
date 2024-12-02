import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from torch import FloatTensor
from sklearn.tree import DecisionTreeRegressor

class LossFunctionMinimizer(nn.Module):
    def __init__(self, type):
        # type can be one of : "regressor" or "classifier"
        super(LossFunctionMinimizer, self).__init__()
        self.type = type
        self.current_leaf_value = nn.Parameter(data=FloatTensor([0.0]), requires_grad=True)
    def reinitialize_variable(self):
        self.current_leaf_value.data = FloatTensor([0.0])
    def refine_previous_predictions(self, previous_predictions):
        new_predictions = previous_predictions + self.current_leaf_value
        return new_predictions
    def loss(self, previous_predictions, targets_leaf_tensor):
        if self.type == "regressor":
            return self.loss_regressor(previous_predictions, targets_leaf_tensor)
        elif self.type == "classifier":
            return self.loss_classifier(previous_predictions, targets_leaf_tensor)
        else:
            raise Exception("Not supported")
    def loss_classifier(self, previous_predictions, targets_leaf_tensor):
        logodds = self.refine_previous_predictions(previous_predictions)
        probabilities = 1.0 / (1.0 + torch.exp(-logodds))
        loss = F.binary_cross_entropy(probabilities, targets_leaf_tensor)
        return loss
    def loss_regressor(self, previous_predictions, targets_leaf_tensor):
        values = self.refine_previous_predictions(previous_predictions)
        loss = F.mse_loss(values, targets_leaf_tensor)
        return loss  
    

class ResidualsCalculator(nn.Module):
    def __init__(self, predicted_values, type):
        super(ResidualsCalculator, self).__init__()
        self.type = type
        self.predicted_values = nn.Parameter(data=torch.zeros(predicted_values.shape), requires_grad=True)
        self.predicted_values.data = predicted_values
    def forward(self):
        my_parameters = self.predicted_values
        return my_parameters
    def loss(self, targets):
        if self.type == "regressor":
            return self.loss_regressor(targets)
        elif self.type == "classifier":
            return self.loss_classifier(targets)
        else:
            raise Exception("Not supported")
    def loss_classifier(self, targets):
        logodds = self.forward()
        probabilities = 1.0 / (1.0 + torch.exp(-logodds))
        loss = F.binary_cross_entropy(probabilities, targets)
        return loss
    def loss_regressor(self, previous_predictions, targets):
        values = self.forward()
        loss = F.mse_loss(values, targets)
        return loss
    

def fit_regression_tree_classifier_to_residuals(X_data, y_data, max_depth): # y_data -> residuals
    tree_regressor = DecisionTreeRegressor(max_depth=max_depth)
    tree_regressor.fit(X_data, y_data)
    leaf_buckets = []
    for i in range(X_data.shape[0]):
        leaf_buckets.append(tuple(tree_regressor.decision_path(X_data[i, :].reshape(1, -1)).todok().keys()))
    unique_paths = list(set(leaf_buckets))
    return (leaf_buckets, unique_paths, tree_regressor)