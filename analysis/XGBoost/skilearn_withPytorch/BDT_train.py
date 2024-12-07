import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from GenericGradientBoost import PytorchBasedGenericGradientBoost

import torch
import torch.nn as nn

# Definition of Hyper-Parameters
NUM_CLASSIFIERS = 5
MAX_DEPTH = 4
GRADIENT_BOOST_LEARNING_RATE = 0.1
MINIMIZER_LEARNING_RATE = 0.005
MINIMIZER_TRAINING_EPOCHS = 1000
USE_CUDA = 0

file_path = '/home/almalinux/workinDir_Andrea/TowardPIDwithGranularCalorimeters/dataset/results_100_100_100/final_combined.tsv'
df = pd.read_csv(file_path, sep="\t")

X = df.iloc[:, :-1] 
y = df.iloc[:, -1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Running the custom algorithm 
custom = PytorchBasedGenericGradientBoost("classifier", 0.2, NUM_CLASSIFIERS, MAX_DEPTH, GRADIENT_BOOST_LEARNING_RATE=GRADIENT_BOOST_LEARNING_RATE, MINIMIZER_LEARNING_RATE=MINIMIZER_LEARNING_RATE, MINIMIZER_TRAINING_EPOCHS=MINIMIZER_TRAINING_EPOCHS,USE_CUDA=USE_CUDA)
custom.fit(X_train.values , y_train.values )
predictions_train = custom.predict(X_train.values )
predictions_test = custom.predict(X_test.values)
print("Accuracy score for training data : {}".format(accuracy_score(np.round(predictions_train), y_train.values )))
print("Accuracy score for testing data : {}".format(accuracy_score(np.round(predictions_test), y_test.values )))