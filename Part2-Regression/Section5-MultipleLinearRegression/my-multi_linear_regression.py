# -*- coding: utf-8 -*-

#Data pre-proccessing

#import the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importling the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Encoding the catagoricall variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding The Dummy Variable Trap
X = X[:, 1:]

#split the data into the training set and test set
from sklearn.cross_validation import train_test_split
X_trian, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)