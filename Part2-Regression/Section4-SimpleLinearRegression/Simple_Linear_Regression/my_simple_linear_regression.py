#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 23:52:46 2018

@author: zara
"""
# import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
list(dataset)

# split dataset into Test and Training set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
 

# Predict the Traing set results
y_predict = regressor.predict(X_test)

# Visualize the Training set results
"""plt.scatter(X_train,y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train),color ="blue") 
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show"""

# Visulaze the  Test set results
plt.scatter(X_test, y_test, color = 'green')
"""no need to change the for plot beacause LinearRegression create only one equation so whether it is train set 
or test set both are the same"""
plt.plot(X_test, y_predict, color = 'pink')
plt.title('Salary Vs Experience (Test set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show