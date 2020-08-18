"""
Class: CS 4267/01
Term: Spring 2020
Name: Sepehr Goshayeshi
Instructor: Dr. Aledhari
Assignment 1, Part 2
"""

# Multiple Linear Regression

# Import Necessary Packages

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D

# Read Data

data = pd.read_csv('3-Products-Multiple.csv')
print(data.columns) # columns

# Declare X & Y and Reshape the Data"""

X = data[['Product_1','Product_3']].values
y = data['Profit'].values.reshape(-1,1)

# Split Data for Training and Testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training Algorithm

reg = LinearRegression()
reg.fit(X_train,y_train)
reg_coef = reg.score(X_train,y_train)
print ("Regression coefficient:", reg_coef)

# Make Prediction on Test Data

y_pred = reg.predict(X_test)

# Error Score

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

# Plot Results in 3D

fig = plt.figure(figsize=(25,20))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='r', label='Actual Profit') # red actual profit
ax.scatter(X_test[:,0], X_test[:,1], y_pred,c='g', label = 'Predict Profit') # green predicted profit
plt.show()