"""
Class: CS 4267/01
Term: Spring 2020
Name: Sepehr Goshayeshi
Instructor: Dr. Aledhari
Assignment 1, Part 1
"""

# Simple Linear Regression

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Read Data

data = pd.read_csv('Salaries-Simple_Linear.csv', header=0) # reading data

# Declare X & Y and Reshape the Data

X = data['Years_of_Expertise'].values.reshape(-1,1)
y = data['Salary'].values #what we are trying to predict

# Split Data for Training and Testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training Algorithm

reg = LinearRegression()  
reg.fit(X_train, y_train) #training the algorithm
reg_coef = reg.score(X_train,y_train)
print ("Regression coefficient:", reg_coef)

# Make Prediction on Test Data

y_pred = reg.predict(X_test)

# Error Score

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

# Plot Results

plt.scatter(X_test,y_test)
plt.plot(X_test, y_pred, color='red')
plt.show()