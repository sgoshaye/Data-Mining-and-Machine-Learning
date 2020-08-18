"""
Class: CS 4267/01
Term: Spring 2020
Name: Sepehr Goshayeshi
Instructor: Dr. Aledhari
Assignment 1, Part 3
"""

# Polynomial Regression

# Import Necessary Packages

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Read Data

data_poly = pd.read_csv('Propose-Salaries-Polynomial.csv')
print(data_poly.columns) # columns
data_poly['Position'] = pd.np.arange(1,len(data_poly)+1)
data_poly_X = pd.np.array(data_poly[['Position','Level']])

# Split Data for Training and Testing

y = data_poly['Salary']
X_train, X_test, y_train, y_test = train_test_split(data_poly_X, y, test_size=0.2, random_state=0, shuffle=False)

# Create Polynomial Features

poly = PolynomialFeatures(degree=3,include_bias=False)
X_ = poly.fit_transform(data_poly_X)
to_predict_level = 6.5
X_to_predict = poly.fit_transform([[to_predict_level,to_predict_level]])
X_train_transform = poly.fit_transform(X_train)
X_test_transform = poly.fit_transform(X_test)

# Training Algorithm

reg = LinearRegression()
reg.fit(X_train_transform,y_train)
train_pred = reg.predict(X_train_transform)

# Make Prediction on Test Data

y_pred = reg.predict(X_test_transform)

# Estimating 6.5 level

print('predicted salary for 6.5 level')
print(reg.predict(X_to_predict)[0])

# Plot Results

plt.plot(X_train[:,1], train_pred, c='g')
plt.scatter(data_poly['Level'], data_poly['Salary'], c='r')
plt.scatter([to_predict_level], reg.predict(X_to_predict), c='b') # blue is estimated 6.5 level
plt.show()