# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values

#Since we are trying to predict for level 6.5 we use entire dataset
#First training using Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Training using Polynomial Linear Regression
# we need to create matrix of features along with the sqaured terms
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
 
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualizing the results of Linear Regression
plt.scatter(X,y,c='red')
plt.plot(X,lin_reg.predict(X), c='blue')
plt.title('Linear Regression')
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()

#Visualizing the results of Polynomial Linear Regression
plt.scatter(X,y,c='red')
plt.plot(X,lin_reg2.predict(X_poly), c='blue')
plt.title('Polynomial Linear Regression')
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()

#Predicting 6.5 level result using Linear Regression
print(lin_reg.predict([[6.5]])) # can also predict for ([[6,5],[2,3]]) inner [2,3] - rows; outer [] columns

#Predicting 6.5 level result using Polynomial Linear Regression
print(lin_reg2.predict(poly_reg.fit_transform([[6.5]]))) #we need to enter x1,x2,x3,x4 and this is the efficient model
