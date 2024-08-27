import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Step 2: Load Data
data = pd.read_csv('position_salaries.csv')

# Extracting independent and dependent variables
X = data.iloc[:, 1:2].values # Using only the 'Level' feature
y = data.iloc[:, 2].values

# Step 5: Fit Polynomial Regression Model
poly_reg = PolynomialFeatures(degree=3) # You can change the degree as per your choice
X_poly = poly_reg.fit_transform(X)
polynomial_regressor = LinearRegression()
polynomial_regressor.fit(X_poly, y)

# Step 6: Visualize Results
plt.scatter(X, y, color='red')
plt.plot(X, polynomial_regressor.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
