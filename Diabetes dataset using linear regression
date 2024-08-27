import numpy as np  
import matplotlib.pyplot as mtp  
import pandas as pd 
from sklearn import datasets, linear_model
diabetes = datasets.load_diabetes()
print ("Feature names:",diabetes['feature_names'])
print("first five rows of data:\n",diabetes['data'][:5])

x = diabetes.data[:, np.newaxis, 2]
y = diabetes.target # define the target variable (dependent variable) as y
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)

from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)  

y_pred= regressor.predict(x_test)  
x_pred= regressor.predict(x_train)  

mtp.scatter(x_train, y_train, color="green")   
mtp.plot(x_train, x_pred, color="red")    
mtp.title("Linear Regression on Diabetes Dataset")  
mtp.xlabel("Feature")  
mtp.ylabel("Target")  
mtp.show()   

mtp.scatter(x_test, y_test, color="blue")   
mtp.plot(x_train, x_pred, color="red")    
mtp.title("Linear Regression on Diabetes Dataset")  
mtp.xlabel("Feature")  
mtp.ylabel("Target")  
mtp.show() 

print('Coefficients:', regressor.coef_)
print('Intercept:', regressor.intercept_)

from sklearn.metrics import mean_squared_error, r2_score
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))

print('Coefficient of Determination (R^2 Score):', r2_score(y_test, y_pred))
