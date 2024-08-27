import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Step 2: Load Data
data = pd.read_csv('position_salaries.csv')

# Extracting independent and dependent variables
X = data.iloc[:, 1:2].values # Using only the 'Level' feature
y = data.iloc[:, 2].values

# Step 3: Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.squeeze(sc_y.fit_transform(y.reshape(-1, 1))) # Reshaping y to avoid warnings

# Step 5: Fit SVR Model
regressor = SVR(kernel='rbf')  # You can try different kernels like 'linear', 'poly', etc.
regressor.fit(X, y)

# Step 6: Visualize Results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Support Vector Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
