import statsmodels.api as sm
import pandas as pd

# Creating a DataFrame from the data
data = {
    'Hours_Studied': [10, 8, ..., 12],
    'Previous_Test_Scores': [85, 75, ..., 95],
    'Attendance': [90, 80, ..., 95],
    'Final_Exam_Score': [90, 85, ..., 95]
}
df = pd.DataFrame(data)

# Convert the DataFrame to numeric type
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Independent variables
X = df[['Hours_Studied', 'Previous_Test_Scores', 'Attendance']]

# Dependent variable
Y = df['Final_Exam_Score']

# Adding a constant term
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(Y, X).fit()

# Print the summary of the regression model
print(model.summary())
