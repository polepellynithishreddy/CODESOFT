import pandas as pd

import pandas as pd

# Load the dataset
file_path = r'C:\Users\HP\OneDrive\Desktop\CODSOFT\SALES PREDICTION USING PYTHON\advertising.csv'
try:
    data = pd.read_csv(file_path)
    print("Data loaded successfully!")
except FileNotFoundError:
    print("The file was not found. Please check the file path.")
    data = None

if data is not None:
    # Display the first few rows
    print("First few rows of the data:")
    print(data.head())

    # Check for missing values
    print("Checking for missing values:")
    print(data.isnull().sum())

    # Check data types
    print("Data types of each column:")
    print(data.dtypes)

# Display the first few rows and summary statistics
print(data.head())
print(data.describe())

from sklearn.model_selection import train_test_split

# Split the data into features and target variable
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plots
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.7, kind='scatter')
plt.show()

# Correlation matrix
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Initialize the models
lr_model = LinearRegression()
dt_model = DecisionTreeRegressor()
rf_model = RandomForestRegressor()

# Train the models
lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

# Predictions
lr_predictions = lr_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Evaluate the models
lr_mse = mean_squared_error(y_test, lr_predictions)
dt_mse = mean_squared_error(y_test, dt_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)

lr_r2 = r2_score(y_test, lr_predictions)
dt_r2 = r2_score(y_test, dt_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print(f"Linear Regression - MSE: {lr_mse}, R2: {lr_r2}")
print(f"Decision Tree - MSE: {dt_mse}, R2: {dt_r2}")
print(f"Random Forest - MSE: {rf_mse}, R2: {rf_r2}")

from sklearn.model_selection import GridSearchCV

# Example: Tuning Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_

print(f"Best parameters: {best_params}")

# Make predictions with the best model
final_predictions = best_rf_model.predict(X_test)

# Evaluate the final model
final_mse = mean_squared_error(y_test, final_predictions)
final_r2 = r2_score(y_test, final_predictions)

print(f"Final Model - MSE: {final_mse}, R2: {final_r2}")
