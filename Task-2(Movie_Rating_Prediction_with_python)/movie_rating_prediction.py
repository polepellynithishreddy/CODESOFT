import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# Load the dataset
file_path = r'C:\Users\HP\OneDrive\Desktop\CODSOFT\MOVIE_RATING_PREDICTION_WITH_PYTHON\IMDb_Movies_India.csv'
movie_data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Function to clean the 'Votes' column
def clean_votes(value):
    try:
        return int(value.replace(',', ''))
    except:
        return None

# Function to clean the 'Year' column
def clean_year(value):
    try:
        return int(value)
    except:
        return None

# Clean the 'Votes' column
movie_data['Votes'] = movie_data['Votes'].apply(clean_votes)

# Remove non-numeric characters from 'Year' and convert to integer
movie_data['Year'] = movie_data['Year'].str.extract(r'(\d{4})')
movie_data['Year'] = movie_data['Year'].apply(clean_year)

# Remove 'min' from 'Duration' and convert to integer
movie_data['Duration'] = movie_data['Duration'].str.replace(' min', '')
movie_data['Duration'] = pd.to_numeric(movie_data['Duration'], errors='coerce')

# Filling missing values with appropriate strategies
movie_data['Rating'] = movie_data['Rating'].fillna(movie_data['Rating'].mean())
movie_data['Duration'] = movie_data['Duration'].fillna(movie_data['Duration'].mean())
movie_data['Votes'] = movie_data['Votes'].fillna(movie_data['Votes'].mean())
movie_data['Year'] = movie_data['Year'].fillna(movie_data['Year'].mode()[0])

# Drop rows with missing values in crucial columns
movie_data = movie_data.dropna(subset=['Genre', 'Director', 'Actor 1'])

# Ensure all columns used for concatenation are strings
movie_data['Director'] = movie_data['Director'].astype(str)
movie_data['Actor 1'] = movie_data['Actor 1'].astype(str)
movie_data['Actor 2'] = movie_data['Actor 2'].astype(str)
movie_data['Actor 3'] = movie_data['Actor 3'].astype(str)
movie_data['Genre'] = movie_data['Genre'].astype(str)

# Feature Engineering
movie_data['Director_Actor1'] = movie_data['Director'] + '_' + movie_data['Actor 1']
movie_data['Director_Actor2'] = movie_data['Director'] + '_' + movie_data['Actor 2']
movie_data['Director_Actor3'] = movie_data['Director'] + '_' + movie_data['Actor 3']
movie_data['Genre_Director'] = movie_data['Genre'] + '_' + movie_data['Director']

# Encode categorical features
label_encoder = LabelEncoder()
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Director_Actor1', 'Director_Actor2', 'Director_Actor3', 'Genre_Director']:
    movie_data[col] = label_encoder.fit_transform(movie_data[col])

# Define the feature matrix and target vector
X = movie_data[['Year', 'Duration', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Votes', 'Director_Actor1', 'Director_Actor2', 'Director_Actor3', 'Genre_Director']]
y = movie_data['Rating']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use a smaller parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Ensure GridSearchCV uses all available cores
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters from grid search
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

# Train a Random Forest model with the best parameters
best_model = RandomForestRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Feature importance
feature_importances = best_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# Select important features
important_features = importance_df[importance_df['Importance'] > 0.01]['Feature']
X = movie_data[important_features]

# Train a Ridge regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = ridge_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (Ridge): {mse}')

# Cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
print(f'Cross-Validated MSE: {-cv_scores.mean()}')
