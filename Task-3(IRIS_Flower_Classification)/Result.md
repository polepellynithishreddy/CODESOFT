import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Option 1: Using double backslashes
# file_path = 'C:\\Users\\HP\\OneDrive\\Desktop\\CODSOFT\\IRIS_Flower_Classification\\IRIS.csv'

# Option 2: Using a raw string
# file_path = r'C:\Users\HP\OneDrive\Desktop\CODSOFT\IRIS_Flower_Classification\IRIS.csv'

# Option 3: Using forward slashes
file_path = 'C:/Users/HP/OneDrive/Desktop/CODSOFT/IRIS_Flower_Classification/IRIS.csv'

# Load the dataset
iris_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(iris_data.head())

# Check for missing values
missing_values = iris_data.isnull().sum()
print(missing_values)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Encode the 'species' column
iris_data['species'] = label_encoder.fit_transform(iris_data['species'])

# Define features (X) and target (y)
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
