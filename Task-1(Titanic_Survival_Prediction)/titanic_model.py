import pandas as pd 

# Load the dataset
data = pd.read_csv('C:/Users/HP/OneDrive/Desktop/titanic_survival_prediction/Titanic-Dataset.csv')

# Display the first 10 rows
data.head(10)

# Display the shape of the dataset
shape = data.shape

# Generate summary statistics
summary_statistics = data.describe()

# Show information about the dataset
info = data.info()

# Check for missing values
missing_values = data.isna().sum()

# Check for duplicate entries
duplicate_entries = data.duplicated().sum()

# Analyze specific columns
survived_counts = data["Survived"].value_counts()
pclass_counts = data["Pclass"].value_counts()
sex_counts = data["Sex"].value_counts()
embarked_counts = data["Embarked"].value_counts()

shape, summary_statistics, info, missing_values, duplicate_entries, survived_counts, pclass_counts, sex_counts, embarked_counts

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

numeric_data = data.select_dtypes(include=[np.number])

# Plot distributions
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
data["Survived"].value_counts().plot(kind='pie', autopct="%0.2f%%", ax=axs[0, 0], title="Survived")
data["Pclass"].value_counts().plot(kind='pie', autopct="%0.2f%%", ax=axs[0, 1], title="Pclass")
data["Sex"].value_counts().plot(kind='pie', autopct="%0.2f%%", ax=axs[1, 0], title="Sex")
data["Embarked"].value_counts().plot(kind='pie', autopct="%0.1f%%", ax=axs[1, 1], title="Embarked")
plt.show()

# Heatmap of correlations
plt.figure(figsize=(12, 12))
sns.heatmap(numeric_data.corr(), annot=True, fmt="0.2f")
plt.show()

# Boxplot of Age
plt.figure(figsize=(10, 6))
sns.boxplot(y="Age", data=data)
plt.show()

# Count plots
fig, axs = plt.subplots(3, 1, figsize=(12, 18))
sns.countplot(data["Survived"], ax=axs[0]).set(title="Count of Passengers by Survived")
sns.countplot(data["Pclass"], ax=axs[1]).set(title="Count of Passengers by Pclass")
sns.countplot(x="Survived", hue="Pclass", data=data, ax=axs[2]).set(title="Count of Passengers by Survived and Pclass")
plt.show()

# Histograms
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].hist(data["Age"].dropna(), color="pink")
axs[0].set(xlabel="Age", ylabel="Frequency", title="Distribution of Age")
axs[1].hist(data["Fare"], color="pink")
axs[1].set(xlabel="Fare", ylabel="Frequency", title="Distribution of Fare")
plt.show()

from sklearn.preprocessing import MinMaxScaler

# Drop irrelevant columns
data = data.drop(["PassengerId", "Name", "Ticket"], axis=1)

# Handle missing values
data["Age"] = data["Age"].fillna(data['Age'].median())
data["Embarked"] = data["Embarked"].fillna(data['Embarked'].mode()[0])

# Encode categorical features
data['Cabin'] = data['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
dummies = pd.get_dummies(data[["Sex", "Embarked"]])
data = pd.concat([data, dummies], axis=1)
data = data.drop(["Sex","Embarked"], axis=1)

# Scale numerical features
minmax = MinMaxScaler()
data['Fare'] = minmax.fit_transform(data[["Fare"]])
data['Age'] = minmax.fit_transform(data[["Age"]])

# Prepare feature matrix X and target vector y
X = data[["Pclass", "Sex_male", "Sex_female", "Age", "SibSp", "Parch", "Cabin", "Embarked_C","Embarked_S","Embarked_Q"]]
y = data['Survived']

X.head(), y.head()

from sklearn.preprocessing import MinMaxScaler

# Drop irrelevant columns
columns_to_drop = ["PassengerId", "Name", "Ticket"]
existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
data = data.drop(columns=existing_columns_to_drop)


# Handle missing values
data["Age"] = data["Age"].fillna(data['Age'].median())
# Check if 'Embarked' exists before attempting to fill missing values
if 'Embarked' in data.columns:
    print("Embarked column exists.")
    try:
        most_frequent_embarked = data['Embarked'].mode()[0]
        print("Most frequent value in 'Embarked':", most_frequent_embarked)
    except Exception as e:
        print("Error occurred while calculating mode:", e)
else:
    print("Column 'Embarked' does not exist in the DataFrame.")


# Scale numerical features
minmax = MinMaxScaler()
data['Fare'] = minmax.fit_transform(data[["Fare"]])
data['Age'] = minmax.fit_transform(data[["Age"]])

# Prepare feature matrix X and target vector y
X = data[["Pclass", "Sex_male", "Sex_female", "Age", "SibSp", "Parch", "Cabin", "Embarked_C","Embarked_S","Embarked_Q"]]
y = data['Survived']

# Encode categorical features
# Ensure 'Cabin' feature is processed as needed (if it should be binary)
data['Cabin'] = data['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)

# Check columns before applying dummies
if 'Sex' in data.columns and 'Embarked' in data.columns:
    print("Columns 'Sex' and 'Embarked' exist.")
else:
    print("One or both columns are missing.")

print(X.head())
print(y.head())

