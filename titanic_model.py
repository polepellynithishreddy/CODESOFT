
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\titanic_survival_prediction\\Titanic-Dataset.csv')
data.head(10)
data.shape
data.describe()
data.info()
data.isna().sum()
data.duplicated().sum()
data["Survived"].value_counts()
data["Pclass"].value_counts()
data["Sex"].value_counts()
data["Embarked"].value_counts()
data["Survived"].value_counts().plot(kind = 'pie', autopct="%0.2f%%")
data["Pclass"].value_counts().plot(kind = 'pie', autopct="%0.2f%%")
data["Sex"].value_counts().plot(kind = 'pie', autopct="%0.2f%%")
data["Embarked"].value_counts().plot(kind = 'pie', autopct="%0.1f%%")

plt.figure(figsize=(12,12))
sns.heatmap(data.corr(), annot = True, fmt ="0.2f")
sns.boxplot(y = "Age", data = data)
sns.countplot(data["Survived"])
plt.title("Count of Passengers by Survived")
sns.countplot(data["Pclass"])
plt.title("Count of Passengers by Pclass")
sns.countplot(data["Survived"], hue = data["Pclass"])
plt.title("Count of Passengers by Survived and Pclass")
sns.countplot(data["Survived"], hue = data["Embarked"])
plt.title("Count of Passengers by Survived and Embarked")
sns.countplot(data["Survived"], hue = data["Sex"])
plt.title("Count of Passengers by Survived and Sex")
plt.hist(data["Age"], color = "pink")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.hist(data["Fare"], color = "pink")
plt.xlabel("Fare")
plt.ylabel("Frequency")
data = data.drop(["PassengerId", "Name", "Ticket"], axis = 1)
data.head(10)
minmax = MinMaxScaler()
data['Fare'] = minmax.fit_transform(data[["Fare"]])
data["Age"] = data["Age"].fillna(data['Age'].median())
data["Age"]
data["Age"] = minmax.fit_transform(data[["Age"]])
# 0 - Do not have a Cabin 
# 1 - Have a Cabin 
data['Cabin'] = data['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
sns.countplot(data["Survived"], hue = data["Cabin"])
plt.title("Count of Passengers by Survived and Cabin")
dummies = pd.get_dummies(data[["Sex", "Embarked"]])
dummies
data = pd.concat([data, dummies], axis = 1)
data = data.drop(["Sex","Embarked"], axis = 1)
plt.figure(figsize=(12,12))
sns.heatmap(data.corr(), annot = True, fmt ="0.2f")
data.isna().sum()

X = data[["Pclass", "Sex_male", "Sex_female", "Age", "SibSp", "Parch", "Cabin", "Embarked_C","Embarked_S","Embarked_Q"]]
y = data['Survived']
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state  = 45) 

pipe1 = RandomForestClassifier(n_estimators = 10)
pipe1.fit(X_train, y_train)
print("Score of training set",pipe1.score(X_train, y_train))
print("Score of testing set", pipe1.score(X_test, y_test))
y_pred = pipe1.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt ="0.2f")

print("Classicication Report ",classification_report(y_test, pipe1.predict(X_test)))

pipe = DecisionTreeClassifier()
pipe.fit(X_train, y_train)

print("Score of training set",pipe.score(X_train, y_train))
print("Score of testing set", pipe.score(X_test, y_test))
y_pred = pipe.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt ="0.2f")

print("Classicication Report ",classification_report(y_test, pipe.predict(X_test)))

lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Score of training set",lr.score(X_train, y_train))
print("Score of testing set", lr.score(X_test, y_test))
y_pred = lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt ="0.2f")

print("Classicication Report ",classification_report(y_test, lr.predict(X_test)))

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print("Score of training set",knn.score(X_train, y_train))
print("Score of testing set", knn.score(X_test, y_test))
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt ="0.2f")

print("Classicication Report ",classification_report(y_test, knn.predict(X_test)))

model = GradientBoostingClassifier()
param_grid = {'n_estimators': [100, 200,300], 'learning_rate': [0.001, 0.01,0.1], 'max_depth': [3, 5]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt ="0.2f")

gbc = GradientBoostingClassifier(n_estimators= 300, learning_rate=0.01, max_depth=5)

estimators = [('gbc', gbc),("forest" , pipe1)]
stacking_clf = StackingClassifier(estimators=estimators,
                                  final_estimator= gbc)

stacking_clf.fit(X_train, y_train)

y_pred = stacking_clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print("Classification Report", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt ="0.2f")

model_names = ["Random Forest","Decision Tree","Logistic Regression", "KNN", "Gradient Boost", "Stacked Classifier"]
train_accuracies = [pipe1.score(X_train, y_train), pipe.score(X_train, y_train),
                    lr.score(X_train, y_train),knn.score(X_train, y_train),best_model.score(X_train, y_train),
                   stacking_clf.score(X_train, y_train)]    
plt.figure(figsize=(10, 6))
plt.bar(model_names, train_accuracies, label='Training Accuracy', color = ["pink", "purple", "blue", "green", "yellow", "red"])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Training Accuracy of Models')
plt.legend()
plt.show()

model_names = ["Random Forest","Decision Tree","Logistic Regression", "KNN", "Gradient Boost", "Stacked Classifier"]
train_accuracies = [pipe1.score(X_test, y_test), pipe.score(X_test, y_test),
                    lr.score(X_test, y_test),knn.score(X_test, y_test),best_model.score(X_test, y_test),
                   stacking_clf.score(X_test, y_test)]
colors = ["pink", "purple", "blue", "green", "yellow", "red"]
plt.figure(figsize=(10, 6))
plt.bar(model_names, train_accuracies, label='Training Accuracy', color = colors)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Training Accuracy of Models')
plt.legend()
plt.show()

