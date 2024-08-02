## Iris Flower Classification

This project uses the Iris dataset to classify flowers into three species: Setosa, Versicolor, and Virginica based on sepal and petal measurements. We use a RandomForestClassifier from scikit-learn for classification.

### Dataset
- **Source**: Iris dataset (IRIS.csv)
- **Features**: Sepal Length, Sepal Width, Petal Length, Petal Width
- **Target**: Species (Setosa, Versicolor, Virginica)

### Model
- **Algorithm**: RandomForestClassifier
- **Train-Test Split**: 80% training, 20% testing

### Results
```
Accuracy: 0.9666666666666667

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         10
           1       0.93      1.00      0.96         12
           2       0.92      0.83      0.87         12

    accuracy                           0.97         34
   macro avg       0.95      0.94      0.94         34
weighted avg       0.97      0.97      0.97         34
```

- **Accuracy**: The model achieved an accuracy of approximately 96.67%.
- **Classification Report**: The detailed classification report includes precision, recall, and f1-score for each class (species).

### Conclusion
The RandomForestClassifier performed well on the Iris dataset, achieving high accuracy and providing reliable classification metrics for each species. This demonstrates the effectiveness of the model for this classification task.
