# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\User\OneDrive - The Pennsylvania State University\Desktop\dsproject#1\data\titanic_cleaned.csv")

# Data Preprocessing
# Drop unnecessary columns
df = df.drop(['Name', 'Ticket'], axis=1)

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Feature Selection
X = df.drop(['Survived', 'PassengerId'], axis=1)  # Features
y = df['Survived']  # Target variable

# Standardize features for Logistic Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training: Logistic Regression
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)

# Model Prediction
y_pred = log_reg.predict(X_test)

# Evaluate Logistic Regression
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report (Logistic Regression):\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix (Logistic Regression):\n", confusion_matrix(y_test, y_pred))

# Confusion Matrix Heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Model Training: Random Forest (with GridSearchCV)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Use the best Random Forest model
rf_clf = grid_search.best_estimator_
print("Best Parameters for Random Forest:", grid_search.best_params_)

# Model Prediction: Random Forest
rf_pred = rf_clf.predict(X_test)

# Evaluate Random Forest
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\nClassification Report (Random Forest):\n", classification_report(y_test, rf_pred))
print("\nConfusion Matrix (Random Forest):\n", confusion_matrix(y_test, rf_pred))

# Confusion Matrix Heatmap for Random Forest
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix: Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Evaluation Metrics
print("\nLogistic Regression Metrics:")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nRandom Forest Metrics:")
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))

# Feature Importance for Random Forest
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_clf.feature_importances_})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

# Plot feature importance
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Identify misclassified samples
X_test_original = pd.DataFrame(scaler.inverse_transform(X_test), columns=X.columns)  # Inverse scaling for readability
misclassified = X_test_original[(y_test.values != rf_pred)]
misclassified['Actual'] = y_test.values[y_test.values != rf_pred]
misclassified['Predicted'] = rf_pred[y_test.values != rf_pred]

print("\nMisclassified Samples:")
print(misclassified)

# Save misclassified samples to a CSV file
misclassified.to_csv('data/misclassified_samples.csv', index=False)
print("Misclassified samples saved successfully as 'misclassified_samples.csv'.")


# Save models, scaler, and predictions
joblib.dump(log_reg, 'logistic_regression_model.pkl')
joblib.dump(rf_clf, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

predictions = pd.DataFrame({
    'Actual': y_test,
    'Logistic Regression': y_pred,
    'Random Forest': rf_pred
})
predictions.to_csv('model_predictions.csv', index=False)
print("\nModels and predictions saved successfully!")
