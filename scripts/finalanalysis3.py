# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib

# Load the dataset
file_path = r"C:\Users\User\Downloads\archive\Titanic-Dataset.csv"
df = pd.read_csv(file_path)

# SECTION 1: Data Cleaning and Preprocessing
print("Dataset Info:")
print(df.info())

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df = df.drop(columns=['Cabin'], errors='ignore')
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop unnecessary columns
df = df.drop(['Name', 'Ticket'], axis=1)

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Save cleaned data
df.to_csv('data/titanic_cleaned.csv', index=False)
print("Cleaned dataset saved successfully!")

# SECTION 2: Exploratory Data Analysis (EDA)
# Plot survival rates by gender
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival Rates by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(['Not Survived', 'Survived'])
plt.savefig('visuals/survival_by_gender.png', dpi=300)
plt.show()

# Plot survival rates by passenger class
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival Rates by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(['Not Survived', 'Survived'])
plt.savefig('visuals/survival_by_class.png', dpi=300)
plt.show()

# Plot age distribution by survival
sns.histplot(data=df, x='Age', kde=True, hue='Survived')
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Density')
plt.savefig('visuals/age_distribution_by_survival.png', dpi=300)
plt.show()

# Correlation heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.savefig('visuals/correlation_heatmap.png', dpi=300)
plt.show()

# SECTION 3: Feature Selection and Splitting Data
X = df.drop(['Survived', 'PassengerId'], axis=1)  # Features
y = df['Survived']  # Target variable

# Standardize features for Logistic Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SECTION 4: Machine Learning Models
# Logistic Regression
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Evaluate Logistic Regression
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('visuals/confusion_matrix_logistic.png', dpi=300)
plt.show()

# Random Forest (with GridSearchCV)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
rf_clf = grid_search.best_estimator_
rf_pred = rf_clf.predict(X_test)

# Evaluate Random Forest
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix: Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('visuals/confusion_matrix_randomforest.png', dpi=300)
plt.show()

# Feature importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_clf.feature_importances_})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title('Feature Importance (Random Forest)')
plt.savefig('visuals/feature_importance.png', dpi=300)
plt.show()

# Save models and scaler
joblib.dump(log_reg, 'logistic_regression_model.pkl')
joblib.dump(rf_clf, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Models and scaler saved successfully!")
