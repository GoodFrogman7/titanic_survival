# Titanic Survival Prediction Project

## Overview
This project analyzes the famous Titanic dataset to predict passenger survival using exploratory data analysis (EDA) and machine learning techniques. Key insights were derived through visualizations, and predictive models were built using Logistic Regression and Random Forest algorithms.

---

## Key Results
- **Logistic Regression**:
  - Accuracy: 81%
  - F1 Score: 76%
- **Random Forest**:
  - Accuracy: 82%
  - F1 Score: 77%
- **Feature Importance**:
  - Key predictors: `Sex`, `Pclass`, `Age`.

---

## Visualizations
### Survival Rates by Gender
![Survival Rates by Gender](visuals/survival_by_gender.png)

### Survival Rates by Passenger Class
![Survival Rates by Passenger Class](visuals/survival_by_class.png)

### Age Distribution by Survival
![Age Distribution by Survival](visuals/age_distribution_by_survival.png)

### Correlation Heatmap
![Correlation Heatmap](visuals/correlation_heatmap.png)

### Confusion Matrices
- Logistic Regression:  
  ![Confusion Matrix Logistic](visuals/confusion_matrix_logistic.png)
- Random Forest:  
  ![Confusion Matrix Random Forest](visuals/confusion_matrix_randomforest.png)

### Feature Importance
![Feature Importance](visuals/feature_importance.png)


Tools and Skills Used
Languages and Libraries
Python: Core programming language for data manipulation, analysis, and modeling.
Pandas: For data cleaning, exploration, and manipulation.
NumPy: For numerical operations and computations.
Matplotlib: For static visualizations of trends and patterns.
Seaborn: For advanced data visualization, such as heatmaps and distribution plots.
Scikit-learn: For implementing machine learning models, evaluation metrics, and preprocessing.
Joblib: For saving and loading trained models.
Machine Learning Techniques
Logistic Regression: A fundamental classification algorithm to predict survival probabilities.
Random Forest: An ensemble-based machine learning model for improved predictions.
Hyperparameter Tuning: Fine-tuned model performance using GridSearchCV.
Data Science Concepts
Exploratory Data Analysis (EDA): To uncover survival patterns by analyzing gender, class, and age.
Feature Engineering: Encoded categorical variables, handled missing data, and optimized feature selection.
Correlation Analysis: To study relationships between features and survival probability.
Visualization Techniques
Created insights-driven visualizations:
Survival rates by gender and passenger class.
Age distribution by survival.
Correlation heatmaps for feature relationships.
Confusion matrices for classification results.
Feature importance plots for Random Forest.
Project Development and Version Control
Git: For version control of scripts, datasets, and visualizations.
GitHub: To host and share the project repository.
Data Handling
Kaggle Titanic Dataset: Sourced, cleaned, and processed raw data for analysis and modeling.
CSV Management: Saved intermediate outputs like cleaned datasets, predictions, and misclassified samples for reproducibility.
Key Skills Gained
Data Cleaning: Imputed missing values and handled inconsistent data.
Model Evaluation: Assessed accuracy, precision, recall, and F1 scores to validate predictions.
Model Interpretability: Leveraged feature importance to explain results.


