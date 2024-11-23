import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#load the titanic dataset
file_path = r"C:\Users\User\Downloads\archive\Titanic-Dataset.csv"
df = pd.read_csv(file_path)


# Display basic information about the dataset
print("Dataset Info:")
print(df.info())

#Display basic statistics for numerical columns
print("\nDataset Statistics:")
print(df.describe())

# Display column names
print("\nColumns in Dataset:")
print(df.columns)

# Check for missing values in each column
print("\nMissing Values:")
print(df.isnull().sum()) #check notebook 

# Percentage of missing values in each column,ask chat gpt
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

#handling missing values
df['Age'] = df['Age'].fillna(df['Age'].median())

#If a column has a significant percentage of missing data (e.g., >50%), it’s often better to drop it:
df = df.drop(columns=['Cabin'], errors='ignore')

# Check for remaining missing values
print(df.isnull().sum())

# Save the cleaned dataset to the 'data' folder
df.to_csv('data/titanic_cleaned.csv', index=False)
print("Cleaned dataset saved successfully!")


#visualize insights
#3.1 Plot survival rates by gender
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival Rates by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(['Not Survived', 'Survived'])
plt.show()

# Plot the distribution of passenger classes
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival Rates by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(['Not Survived', 'Survived'])
plt.show()

sns.histplot(data=df, x='Age', kde=True, hue='Survived')
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()





# Drop unnecessary columns early
df = df.drop(['Name', 'Ticket'], axis=1)

# Confirm the updated columns
print("Columns after dropping unnecessary ones:", df.columns)

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Fill missing values in 'Age' and 'Embarked'
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Calculate the correlation matrix
correlation_matrix = df.corr()
print("Correlation Matrix:\n", correlation_matrix)

# Plot the heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()




