import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# Load the data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

# Feature Engineering
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)
train_df['Title'] = train_df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df['IsAlone'] = (test_df['FamilySize'] == 1).astype(int)
test_df['Title'] = test_df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())

# Combine train and test for consistent encoding
combined_df = pd.concat([train_df, test_df], sort=False)

# Encode categorical variables
le = LabelEncoder()
combined_df['Sex'] = le.fit_transform(combined_df['Sex'].astype(str))
combined_df['Embarked'] = le.fit_transform(combined_df['Embarked'].astype(str))
combined_df['Title'] = le.fit_transform(combined_df['Title'].astype(str))

# Separate the data back
train_df['Sex'] = combined_df['Sex'].iloc[:len(train_df)]
train_df['Embarked'] = combined_df['Embarked'].iloc[:len(train_df)]
train_df['Title'] = combined_df['Title'].iloc[:len(train_df)]

test_df['Sex'] = combined_df['Sex'].iloc[len(train_df):]
test_df['Embarked'] = combined_df['Embarked'].iloc[len(train_df):]
test_df['Title'] = combined_df['Title'].iloc[len(train_df):]

# Handle missing values
imputer = SimpleImputer(strategy='median')
train_df['Age'] = imputer.fit_transform(train_df[['Age']])
test_df['Age'] = imputer.transform(test_df[['Age']])

imputer = SimpleImputer(strategy='most_frequent')
train_df['Embarked'] = imputer.fit_transform(train_df[['Embarked']])
test_df['Embarked'] = imputer.transform(test_df[['Embarked']])

# Drop irrelevant columns
X = train_df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'])
y = train_df['Survived']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_val)

# Evaluate the model
print('Logistic Regression Accuracy:', accuracy_score(y_val, y_pred_logreg))
print(classification_report(y_val, y_pred_logreg))

# Make predictions on the test data
X_test = test_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
test_df['Survived'] = logreg.predict(X_test)

# Prepare the submission file
submission = test_df[['PassengerId', 'Survived']]
submission.to_csv('titanic_predictions.csv', index=False)
