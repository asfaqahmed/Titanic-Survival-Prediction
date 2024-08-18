import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load the data
train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)
train_df['Title'] = train_df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())

# Encode categorical variables
le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
train_df['Embarked'] = le.fit_transform(train_df['Embarked'])
train_df['Title'] = le.fit_transform(train_df['Title'])


# # Visualize survival rates
# sns.barplot(x='Sex', y='Survived', data=train_df)
# plt.title('Survival Rate by Sex')
# plt.show()

# sns.barplot(x='Pclass', y='Survived', data=train_df)
# plt.title('Survival Rate by Pclass')
# plt.show()

# sns.barplot(x='Embarked', y='Survived', data=train_df)
# plt.title('Survival Rate by Embarked')
# plt.show()

# sns.histplot(data=train_df, x='Age', hue='Survived', multiple='stack')
# plt.title('Age Distribution by Survival')
# plt.show()


# Explore the data
# print(train_df.head())
# print(train_df.info())
print(train_df.describe())

