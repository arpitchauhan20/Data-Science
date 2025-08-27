# 1️ Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 2️ Load Dataset
data = pd.read_csv('Telco-Customer-Churn.csv')

# 3️ Initial Exploration
print("First 5 rows of data:\n", data.head())
print("\nDataset Info:")
print(data.info())
print("\nStatistical Summary:\n", data.describe())

# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())   

# Check skewness of numeric columns
print("\nSkewness of Numeric Columns:\n", data.select_dtypes(include=['float64', 'int64']).skew())

# 4️ Data Visualization (basic EDA)
# Churn distribution
plt.figure(figsize=(6,4))
sns.countplot(data=data, x='Churn', palette='Set2')
plt.title("Churn Distribution")
plt.show()

# Tenure vs TotalCharges
plt.figure(figsize=(8,5))
sns.scatterplot(data=data, x='tenure', y='TotalCharges', hue='Churn')
plt.title("Tenure vs Total Charges by Churn")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# 5️ Preprocessing
# Drop customerID
data = data.drop('customerID', axis=1)

# Convert TotalCharges to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with median
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# Encode categorical variables
for col in data.select_dtypes(include='object').columns:
    if col != 'Churn':
        data[col] = LabelEncoder().fit_transform(data[col])

# Encode target
data['Churn'] = data['Churn'].map({'Yes':1, 'No':0})

# Features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# 6️ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7️ Logistic Regression Model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLogistic Regression Evaluation:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]))

# 8️ Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Evaluation:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))

# 9️ Feature Importance (Random Forest)
feat_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_importances, y=feat_importances.index)
plt.title("Feature Importance")
plt.show()

# 1️0️ Additional Insights
print("\nTop 5 Features affecting Churn:\n", feat_importances.head())
print("\nSkewness corrected (if any):")
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    print(f"{col} skewness: {data[col].skew():.2f}")
