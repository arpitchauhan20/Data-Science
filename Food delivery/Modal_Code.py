# Food Delivery Time Prediction Project

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load Dataset
data = pd.read_csv('/mnt/data/food_delivery_data.csv')

# 3. Quick Overview
print(data.head())
print(data.info())
print(data.describe())

# 4. Handling Missing Values
data = data.dropna()  # Drop rows with missing values

# 5. EDA
plt.figure(figsize=(8,5))
sns.histplot(data['Delivery_Time_min'], bins=30, kde=True)
plt.title("Distribution of Delivery Time")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x='Distance_km', y='Delivery_Time_min', data=data, hue='Traffic_Density')
plt.title("Delivery Time vs Distance")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Restaurant_Type', y='Delivery_Time_min', data=data)
plt.title("Delivery Time by Restaurant Type")
plt.xticks(rotation=45)
plt.show()

# 6. Encoding Categorical Variables
categorical_cols = ['Restaurant_Type', 'Weather', 'Traffic_Density', 'Customer_Location']
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_features = pd.DataFrame(encoder.fit_transform(data[categorical_cols]),
                                columns=encoder.get_feature_names_out(categorical_cols))

# Combine numeric and encoded features
numeric_cols = ['Distance_km', 'Order_Size', 'Prep_Time_min']
X = pd.concat([data[numeric_cols], encoded_features], axis=1)
y = data['Delivery_Time_min']

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))

# 9. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))

# 10. Feature Importance
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title("Feature Importance (Random Forest)")
plt.show()

# 11. Sample Prediction
sample_order = pd.DataFrame({
    'Distance_km': [6],
    'Order_Size': [3],
    'Prep_Time_min': [12],
    # Update one-hot encoded categorical columns according to your dataset
    'Restaurant_Type_Cafe': [0],
    'Restaurant_Type_Casual Dining': [1],
    'Restaurant_Type_Fast Food': [0],
    'Weather_Cloudy': [0],
    'Weather_Rainy': [1],
    'Traffic_Density_High': [1],
    'Traffic_Density_Low': [0],
    'Customer_Location_Zone B': [1],
    'Customer_Location_Zone C': [0]
})

predicted_time = rf_model.predict(sample_order)
print("Predicted Delivery Time:", predicted_time[0], "minutes")
