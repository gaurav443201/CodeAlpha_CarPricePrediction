import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Load the dataset
print("Loading car price dataset...")
csv_path = os.path.join(base_dir, "car data.csv")
df = pd.read_csv(csv_path)

# Compute age of the car (relative to 2026)
df['Car_Age'] = 2026 - df['Year']

# Drop columns that are not useful or have too many categories (like Car_Name, Year)
df = df.drop(columns=['Car_Name', 'Year'])

print("Dataset preview:")
print(df.head())

# 2. Encode categorical variables (Fuel_Type, Selling_type, Transmission)
df_encoded = pd.get_dummies(df, drop_first=True)

# Convert boolean columns (True/False) to 0/1 integers
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'bool':
        df_encoded[col] = df_encoded[col].astype(int)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(base_dir, "correlation_heatmap.png"))
plt.close()

# 3. Split dataset into Features (X) and Target (y)
X = df_encoded.drop(columns=['Selling_Price'])
y = df_encoded['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train a Random Forest Regressor
print("Training the Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R-squared Score on Test Set: {r2 * 100:.2f}%")
print(f"Mean Absolute Error (MAE): {mae:.2f} Lakhs")

# Plot Actual vs Predicted prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title("Actual vs. Predicted Selling Price")
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.savefig(os.path.join(base_dir, "actual_vs_predicted.png"))
plt.close()

# Save feature importance plot
importances = model.feature_importances_
feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importances.values, y=feat_importances.index, palette='viridis', hue=feat_importances.index, legend=False)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.savefig(os.path.join(base_dir, "feature_importance.png"))
plt.close()

print("Plots saved successfully!")
