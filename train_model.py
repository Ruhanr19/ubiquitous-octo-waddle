import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor


print("Loading dataset...")

# 🔥 Use smaller sample for speed
df = pd.read_csv("household_power_consumption.csv").sample(20000, random_state=42)

# Replace '?' and remove missing
df = df.replace('?', np.nan).dropna()

# Combine date and time
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')

# Feature engineering
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month

# Drop unnecessary columns
df = df.drop(['Date', 'Time', 'datetime'], axis=1)

# Convert to float
df = df.astype(float)

# Define features and target
X = df.drop('Global_active_power', axis=1)
y = df['Global_active_power']

print("\nFeatures used by model:")
print(list(X.columns))

# Train model
model = RandomForestRegressor(n_estimators=20, random_state=42)
model.fit(X, y)

print("\n✅ Model trained successfully!")

# ==============================
# USER INPUT SECTION
# ==============================

print("\nEnter values to predict electricity consumption:\n")

inputs = []

for col in X.columns:
    val = float(input(f"{col}: "))
    inputs.append(val)

# Convert to DataFrame (IMPORTANT FIX)
input_df = pd.DataFrame([inputs], columns=X.columns)

# Predict
prediction = model.predict(input_df)

# Output
print(f"\n⚡ Predicted Power Consumption: {prediction[0]:.4f} kW")
print(f"⚡ Estimated Energy (1 hour): {prediction[0]:.4f} kWh")

import joblib
joblib.dump(model, "model.pkl")
print("Model saved!")