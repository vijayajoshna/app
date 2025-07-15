import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load diabetes dataset from sklearn or a CSV
from sklearn.datasets import load_diabetes
data = load_diabetes(as_frame=True)
X = data.data
y = (data.target > 140).astype(int)  # Make it a binary classification

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and Scaler saved!")
