import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

print("⏳ Reading Excel File...")
file_path = "E-commers sales analysis.xlsx"

# Read main transactions data from Sheet1
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 1. Feature Aggregation (Customer Level RFM & Behavior Metrics)
# Convert Date to datetime format
df['Date'] = pd.to_datetime(df['Date'])
max_date = df['Date'].max()

# Customer level grouping
customer_df = df.groupby('Cust ID').agg(
    Recency_Days=('Date', lambda x: (max_date - x.max()).days),
    Frequency_Orders=('Order ID', 'nunique'),
    Monetary_Value=('Amount', 'sum'),
    Total_Qty=('Qty', 'sum'),
    Age=('Age', 'mean'),
    Cancelled_Orders=('Status', lambda x: (x == 'Cancelled').sum())
).reset_index()

# Calculate Cancellation / Return Rate
customer_df['Return_Rate'] = customer_df['Cancelled_Orders'] / customer_df['Frequency_Orders']

# 2. Define Target Label (Churn / At-Risk Customer)
# Customer is high risk if Recency > 90 days or high Cancellation Rate (> 20%)
customer_df['Churn'] = ((customer_df['Recency_Days'] > 90) | (customer_df['Return_Rate'] > 0.2)).astype(int)

# Features selected for ML model
features = ['Recency_Days', 'Frequency_Orders', 'Monetary_Value', 'Total_Qty', 'Age', 'Return_Rate']

X = customer_df[features]
y = customer_df['Churn']

# 3. Train ML Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy Check
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy on Real Dataset: {accuracy * 100:.2f}%")

# 4. Save Model File
with open('ecommerce_model.pkl', 'wb') as f:
    pickle.dump(model, f)
