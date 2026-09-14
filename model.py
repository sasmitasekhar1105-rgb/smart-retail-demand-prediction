import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("dataset/train.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Date features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek

# Sort data
df = df.sort_values(["store", "item", "date"])

# Previous day's sales
df["previous_day_sales"] = df.groupby(
    ["store", "item"]
)["sales"].shift(1)

# Remove missing values
df = df.dropna()

# Features
X = df[
    ["store", "item", "year", "month", "day",
     "day_of_week", "previous_day_sales"]
]

# Target
y = df["sales"]

# Use only 200,000 rows for faster training
X = X.iloc[:200000]
y = y.iloc[:200000]

# Train-test split
split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Faster Random Forest
model = RandomForestRegressor(
    n_estimators=30,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)

print("Model trained successfully!")
print("Mean Absolute Error:", mae)
import joblib

joblib.dump(model, "demand_model.pkl")

print("model saved successfully!")
