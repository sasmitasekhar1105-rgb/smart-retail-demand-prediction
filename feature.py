import pandas as pd 
# Load dataset
df = pd.read_csv("dataset/train.csv")

# Convert data to datetime
df['date'] = pd.to_datetime(df["date"])

#create date-based feactures
df['year'] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek

print(df.head())
print(df.columns)
# Previous_day_sales"] = df.groupby(["store", "item"])["sales"].shift(1)

#remove rows where previous sales are not available
df = df.dropna()

print(df.head())