import pandas as pd 
df = pd.read_csv("dataset/train.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
