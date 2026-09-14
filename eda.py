import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/train.csv")

# Calculate total sales for each item
item_sales = df.groupby("item")["sales"].sum()

# Plot item-wise sales
plt.figure(figsize=(12, 5))
item_sales.plot(kind="bar")
plt.xlabel("Item")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.show()