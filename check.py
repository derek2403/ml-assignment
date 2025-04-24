import pandas as pd

# Load the data
df = pd.read_csv("data.csv")

# Get the min and max of each feature
feature_ranges = pd.DataFrame({
    'Low': df.min(),
    'High': df.max()
})

print(feature_ranges)