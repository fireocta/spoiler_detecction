import pandas as pd

# Load dataset (Assume 'review' column has text and 'label' is 1 for spoilers, 0 for non-spoilers)
df = pd.read_csv("spoiler_dataset.csv")
df.head()
