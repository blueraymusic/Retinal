import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load CSV
df = pd.read_csv("classification/mislabel_log.csv")

# Count per label
error_counts = Counter()

for _, row in df.iterrows():
    true = eval(row["true"])   # string → list
    pred = eval(row["pred"])
    for j, (t, p) in enumerate(zip(true, pred)):
        if t != p:
            error_counts[j] += 1

# Convert counter to DataFrame for plotting
errors_df = pd.DataFrame.from_dict(error_counts, orient="index", columns=["errors"])
errors_df = errors_df.sort_index()

# Plot
plt.figure(figsize=(10,6))
plt.bar(errors_df.index, errors_df["errors"])
plt.xlabel("Class Index")
plt.ylabel("Number of Misclassifications")
plt.title("Misclassification Frequency per Class")
plt.xticks(errors_df.index)  # show all class indices
plt.tight_layout()
plt.show()
