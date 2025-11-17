import pandas as pd

# Load the CSV file
df = pd.read_csv("C:\\Users\\weron\\Pulpit\\sem2\\autoML\\hyperparameter-tunability-project\\poc\\results_part2\\usage_100\\tunability_analysis.csv")

# Group by model and sampling_method, then calculate mean tunability_risk_diff
avg_tunability = df.groupby(["model", "sampling_method"])["tunability_risk_diff"].mean().reset_index()
avg_tunability.columns = ["model", "sampling_method", "avg_tunability_risk_diff"]

# Display results
print(avg_tunability)

# Optionally save to CSV
avg_tunability.to_csv("average_tunability_risk_diff.csv", index=False)


import pandas as pd

# Load the CSV file
df = pd.read_csv("c:\\Users\\weron\\Pulpit\\sem2\\autoML\\hyperparameter-tunability-project\\poc\\results_part2\\usage_100\\tunability_analysis.csv")

# Keep only the required columns
df = df[["model", "sampling_method", "dataset_id", "tunability_risk_diff"]]
df["model_sampling"] = df["model"] + "__" + df["sampling_method"]
df = df[["model_sampling", "dataset_id", "tunability_risk_diff"]]

df.to_csv("example.csv", index=False)