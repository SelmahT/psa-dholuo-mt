import pandas as pd

df = pd.read_csv("data/processed/psa_dataset_ekegusii.csv")
df["Domain"] = df["Domain"].replace({"Security & Safety": "Security"})
df.to_csv("data/processed/psa_dataset_ekegusii.csv", index=False)
print(df["Domain"].value_counts())