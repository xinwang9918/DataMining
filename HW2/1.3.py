import pandas as pd

# Load your dataset
df = pd.read_csv("Features of my lunch - Pre-processed example.csv")

# Treat "-" as missing
df = df.replace("-", pd.NA)

# Question 1.3
# Fill missing values in "Temperature" with the mode
temp_mode = df["Temperature"].mode()[0]
df["Temperature"] = df["Temperature"].fillna(temp_mode)

# One-hot encode "Temperature"
df_encoded = pd.get_dummies(df, columns=["Temperature"], prefix="Temp" , dtype=int)

# Save the new dataset to CSV
df_encoded.to_csv("1.3_Temperature_encoded.csv", index=False)

print("Missing values replaced with:", temp_mode)
print("New file saved: Features_of_my_lunch_Temperature_encoded.csv")


