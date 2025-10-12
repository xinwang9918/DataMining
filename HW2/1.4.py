import string
import pandas as pd
import re


# Question 1.4 Integrate FL25
orig = pd.read_csv("Features of my lunch - Pre-processed example.csv")
wide = pd.read_csv("Features of my lunch - Fall 25.csv")

# print(orig.columns)
print(len(wide.columns))

wide_t = wide.T

letters = list(string.ascii_uppercase)
ids = []
for i in range(len(wide_t)):
    if i < 26:
        ids.append(f"FL25{letters[i]}")
    else:
        first = letters[(i // 26) - 1]
        second = letters[i % 26]
        ids.append(f"FL25{first}{second}")


# Add ID column
wide_t.insert(0, "ID", ids)
wide_t.columns = ["ID", "Price", "Temperature", "Method", "Num. Ingredients", "Weight (g)", "Ethnicity"]
wide_t["Num. Ingredients"] = wide_t["Num. Ingredients"].astype(str).apply(
    lambda x: re.sub(r"\D", "", x) if pd.notna(x) else x
)
wide_t["Price"] = wide_t["Price"].astype(str).apply(
    lambda x: re.sub(r"[^\d.]", "", x) if pd.notna(x) else x
)
wide_t["Weight (g)"] = wide_t["Weight (g)"].astype(str).apply(
    lambda x: re.sub(r"[^\d.]", "", x) if pd.notna(x) else x
)
wide_t["Method"] = wide_t["Method"].astype(str).str.replace("-", " ")
# print(wide_t)

integrated = pd.concat([orig, wide_t.head(9)], ignore_index=True)
print(integrated)

integrated.to_csv("1.4_Features_of_my_lunch_INTEGRATED.csv", index=False)