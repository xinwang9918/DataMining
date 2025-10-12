import pandas as pd

excelErrorsRaw = pd.read_csv("excelErrorsRaw.csv")
excelErrorsPost = pd.read_csv("excelErrorsPost.csv")

print(excelErrorsRaw.iloc[0, 3])  # Output: 5.7330971589371146e+19
print(excelErrorsPost.iloc[0, 3]) # Output: 5.7330971589371146e+19

# Load files as raw text
with open("excelErrorsRaw.csv", "r", encoding="utf-8") as f:
    raw_lines = f.read().strip().split("\n")

with open("excelErrorsPost.csv", "r", encoding="utf-8") as f:
    post_lines = f.read().strip().split("\n")

# Split into matrices
raw_matrix = [line.split(",") for line in raw_lines]
post_matrix = [line.split(",") for line in post_lines]

# Compare
diffs = []
for i in range(len(raw_matrix)):
    for j in range(len(raw_matrix[i])):
        val_raw = raw_matrix[i][j].strip()
        val_post = post_matrix[i][j].strip()
        if val_raw != val_post:
            diffs.append((i+1, j+1, val_raw, val_post))

# Output differences
print(f"Found {len(diffs)} differences:")
for row, col, v1, v2 in diffs:
    print(f"Row {row}, Column {col}:\n  Raw = {v1}\n  Post = {v2}\n")

# Compare column 6 (index 5)
changed_samples = []
for i in range(len(raw_matrix)):
    raw_val = raw_matrix[i][5].strip()
    post_val = post_matrix[i][5].strip()
    if raw_val != post_val:
        changed_samples.append((i+1, raw_val, post_val))

# Output
print(f"Number of samples changed in column 6: {len(changed_samples)}")

# Q3.2
# Compare Column 2 (index = 1)
differences = []
for i in range(len(raw_matrix)):
    raw_val = raw_matrix[i][1].strip()
    post_val = post_matrix[i][1].strip()
    if raw_val != post_val:
        differences.append((i + 1, raw_val, post_val))  # Row number is 1-indexed

# Output results
print(f"Number of samples changed in Column 2: {len(differences)}\n")

if differences:
    row, raw_val, post_val = differences[0]
    print(f"First change found at row {row}:")
    print(f"  Original (raw):  {raw_val}")
    print(f"  Changed (post): {post_val}")

# Q3.3
# Compare Column 2 (index = 1)
differences = []
for i in range(len(raw_matrix)):
    raw_val = raw_matrix[i][2].strip()
    post_val = post_matrix[i][2].strip()
    if raw_val != post_val:
        differences.append((i + 1, raw_val, post_val))  # Row number is 1-indexed

# Output results
print(f"Number of samples changed in Column 2: {len(differences)}\n")

if differences:
    row, raw_val, post_val = differences[0]
    print(f"First change found at row {row}:")
    print(f"  Original (raw):  {raw_val}")
    print(f"  Changed (post): {post_val}")

# Q3.4
# Compare Column 3 (index = 1)
differences = []
for i in range(len(raw_matrix)):
    raw_val = raw_matrix[i][3].strip()
    post_val = post_matrix[i][3].strip()
    if raw_val != post_val:
        differences.append((i + 1, raw_val, post_val))  # Row number is 1-indexed

# Output results
print(f"Number of samples changed in Column 2: {len(differences)}\n")

if differences:
    row, raw_val, post_val = differences[0]
    print(f"First change found at row {row}:")
    print(f"  Original (raw):  {raw_val}")
    print(f"  Changed (post): {post_val}")

# Q3.5
# Compare Column 5 (index = 4)
differences = []
for i in range(len(raw_matrix)):
    raw_val = raw_matrix[i][7].strip()
    post_val = post_matrix[i][7].strip()
    if raw_val != post_val:
        differences.append((i + 1, raw_val, post_val))  # Row number is 1-indexed

# Output results
print(f"Number of samples changed in Column 8: {len(differences)}\n")

if differences:
    row, raw_val, post_val = differences[0]
    print(f"First change found at row {row}:")
    print(f"  Original (raw):  {raw_val}")
    print(f"  Changed (post): {post_val}")