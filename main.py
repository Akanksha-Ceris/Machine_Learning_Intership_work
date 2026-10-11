import pandas as pd
import numpy as np

# 1. Load the Dataset
print("--- 1. Loading Dataset ---")
df = pd.read_csv("data/student_data.csv")
print(df.head())

# 2. Check for Missing Values
print("\n--- 2. Checking Missing Values ---")
print(df.isnull().sum())

# 3. Data Cleaning & Handling Missing Values
# We convert columns to numeric and fill empty spots with the median value
cols_to_fix = ['Study_Hours', 'Attendance', 'Previous_Score', 'Final_Score']
for col in cols_to_fix:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 4. Encoding Categorical Variables (Converting 'Gender' to numbers: Male=0, Female=1)
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    print("\n--- 3. Encoded Gender Column ---")
    print(df[['Name', 'Gender']].head())

# 5. Normalization / Feature Scaling (Scaling Study Hours between 0 and 1)
if 'Study_Hours' in df.columns:
    min_val = df['Study_Hours'].min()
    max_val = df['Study_Hours'].max()
    df['Study_Hours_Normalized'] = (df['Study_Hours'] - min_val) / (max_val - min_val)
    print("\n--- 4. Normalized Study Hours ---")
    print(df[['Study_Hours', 'Study_Hours_Normalized']].head())

# 6. Save the Cleaned Dataset
df.to_csv("outputs/cleaned_student_data.csv", index=False)
print("\n--- Success! Cleaned file saved to outputs/cleaned_student_data.csv ---")