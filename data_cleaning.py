# Data Cleaning Practice
# Haochen Li - SMU Computer Engineering

import pandas as pd
import numpy as np


data = {
    'name':   ['Alice', 'Bob', '  Charlie  ', 'Diana', 'Bob', 'Eve', 'Frank', None],
    'age':    [25, 30, 22, 999, 30, 17, 45, 28],
    'email':  ['alice@gmail.com', 'BOB@yahoo.com', 'charlie@email.com',
               'diana@gmail.com', 'BOB@yahoo.com', 'not-an-email',
               'frank@gmail.com', 'eve@gmail.com'],
    'salary': [50000, 60000, None, 75000, 60000, 45000, None, 55000],
    'join_date': ['2023-01-15', '2022-06-30', '2023-03-01',
                  '01/15/2021', '2022-06-30', '2023-07-20',
                  '2021-11-05', '2023-09-01']
}

df = pd.DataFrame(data)
print("=== Raw Dirty Data ===")
print(df)
print(f"\nShape: {df.shape}")


print("\n=== Data Quality Check ===")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")


df = df.drop_duplicates()
print(f"\n=== After removing duplicates ===")
print(f"Shape: {df.shape}")


df = df.dropna(subset=['name'])


df['salary'] = df['salary'].fillna(df['salary'].median())

print(f"\n=== After handling missing values ===")
print(f"Missing values:\n{df.isnull().sum()}")


df['name'] = df['name'].str.strip()


df['email'] = df['email'].str.lower()

print("\n=== After string cleaning ===")
print(df[['name', 'email']])


print(f"\n=== Outlier Detection ===")
print(f"Age stats:\n{df['age'].describe()}")


median_age = df[df['age'] <= 100]['age'].median()
df['age'] = df['age'].apply(lambda x: median_age if x > 100 else x)
print(f"\nAfter fixing age outliers:\n{df['age'].values}")


def is_valid_email(email):
    return '@' in str(email) and '.' in str(email).split('@')[-1]

df['email_valid'] = df['email'].apply(is_valid_email)
print(f"\n=== Email Validation ===")
print(df[['name', 'email', 'email_valid']])


df['join_date'] = pd.to_datetime(df['join_date'], format='mixed')
print(f"\n=== Standardized Dates ===")
print(df[['name', 'join_date']])


print("\n=== Final Clean Data ===")
print(df)
print(f"\nFinal shape: {df.shape}")