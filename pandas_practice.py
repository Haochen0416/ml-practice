# Pandas Data Analysis Practice
# Haochen Li - SMU Computer Engineering

import pandas as pd
import numpy as np

data = {
    'name':    ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'score':   [85, 92, 78, 95, 60],
    'hours':   [5, 8, 4, 9, 2],
    'passed':  [True, True, True, True, False]
}

df = pd.DataFrame(data)
print("=== Original Data ===")
print(df)

print("\n=== Basic Info ===")
print(df.shape)          
print(df.dtypes)         
print(df.describe())     

print("\n=== Select Column ===")
print(df['score'])

print("\n=== Select Row by condition ===")
print(df[df['score'] > 80])   


df.loc[2, 'hours'] = np.nan
df.loc[4, 'score'] = np.nan

print("\n=== Data with missing values ===")
print(df)

print("\n=== Check missing values ===")
print(df.isnull().sum())


df['hours'] = df['hours'].fillna(df['hours'].mean())
df['score'] = df['score'].fillna(df['score'].mean())

print("\n=== After filling missing values ===")
print(df)


df['efficiency'] = df['score'] / df['hours']
print("\n=== Add efficiency column ===")
print(df)


print("\n=== Sort by score descending ===")
print(df.sort_values('score', ascending=False))


print("\n=== Group by passed ===")
print(df.groupby('passed')['score'].mean())