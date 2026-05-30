# Titanic Survival Prediction
# Haochen Li - SMU Computer Engineering

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500

df = pd.DataFrame({
    'pclass':   np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
    'age':      np.random.normal(30, 12, n).clip(1, 80),
    'fare':     np.random.exponential(30, n).clip(5, 200),
    'sex':      np.random.choice([0, 1], n),  # 0=male, 1=female
    'sibsp':    np.random.choice([0, 1, 2, 3], n, p=[0.6, 0.2, 0.1, 0.1]),
})


survival_prob = (
    0.2 +
    df['sex'] * 0.4 +
    (df['pclass'] == 1) * 0.2 +
    (df['age'] < 15) * 0.2
).clip(0, 1)

df['survived'] = (np.random.random(n) < survival_prob).astype(int)

print("=== Titanic Dataset ===")
print(df.head())
print(f"\nSurvival rate: {df['survived'].mean():.1%}")


df.loc[np.random.choice(n, 30), 'age'] = np.nan
df['age'] = df['age'].fillna(df['age'].median())
print(f"\nMissing values after cleaning: {df.isnull().sum().sum()}")


df['family_size'] = df['sibsp'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)

X = df[['pclass', 'age', 'fare', 'sex', 'family_size', 'is_alone']]
y = df['survived']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred,
                                 target_names=['Died', 'Survived']))


rf_model = models['Random Forest']
feature_names = X.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.bar(range(len(feature_names)),
        importances[indices], color='#3498db')
plt.xticks(range(len(feature_names)),
           [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("\nChart saved as feature_importance.png")