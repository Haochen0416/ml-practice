 # Scikit-learn Complete ML Pipeline
# Haochen Li - SMU Computer Engineering

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


np.random.seed(42)
n = 100

hours   = np.random.normal(5, 2, n).clip(1, 10)
prev    = np.random.normal(70, 15, n).clip(30, 100)
passed  = ((hours * 5 + prev * 0.5) > 70).astype(int)

df = pd.DataFrame({
    'hours_studied': hours,
    'previous_score': prev,
    'passed': passed
})

print("=== Dataset ===")
print(df.head())
print(f"\nPass rate: {passed.mean():.1%}")


X = df[['hours_studied', 'previous_score']]
y = df['passed']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n=== Data Split ===")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print("\n=== Model Trained ===")


y_pred = model.predict(X_test_scaled)

print(f"\n=== Model Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['Fail', 'Pass']))


print("=== Predict New Students ===")
new_students = pd.DataFrame({
    'hours_studied':  [2, 5, 9],
    'previous_score': [50, 70, 90]
})
new_scaled = scaler.transform(new_students)
predictions = model.predict(new_scaled)
proba       = model.predict_proba(new_scaled)

for i, (pred, prob) in enumerate(zip(predictions, proba)):
    result = "PASS" if pred == 1 else "FAIL"
    print(f"Student {i+1} "
          f"(hours={new_students.iloc[i,0]}, "
          f"prev={new_students.iloc[i,1]}): "
          f"{result} (confidence: {max(prob):.1%})")