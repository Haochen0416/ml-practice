# Machine Learning Practice
# Haochen Li - SMU Computer Engineering

import random

# Simple linear regression from scratch
def mean(values):
    return sum(values) / len(values)

def variance(values):
    m = mean(values)
    return sum((x - m) ** 2 for x in values)

def covariance(x, y):
    m_x = mean(x)
    m_y = mean(y)
    return sum((x[i] - m_x) * (y[i] - m_y) for i in range(len(x)))

def linear_regression(x, y):
    b1 = covariance(x, y) / variance(x)
    b0 = mean(y) - b1 * mean(x)
    return b0, b1

# Test data: hours studied vs exam score
hours = [1, 2, 3, 4, 5, 6, 7, 8]
scores = [50, 55, 65, 70, 75, 80, 85, 95]

b0, b1 = linear_regression(hours, scores)
print(f"Linear Regression Model:")
print(f"Intercept (b0): {b0:.2f}")
print(f"Slope (b1): {b1:.2f}")
print(f"Prediction for 9 hours studied: {b0 + b1 * 9:.2f}")