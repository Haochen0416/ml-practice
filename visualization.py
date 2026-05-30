# Data Visualization Practice
# Haochen Li - SMU Computer Engineering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
n = 100

df = pd.DataFrame({
    'hours_studied':  np.random.normal(5, 2, n).clip(1, 10),
    'previous_score': np.random.normal(70, 15, n).clip(30, 100),
    'final_score':    np.random.normal(75, 12, n).clip(30, 100),
    'passed':         np.random.choice([0, 1], n, p=[0.3, 0.7])
})

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Student Performance Analysis', fontsize=16, fontweight='bold')


ax1 = axes[0, 0]
counts = df['passed'].value_counts()
ax1.bar(['Fail', 'Pass'], counts.values, color=['#e74c3c', '#2ecc71'])
ax1.set_title('Pass vs Fail Count')
ax1.set_ylabel('Number of Students')
for i, v in enumerate(counts.values):
    ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')


ax2 = axes[0, 1]
ax2.hist(df['final_score'], bins=20, color='#3498db', edgecolor='white')
ax2.set_title('Final Score Distribution')
ax2.set_xlabel('Score')
ax2.set_ylabel('Frequency')
ax2.axvline(df['final_score'].mean(), color='red',
            linestyle='--', label=f"Mean: {df['final_score'].mean():.1f}")
ax2.legend()


ax3 = axes[1, 0]
colors = ['#e74c3c' if p == 0 else '#2ecc71' for p in df['passed']]
ax3.scatter(df['hours_studied'], df['final_score'],
            c=colors, alpha=0.6)
ax3.set_title('Hours Studied vs Final Score')
ax3.set_xlabel('Hours Studied')
ax3.set_ylabel('Final Score')


ax4 = axes[1, 1]
pass_scores = df[df['passed'] == 1]['final_score']
fail_scores = df[df['passed'] == 0]['final_score']
ax4.boxplot([fail_scores, pass_scores], labels=['Fail', 'Pass'])
ax4.set_title('Score Distribution by Result')
ax4.set_ylabel('Final Score')

plt.tight_layout()
plt.savefig('student_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved as student_analysis.png")