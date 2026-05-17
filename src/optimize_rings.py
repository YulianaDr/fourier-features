"""
Подбор оптимального числа колец
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from train import load_dataset

GOOD_DIR = "../data/good"
DEFECT_DIR = "../data/defective"

rings_options = [2, 4, 8, 16, 32, 64, 128]
results = {}

for n_rings in rings_options:
    print(f"\nТестирование {n_rings} колец...")
    
    X, y = load_dataset(GOOD_DIR, DEFECT_DIR, 
                        use_rings=True, use_angular=False, use_stats=False,
                        n_rings=n_rings)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[n_rings] = acc
    print(f"  Точность: {acc:.2%}")

print("\n     ОПТИМАЛЬНОЕ ЧИСЛО КОЛЕЦ     ")
for n, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{n} колец: {acc:.2%}")

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(list(results.keys()), list(results.values()), 'o-', linewidth=2, markersize=10)
plt.xlabel('Число колец')
plt.ylabel('Точность')
plt.title('Зависимость точности от числа колец')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/rings_optimization.png', dpi=150)
plt.show()