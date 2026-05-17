"""
Подбор оптимального числа секторов
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

sectors_options = [2, 4, 8, 16, 32, 64, 128]
results = {}

for n_sectors in sectors_options:
    print(f"\nТестирование {n_sectors} секторов...")
    
    X, y = load_dataset(GOOD_DIR, DEFECT_DIR, 
                        use_rings=False, use_angular=True, use_stats=False,
                        n_sectors=n_sectors)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[n_sectors] = acc
    print(f"  Точность: {acc:.2%}")

print("\n     ОПТИМАЛЬНОЕ ЧИСЛО СЕКТОРОВ     ")
for n, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{n} секторов: {acc:.2%}")

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(list(results.keys()), list(results.values()), 'o-', linewidth=2, markersize=10)
plt.xlabel('Число секторов')
plt.ylabel('Точность')
plt.title('Зависимость точности от числа секторов (угловые признаки)')
plt.grid(True, alpha=0.3)
plt.xticks(sectors_options)
plt.tight_layout()
plt.savefig('../results/sectors_optimization.png', dpi=150)
plt.show()