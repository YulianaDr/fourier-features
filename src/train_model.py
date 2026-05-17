"""
Сохранение лучшей модели (кольцевые + статистические)
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from train import load_dataset

GOOD_DIR = "../data/good"
DEFECT_DIR = "../data/defective"

X, y = load_dataset(GOOD_DIR, DEFECT_DIR, 
                     use_rings=True, use_angular=False, use_stats=True)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, '../models/fourier_defect_model.pkl')
print("Модель сохранена")