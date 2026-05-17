"""
Предсказание для одного изображения
Запуск: python predict.py путь_к_фото.*
"""

import sys
import cv2
import numpy as np
import joblib
from fourier_pipeline import extract_all_features

if len(sys.argv) < 2:
    print("Использование: python predict.py image.*")
    sys.exit(1)

image_path = sys.argv[1]


model = joblib.load('../models/fourier_defect_model.pkl')

features, _ = extract_all_features(image_path, use_rings=True, use_angular=False, use_stats=True)

pred = model.predict([features])[0]
proba = model.predict_proba([features])[0]

print(f"\n  РЕЗУЛЬТАТ  ")
print(f"Изображение: {image_path}")
print(f"Дефект: {'ДА' if pred == 1 else 'НЕТ'}")
print(f"Уверенность: {proba[pred]:.2%}")