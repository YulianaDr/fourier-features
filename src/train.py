"""
Обучение классификатора и сравнение всех комбинаций признаков
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

from fourier_pipeline import extract_all_features

def load_dataset(data_dir_good, data_dir_defective, use_rings=True, use_angular=True, use_stats=False, n_rings=16, n_sectors=4):
    features = []
    labels = []
    
    for filename in os.listdir(data_dir_good):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(data_dir_good, filename)
            try:
                feats, _ = extract_all_features(path, use_rings, use_angular, use_stats, n_rings=n_rings, n_sectors=n_sectors)
                features.append(feats)
                labels.append(0)
            except Exception as e:
                print(f"Ошибка {path}: {e}")
    
    for filename in os.listdir(data_dir_defective):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(data_dir_defective, filename)
            try:
                feats, _ = extract_all_features(path, use_rings, use_angular, use_stats, n_rings=n_rings, n_sectors=n_sectors)
                features.append(feats)
                labels.append(1)
            except Exception as e:
                print(f"Ошибка {path}: {e}")
    
    return np.array(features), np.array(labels)

def evaluate_combination(features, labels, name="Комбинация"):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = (time.time() - start_time) / len(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{name}:")
    print(f"  Признаков: {features.shape[1]}")
    print(f"  Точность: {acc:.2%}")
    print(f"  Время обучения: {train_time:.2f} с")
    print(f"  Время предсказания (1 фото): {predict_time:.6f} с")
    
    return acc


if __name__ == "__main__":
    GOOD_DIR = "../data/good"
    DEFECTIVE_DIR = "../data/defective"
    
    if not os.path.exists(GOOD_DIR):
        print(f"Ошибка: папка {GOOD_DIR} не найдена")
        exit()
    if not os.path.exists(DEFECTIVE_DIR):
        print(f"Ошибка: папка {DEFECTIVE_DIR} не найдена")
        exit()
    
    print("     ЗАГРУЗКА ДАТАСЕТА     ")
    
    X, y = load_dataset(GOOD_DIR, DEFECTIVE_DIR, use_rings=True, use_angular=False, use_stats=False)
    acc1 = evaluate_combination(X, y, "Только кольцевые (16)")
    
    X, y = load_dataset(GOOD_DIR, DEFECTIVE_DIR, use_rings=False, use_angular=True, use_stats=False)
    acc2 = evaluate_combination(X, y, "Только угловые (4)")
    
    X, y = load_dataset(GOOD_DIR, DEFECTIVE_DIR, use_rings=False, use_angular=False, use_stats=True)
    acc3 = evaluate_combination(X, y, "Только статистические (4)")
    
    X, y = load_dataset(GOOD_DIR, DEFECTIVE_DIR, use_rings=True, use_angular=True, use_stats=False)
    acc4 = evaluate_combination(X, y, "Кольцевые + угловые (20)")
    
    X, y = load_dataset(GOOD_DIR, DEFECTIVE_DIR, use_rings=True, use_angular=False, use_stats=True)
    acc5 = evaluate_combination(X, y, "Кольцевые + статистические (20)")
    
    X, y = load_dataset(GOOD_DIR, DEFECTIVE_DIR, use_rings=False, use_angular=True, use_stats=True)
    acc6 = evaluate_combination(X, y, "Угловые + статистические (8)")
    
    X, y = load_dataset(GOOD_DIR, DEFECTIVE_DIR, use_rings=True, use_angular=True, use_stats=True)
    acc7 = evaluate_combination(X, y, "Все вместе (24)")
    
    print("\n" + "="*50)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*50)
    
    results = [
        ("Только кольцевые (16)", acc1),
        ("Только угловые (4)", acc2),
        ("Только статистические (4)", acc3),
        ("Кольцевые + угловые (20)", acc4),
        ("Кольцевые + статистические (20)", acc5),
        ("Угловые + статистические (8)", acc6),
        ("Все вместе (24)", acc7),
    ]
    
    for name, acc in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:35} : {acc:.2%}")