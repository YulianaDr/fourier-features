"""
Эксперимент на датасете MNIST: тестирование всех комбинаций признаков Фурье.
Оптимизированная версия: признаки вычисляются один раз.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import time
import cv2
from fourier_pipeline import compute_spectrum


# ============================================================
# ФУНКЦИИ ДЛЯ ПРИЗНАКОВ
# ============================================================

def rings_features(spectrum, n_rings=16):
    """Кольцевые признаки."""
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    distances = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    max_radius = min(cy, cx)
    features = []
    
    for i in range(n_rings):
        r1 = i * (max_radius / n_rings)
        r2 = (i + 1) * (max_radius / n_rings)
        mask = (distances >= r1) & (distances < r2)
        
        if np.any(mask):
            features.append(np.mean(spectrum[mask]))
        else:
            features.append(0.0)
    
    return np.array(features)


def angular_features(spectrum, n_sectors=4):
    """Угловые признаки."""
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    angles = np.arctan2(y - cy, x - cx)
    
    features = []
    for i in range(n_sectors):
        start_angle = i * (2 * np.pi / n_sectors)
        end_angle = (i + 1) * (2 * np.pi / n_sectors)
        mask = (angles >= start_angle) & (angles < end_angle)
        
        if np.any(mask):
            features.append(np.mean(spectrum[mask]))
        else:
            features.append(0.0)
    
    return np.array(features)


def stats_features(spectrum):
    """Статистические признаки."""
    return np.array([
        np.mean(spectrum),
        np.std(spectrum),
        np.max(spectrum),
        np.sum(spectrum ** 2)
    ])


def extract_all_features_from_image(img, n_rings=16, n_sectors=4):
    """
    Извлекает ВСЕ признаки из одного изображения.
    Возвращает кортеж: (кольцевые, угловые, статистические)
    """
    # Приводим к правильному формату
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Увеличиваем до 64×64
    img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img_norm = img_resized.astype(np.float32) / 255.0
    
    # Фурье
    spectrum = compute_spectrum(img_norm)
    
    rings = rings_features(spectrum, n_rings)
    angular = angular_features(spectrum, n_sectors)
    stats = stats_features(spectrum)
    
    return rings, angular, stats


def load_mnist_all_features(n_samples=2000, n_rings=16, n_sectors=4):
    """
    Загружает MNIST и вычисляет ВСЕ признаки один раз.
    Возвращает словарь с признаками и метки.
    """
    print(f"Загрузка MNIST ({n_samples} изображений)...")
    mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
    X, y = mnist.data[:n_samples], mnist.target[:n_samples].astype(int)
    X_images = X.reshape(-1, 28, 28)
    
    print("Вычисление признаков (один раз)...")
    
    all_rings = []
    all_angular = []
    all_stats = []
    
    total = len(X_images)
    for i, img in enumerate(X_images):
        rings, angular, stats = extract_all_features_from_image(img, n_rings, n_sectors)
        all_rings.append(rings)
        all_angular.append(angular)
        all_stats.append(stats)
        
        if (i + 1) % 500 == 0:
            print(f"  Обработано {i+1} из {total}")
    
    return {
        'rings': np.array(all_rings),
        'angular': np.array(all_angular),
        'stats': np.array(all_stats),
        'labels': y
    }


def evaluate_combination(name, X, y, test_size=0.3):
    """Обучает Random Forest и возвращает точность."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"{name:35} : {acc:.2%}")
    return acc


if __name__ == "__main__":
    print("="*60)
    print("ЭКСПЕРИМЕНТ НА MNIST (оптимизированная версия)")
    print("="*60)
    
    # Параметры: 2000 изображений для скорости (можно увеличить)
    N_SAMPLES = 2000
    N_RINGS = 16
    N_SECTORS = 4
    
    print(f"\nПараметры:")
    print(f"  Изображений: {N_SAMPLES}")
    print(f"  Колец: {N_RINGS}")
    print(f"  Секторов: {N_SECTORS}")
    
    # 1. Вычисляем ВСЕ признаки один раз
    data = load_mnist_all_features(N_SAMPLES, N_RINGS, N_SECTORS)
    
    # 2. Тестируем комбинации (без повторного вычисления признаков)
    print("\n=== РЕЗУЛЬТАТЫ НА MNIST ===")
    
    results = {}
    
    # Только кольцевые
    results["Только кольцевые (16)"] = evaluate_combination(
        "Только кольцевые (16)", data['rings'], data['labels']
    )
    
    # Только угловые
    results["Только угловые (4)"] = evaluate_combination(
        "Только угловые (4)", data['angular'], data['labels']
    )
    
    # Только статистические
    results["Только статистические (4)"] = evaluate_combination(
        "Только статистические (4)", data['stats'], data['labels']
    )
    
    # Кольцевые + угловые
    X_rings_angular = np.hstack([data['rings'], data['angular']])
    results["Кольцевые + угловые (20)"] = evaluate_combination(
        "Кольцевые + угловые (20)", X_rings_angular, data['labels']
    )
    
    # Кольцевые + статистические
    X_rings_stats = np.hstack([data['rings'], data['stats']])
    results["Кольцевые + статистические (20)"] = evaluate_combination(
        "Кольцевые + статистические (20)", X_rings_stats, data['labels']
    )
    
    # Угловые + статистические
    X_angular_stats = np.hstack([data['angular'], data['stats']])
    results["Угловые + статистические (8)"] = evaluate_combination(
        "Угловые + статистические (8)", X_angular_stats, data['labels']
    )
    
    # Все вместе
    X_all = np.hstack([data['rings'], data['angular'], data['stats']])
    results["Все вместе (24)"] = evaluate_combination(
        "Все вместе (24)", X_all, data['labels']
    )
    
    # ИТОГОВАЯ ТАБЛИЦА
    print("\n" + "="*60)
    print("ИТОГОВОЕ СРАВНЕНИЕ НА MNIST (отсортировано по точности)")
    print("="*60)
    
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:35} : {acc:.2%}")