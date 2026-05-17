"""
Основные шаги: загрузка фото -> Фурье -> спектр -> признаки
"""

import cv2
import numpy as np
from features import rings_features, angular_features, stats_features

def load_and_preprocess(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Файл не найден: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    square = cv2.resize(gray, target_size)
    normalized = square.astype(np.float32) / 255.0
    return normalized


def compute_spectrum(image):
    f = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f)
    magnitude = np.abs(f_shifted)
    return magnitude


def extract_all_features(image_path, use_rings=True, use_angular=True, use_stats=False, n_rings=16, n_sectors=4):
    img = load_and_preprocess(image_path)
    spectrum = compute_spectrum(img)
    
    all_features = []
    
    if use_rings:
        all_features.extend(rings_features(spectrum, n_rings))
    if use_angular:
        all_features.extend(angular_features(spectrum, n_sectors))
    if use_stats:
        all_features.extend(stats_features(spectrum))
    
    return np.array(all_features), spectrum