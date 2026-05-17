"""
Функции для вычисления признаков
"""

import numpy as np

def rings_features(spectrum, n_rings=16):
    """
    Кольцевые признаки
    """
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
    """
    Угловые признаки
    """
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
    """
    Статистические признаки
    """
    return np.array([
        np.mean(spectrum),      
        np.std(spectrum),       
        np.max(spectrum),       
        np.sum(spectrum ** 2)  
    ])