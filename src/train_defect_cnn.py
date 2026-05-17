"""
Сравнение Фурье-признаков (кольцевые + стастистические) + RF с CNN на датасете дефектов
"""

import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from fourier_pipeline import extract_all_features


def load_defect_dataset(good_dir, defect_dir, img_size=(128, 128)):
    images = []
    labels = []
    
    for fname in os.listdir(good_dir):
        if fname.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(good_dir, fname)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(0)
    
    for fname in os.listdir(defect_dir):
        if fname.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(defect_dir, fname)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(1)
    
    return np.array(images), np.array(labels)


def create_cnn(input_shape=(128, 128, 1)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    GOOD_DIR = "../data/good"
    DEFECT_DIR = "../data/defective"
    
    print("Загрузка датасета дефектов...")
    X, y = load_defect_dataset(GOOD_DIR, DEFECT_DIR)
    print(f"Загружено {len(X)} изображений, размер {X[0].shape}")
    
    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("\n   ФУРЬЕ + RANDOM FOREST   ")
    start = time.time()
    X_train_fourier = []
    X_test_fourier = []
    
    for img in X_train:
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, img)
        feats, _ = extract_all_features(temp_path, use_rings=True, use_angular=False, use_stats=True)
        X_train_fourier.append(feats)
        os.remove(temp_path)
    
    for img in X_test:
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, img)
        feats, _ = extract_all_features(temp_path, use_rings=True, use_angular=False, use_stats=True)
        X_test_fourier.append(feats)
        os.remove(temp_path)
    
    X_train_fourier = np.array(X_train_fourier)
    X_test_fourier = np.array(X_test_fourier)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_fourier, y_train)
    
    start_pred = time.time()
    y_pred_rf = rf.predict(X_test_fourier)
    pred_time_rf = (time.time() - start_pred) / len(X_test_fourier)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Время предсказания (1 фото) RF: {pred_time_rf:.6f} с")
    print(f"Точность: {acc_rf:.2%}")
    print(f"Время (включая признаки): {time.time()-start:.2f} с")
    
    print("\n   CNN    ")
    X_train_cnn = X_train.reshape(-1, 128, 128, 1).astype(np.float32) / 255.0
    X_test_cnn = X_test.reshape(-1, 128, 128, 1).astype(np.float32) / 255.0
    
    model = create_cnn()
    start = time.time()
    history = model.fit(X_train_cnn, y_train, epochs=10, batch_size=16,
                        validation_data=(X_test_cnn, y_test), verbose=0)
    cnn_time = time.time() - start
    cnn_acc = history.history['val_accuracy'][-1]
    print(f"Точность: {cnn_acc:.2%}")
    print(f"Время обучения: {cnn_time:.2f} с")
    
    start_pred = time.time()
    y_pred_cnn = model.predict(X_test_cnn, verbose=0)
    pred_time_cnn = (time.time() - start_pred) / len(X_test_cnn)
    print(f"Время предсказания (1 фото) CNN: {pred_time_cnn:.6f} с")
    
    print("\n   СРАВНЕНИЕ   ")
    print(f"Фурье + Random Forest: {acc_rf:.2%}")
    print(f"CNN:                   {cnn_acc:.2%}")