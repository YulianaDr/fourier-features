from fourier_pipeline import extract_all_features
from visualize import show_spectrum_with_rings, plot_features

# Укажи путь к своему фото
features, spectrum = extract_all_features("../data/test.png")

print("Показываем спектр с кольцами...")
show_spectrum_with_rings(spectrum, n_rings=16)

print("Показываем график признаков...")
plot_features(features, title="Признаки изображения")