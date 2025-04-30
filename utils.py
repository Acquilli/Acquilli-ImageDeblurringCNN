import os
import cv2
import numpy as np

def process_image(image_path, target_size):
    """Wczytuje, zmienia rozmiar i normalizuje obraz."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Nie można wczytać obrazu: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img

def load_images_from_folder(folder, target_size):
    """Wczytuje i przetwarza wszystkie obrazy z folderu."""
    images = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        try:
            img = process_image(path, target_size)
            images.append(img)
        except Exception as e:
            print(f"[Błąd] Pominięto {filename}: {e}")
    return np.array(images)

def apply_blur(image, kernel_size=(7, 7)):
    """Zastosowuje rozmycie Gaussa do obrazu."""
    image_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(image_bgr, kernel_size, 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB) / 255.0

def save_image(image, path):
    """Zapisuje obraz RGB [0,1] jako plik JPG."""
    image_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image_bgr)
