import os
import cv2
import numpy as np

def process_image(path, target_size):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return img

def apply_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def load_images_from_folder(folder, target_size):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            img = process_image(path, target_size)
            if img is not None:
                images.append(img)
    return np.array(images)
