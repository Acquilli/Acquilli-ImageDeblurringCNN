import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
from tensorflow.keras.callbacks import EarlyStopping

import config
from utils import load_images_from_folder, apply_blur, process_image, save_image
from model import deblur_model

# Ensure output folders exist
os.makedirs(config.MODELS_FOLDER, exist_ok=True)
os.makedirs(config.RESULTS_FOLDER, exist_ok=True)
os.makedirs(config.TRAIN_Y_FOLDER, exist_ok=True)

# Load sharp images from train_x
sharp_images = load_images_from_folder(config.TRAIN_X_FOLDER, config.TARGET_SIZE)

# Generate and save blurred images to train_y
for idx, img in enumerate(sharp_images):
    blurred = apply_blur(img)
    filename = f"{idx:04d}.jpg"
    path = os.path.join(config.TRAIN_Y_FOLDER, filename)
    cv2.imwrite(path, cv2.cvtColor((blurred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# Load blurred images from train_y
blurred_images = load_images_from_folder(config.TRAIN_Y_FOLDER, config.TARGET_SIZE)

# Split into train/test sets
train_x, test_x, train_y, test_y = train_test_split(sharp_images, blurred_images, test_size=0.2, random_state=42)

# Build and compile model
model = deblur_model(input_shape=(config.TARGET_SIZE[1], config.TARGET_SIZE[0], 3))
model.compile(optimizer='adam', loss='mean_squared_error')

# Add EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_x, train_y,
    epochs=config.EPOCHS,
    validation_data=(test_x, test_y),
    callbacks=[early_stopping]
)

# Save model and weights
model.save(os.path.join(config.MODELS_FOLDER, 'deep_deblur_model.h5'))
model.save_weights(os.path.join(config.MODELS_FOLDER, 'deep_deblur.weights.h5'))

# Save training loss plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(config.RESULTS_FOLDER, 'training_plot.png'))
plt.show()

# Test on one image
test_image = process_image(config.TEST_IMAGE_PATH, config.TARGET_SIZE)
blurred_test = apply_blur(test_image)
deblurred = model.predict(np.expand_dims(test_image, axis=0))[0]

# Konwersja wszystkich obrazów do uint8 dla poprawnego wyświetlania
test_image_disp = (test_image * 255).astype(np.uint8)
blurred_disp = (blurred_test * 255).astype(np.uint8)
deblurred_disp = (deblurred * 255).astype(np.uint8)

# Plot original, blurred, and deblurred images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(test_image_disp)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(blurred_disp)
plt.title('Blurred Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(deblurred_disp)
plt.title('Deblurred Output')
plt.axis('off')

plt.savefig(os.path.join(config.RESULTS_FOLDER, 'test_result.png'))
save_image(test_image, os.path.join(config.RESULTS_FOLDER, 'original.jpg'))
save_image(blurred_test, os.path.join(config.RESULTS_FOLDER, 'blurred.jpg'))
save_image(deblurred, os.path.join(config.RESULTS_FOLDER, 'deblurred.jpg'))
plt.show()