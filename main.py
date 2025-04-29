import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import save_model, load_model

from utils import load_images_from_folder, apply_blur, process_image
from model import deblur_model
import config

# Create folders for results
os.makedirs(os.path.join(config.OUTPUT_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(config.OUTPUT_DIR, 'results'), exist_ok=True)

# Load sharp images
sharp_images = load_images_from_folder(config.TRAIN_X_FOLDER, config.TARGET_SIZE)

# Generate blurred images as ground truth
blurred_images = np.array([apply_blur(img) for img in sharp_images])

# Split the data
train_x, test_x, train_y, test_y = train_test_split(
    blurred_images, sharp_images, test_size=0.2, random_state=42
)

# Initialize the model
model = deblur_model(input_shape=(config.TARGET_SIZE[1], config.TARGET_SIZE[0], 3))
model.compile(optimizer='adam', loss='mean_squared_error')

# Training
history = model.fit(train_x, train_y, epochs=config.EPOCHS, validation_data=(test_x, test_y))

# Save the model
model.save(os.path.join(config.OUTPUT_DIR, 'models', 'deep_deblur_model.h5'))
model.save_weights(os.path.join(config.OUTPUT_DIR, 'models', 'deep_deblur.weights.h5'))

# Loss plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(config.OUTPUT_DIR, 'results', 'training_plot.png'))
plt.show()

# Test on a single image
test_image = process_image(config.TEST_IMAGE_PATH, config.TARGET_SIZE)

# Predict deblurred version
model = load_model(os.path.join(config.OUTPUT_DIR, 'models', 'deep_deblur_model.h5'))
deblurred = model.predict(np.expand_dims(apply_blur(test_image), axis=0))[0]
deblurred = (deblurred * 255).astype(np.uint8)

# Comparison plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(test_image)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(apply_blur(test_image))
plt.title('Blurred')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(deblurred)
plt.title('Deblurred')
plt.axis('off')

plt.savefig(os.path.join(config.OUTPUT_DIR, 'results', 'test_result.png'))
plt.show()
