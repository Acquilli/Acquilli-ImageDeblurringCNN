import os

# Paths
DATASET_DIR = r'C:\Users\User\Desktop\dataset'
TRAIN_X_FOLDER = os.path.join(DATASET_DIR, 'train_x')
TRAIN_Y_FOLDER = os.path.join(DATASET_DIR, 'train_y')
TEST_IMAGE_PATH = os.path.join(DATASET_DIR, 'train_x', '203.jpg')

# Output folders
OUTPUT_DIR = TRAIN_Y_FOLDER  # For compatibility
MODELS_FOLDER = os.path.join(TRAIN_Y_FOLDER, 'models')
RESULTS_FOLDER = os.path.join(TRAIN_Y_FOLDER, 'results')

# Image and training parameters
TARGET_SIZE = (150, 150)
EPOCHS = 10