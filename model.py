from tensorflow.keras import layers, models

def deblur_model(input_shape=(100, 250, 3)):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')
    ])
    return model