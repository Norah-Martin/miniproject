import tensorflow as tf
from tensorflow import keras
from keras import layers
import os

input_shape = (224, 224, 3)

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

model_dir = "C:\Mini - Project\Drowsiness_detection\dataset\model"
os.makedirs(model_dir, exist_ok=True)

# Set the paths to the training and validation directories
train_dir = "C:\Mini - Project\Drowsiness_detection\dataset/train"
val_dir = "C:\Mini - Project\Drowsiness_detection\dataset/val"

# Set the batch size and number of epochs
batch_size = 32
epochs = 10

# Create data generators for training and validation sets
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

# Train the model on the data generators
model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Save the trained model weights to disk
model.save_weights(os.path.join(model_dir, "model_weights.h5"))

