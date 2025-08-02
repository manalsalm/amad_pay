import os
import shutil
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ================= CONFIGURATION =================
INPUT_DIR = "data"                  # Root folder with structure: Directory/Category/images/
TRAIN_TEST_DIR = "split_dataset"    # Output for train/test splits
TEST_SIZE = 0.25                    # Test set proportion
IMG_SIZE = (128, 128)               # Image dimensions
BATCH_SIZE = 32                     # Training batch size
EPOCHS = 50                         # Training epochs

# ================= FUNCTIONS =================
def build_cnn_model(input_shape, num_classes):
    """Create a CNN model for classification"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model

# ================= MAIN PIPELINE =================
if __name__ == "__main__":
    #print("=== Step 1: Processing directory structure ===")
    #process_directory_structure(INPUT_DIR, TRAIN_TEST_DIR, TEST_SIZE)
    
    print("\n=== Step 2: Preparing data generators ===")
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data should not be augmented
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(TRAIN_TEST_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        color_mode='grayscale'
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(TRAIN_TEST_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        color_mode='grayscale',
        shuffle=False
    )
    
    print("\n=== Step 3: Building CNN model ===")
    num_classes = len(train_generator.class_indices)
    model = build_cnn_model((IMG_SIZE[0], IMG_SIZE[1], 1), num_classes)
    model.summary()
    
    print("\n=== Step 4: Training model ===")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_generator,
        validation_steps=test_generator.samples // BATCH_SIZE
    )
    
    print("\n=== Step 5: Evaluation ===")
    # Get true labels and predictions
    test_generator.reset()
    y_true = test_generator.classes
    y_pred = model.predict(test_generator).argmax(axis=1)
    
    # Classification report
    label_names = list(train_generator.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))
    
    # Save model (optional)
    model.save("knuckle_palm_cnn.h5")