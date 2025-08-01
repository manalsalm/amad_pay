import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

# ================= CONFIGURATION =================
DATA_DIR = "split_dataset"                # Folder with train/test subdirectories
IMG_SIZE = (224, 224)              # MobileNet's default input size
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
FREEZE_BASE = True                  # Whether to freeze MobileNet layers
CLASS_MODE = 'categorical'          # 'binary' for 2 classes

# ================= DATA PREPARATION =================
def prepare_data():
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
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        shuffle=False
    )
    
    return train_generator, test_generator

# ================= MODEL CREATION =================
def build_mobilenet_model(num_classes):
    # Load MobileNetV2 without top layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze base layers if needed
    if FREEZE_BASE:
        for layer in base_model.layers:
            layer.trainable = False
    
    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ================= MAIN PIPELINE =================
if __name__ == "__main__":
    print("=== Step 1: Preparing data generators ===")
    train_gen, test_gen = prepare_data()
    num_classes = len(train_gen.class_indices)
    print(f"Detected {num_classes} classes: {list(train_gen.class_indices.keys())}")
    
    print("\n=== Step 2: Building MobileNetV2 model ===")
    model = build_mobilenet_model(num_classes)
    model.summary()
    
    print("\n=== Step 3: Setting up callbacks ===")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    print("\n=== Step 4: Training model ===")
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=test_gen,
        validation_steps=test_gen.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    print("\n=== Step 5: Evaluation ===")
    test_gen.reset()
    y_true = test_gen.classes
    y_pred = model.predict(test_gen).argmax(axis=1)
    
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=list(train_gen.class_indices.keys())
    ))
    
    print("\n=== Step 6: Saving final model ===")
    model.save("mobilenet_knuckle_classifier.h5")
    print("Model saved as 'mobilenet_knuckle_classifier.h5'")