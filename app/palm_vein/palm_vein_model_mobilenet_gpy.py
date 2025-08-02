import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report
import numpy as np

# ================= CONFIGURATION =================
DATA_DIR = "split_dataset"          # Folder with train/test subdirectories
IMG_SIZE = (224, 224)              # MobileNet's default input size
BATCH_SIZE = 64                    # Larger batch size for GPU
EPOCHS = 30                       
INIT_LR = 0.0001                   # Lower initial learning rate
FINE_TUNE_AT = 100                 # Layer index from which to fine-tune

# ================= GPU SETUP =================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print(gpus)
# ================= DATA GENERATORS =================
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, test_generator

# ================= MOBILENET MODEL =================
def build_mobilenet_model(num_classes):
    # Load MobileNetV2 without top layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze base layers initially
    base_model.trainable = False
    
    # Build new model on top
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=INIT_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    return model, base_model

# ================= FINE-TUNING =================
def unfreeze_for_finetuning(model, base_model):
    # Unfreeze top layers of base model
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
        
    model.compile(
        optimizer=Adam(learning_rate=INIT_LR/10),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ================= MAIN PIPELINE =================
if __name__ == "__main__":
    print("=== Loading Data ===")
    train_gen, test_gen = create_data_generators()
    num_classes = len(train_gen.class_indices)
    print(f"Found {num_classes} classes: {list(train_gen.class_indices.keys())}")
    
    print("\n=== Building Initial Model ===")
    model, base_model = build_mobilenet_model(num_classes)
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(
            'mobilenet_best_weights.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    print("\n=== Phase 1: Training Top Layers ===")
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=test_gen,
        validation_steps=test_gen.samples // BATCH_SIZE,
        epochs=EPOCHS//2,  # First phase training
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n=== Phase 2: Fine-Tuning ===")
    model = unfreeze_for_finetuning(model, base_model)
    model.summary()
    
    history_fine = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=test_gen,
        validation_steps=test_gen.samples // BATCH_SIZE,
        epochs=EPOCHS,  # Second phase training
        initial_epoch=history.epoch[-1]+1,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n=== Evaluation ===")
    test_gen.reset()
    y_true = test_gen.classes
    y_pred = model.predict(test_gen).argmax(axis=1)
    
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=list(train_gen.class_indices.keys())
    ))
    
    print("\n=== Saving Model ===")
    model.save("mobilenet_knuckle_palm.h5")
    print("Model saved as 'mobilenet_knuckle_palm.h5'")