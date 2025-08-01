import os
import shutil
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ================= CONFIGURATION =================
INPUT_DIR =  'C:\\Users\\manal_qckxaa\\Desktop\\amad_pay_github\\multimodal pp and pv\\DB_Vein\\'                 # Root folder with structure: Directory/Category/images/
TRAIN_TEST_DIR = "split_dataset"    # Output for train/test splits
TEST_SIZE = 0.25                    # Test set proportion
RESIZE_DIM = (128, 128)             # Standard image size

# HOG Parameters
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "transform_sqrt": True
}

# ================= FUNCTIONS =================
def process_directory_structure(input_dir, output_dir, test_size):
    """Process Directory/Category/images structure and create train/test splits"""
    categories = set()
    
    # First pass: Discover all unique categories
    for root, dirs, files in os.walk(input_dir):
        category = os.path.basename(root)
        categories.add(category)
    
    # Create train/test folders for each category
    for split in ["train", "test"]:
        for category in categories:
            os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)
    
    # Second pass: Copy images to train/test
    for root, dirs, files in os.walk(input_dir):
        for cat in categories:
            if cat == '':
                continue
            
            #category = os.path.basename(root)
            image_dir = os.path.join(root, cat)
            if not os.path.isdir(image_dir):
                continue
            # Get all image files
            image_files = [f for f in os.listdir(image_dir) 
                        if f.lower().endswith(('.jpg', '.png', '.jpeg','.bmp'))]
            
            # Split into train/test
            train_files, test_files = train_test_split(
                image_files, test_size=test_size, random_state=42
            )
            
            # Copy files
            for f in train_files:
                src = os.path.join(image_dir, f)
                dst = os.path.join(output_dir, "train", cat, f)
                shutil.copy2(src, dst)
                
            for f in test_files:
                src = os.path.join(image_dir, f)
                dst = os.path.join(output_dir, "test", cat, f)
                shutil.copy2(src, dst)
            
            print(f"{cat}: {len(train_files)} train, {len(test_files)} test")

def extract_hog_features(folder):
    """Extract HOG features from organized train/test folders"""
    features, labels = [], []
    
    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)
        if not os.path.isdir(category_path):
            continue
            
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Preprocessing
            img = cv2.resize(img, RESIZE_DIM)
            img = cv2.equalizeHist(img)
            
            # HOG Feature Extraction
            hog_features = hog(img, **HOG_PARAMS)
            
            features.append(hog_features)
            labels.append(category)
    
    return np.array(features), np.array(labels)

# ================= MAIN PIPELINE =================
if __name__ == "__main__":
    print("=== Step 1: Processing directory structure ===")
    process_directory_structure(INPUT_DIR, TRAIN_TEST_DIR, TEST_SIZE)
    
    print("\n=== Step 2: Extracting HOG features ===")
    X_train, y_train = extract_hog_features(os.path.join(TRAIN_TEST_DIR, "train"))
    X_test, y_test = extract_hog_features(os.path.join(TRAIN_TEST_DIR, "test"))
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    print("\n=== Step 3: Training SVM Classifier ===")
    clf = SVC(kernel='linear', C=1.0, probability=True)
    clf.fit(X_train, y_train_encoded)
    
    print("\n=== Step 4: Evaluation ===")
    y_pred = clf.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred, 
                              target_names=le.classes_))
    
    print(f"\nAccuracy: {accuracy_score(y_test_encoded, y_pred):.2%}")
    
    # Save model (optional)
    import joblib
    joblib.dump((clf, le), "hog_svm_model.pkl")