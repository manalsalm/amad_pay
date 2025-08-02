import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =============================================
# Configuration (Edit these paths/parameters)
# =============================================
DATASET_PATH = "C:\\Users\\manal_qckxaa\\Desktop\\amad_pay_github\\split_dataset\\"  # Root folder with train/test subfolders
RESIZE_DIM = (128, 128)  # Fixed image size
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True
}
CLASSIFIER_PARAMS = {
    "kernel": "linear",
    "C": 1.0,
    "probability": True
}

# =============================================
# Functions
# =============================================
def load_dataset(base_path):
    """Load images and labels from train/test folders."""
    data = {"train": {"images": [], "labels": []}, "test": {"images": [], "labels": []}}
    
    for split in ["train", "test"]:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Folder '{split_path}' not found!")
        
        for person_id in os.listdir(split_path):
            person_path = os.path.join(split_path, person_id)
            if not os.path.isdir(person_path):
                continue
                
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                    
                data[split]["images"].append(img)
                data[split]["labels"].append(person_id)
    
    return data

def preprocess_image(img, resize_dim):
    """Preprocess a single image."""
    img = cv2.resize(img, resize_dim)
    img = cv2.equalizeHist(img)  # Enhance contrast
    return img

def extract_hog_features(images, hog_params):
    """Extract HOG features for all images."""
    features = []
    for img in images:
        img_pp = preprocess_image(img, RESIZE_DIM)
        feat = hog(
            img_pp,
            orientations=hog_params["orientations"],
            pixels_per_cell=hog_params["pixels_per_cell"],
            cells_per_block=hog_params["cells_per_block"],
            block_norm=hog_params["block_norm"],
            transform_sqrt=hog_params["transform_sqrt"]
        )
        features.append(feat)
    return np.array(features)

# =============================================
# Main Pipeline
# =============================================
if __name__ == "__main__":
    # Step 1: Load and prepare dataset
    print("[1/4] Loading dataset...")
    dataset = load_dataset(DATASET_PATH)
    
    # Step 2: Extract HOG features
    print("[2/4] Extracting HOG features...")
    X_train = extract_hog_features(dataset["train"]["images"], HOG_PARAMS)
    X_test = extract_hog_features(dataset["test"]["images"], HOG_PARAMS)
    
    # Encode labels as integers
    le = LabelEncoder()
    y_train = le.fit_transform(dataset["train"]["labels"])
    y_test = le.transform(dataset["test"]["labels"])
    
    # Step 3: Train classifier
    print("[3/4] Training SVM...")
    clf = SVC(**CLASSIFIER_PARAMS)
    clf.fit(X_train, y_train)
    
    # Step 4: Evaluate
    print("[4/4] Evaluating...")
    y_pred = clf.predict(X_test)
    
    print("\n=== Results ===")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Optional: Save model
    # import joblib
    # joblib.dump(clf, "hog_svm_model.pkl")