import cv2
import numpy as np
import joblib
from skimage.feature import hog
from io import BytesIO
from PIL import Image
import numpy as np
from PIL import Image
from skimage import exposure
import matplotlib.pyplot as plt

arabic_names = [
    ("Ahmed", "Al-Mansour"),
    ("Mohammed", "Al-Farsi"),
    ("Ali", "Al-Hashimi"),
    ("Omar", "Al-Qurashi"),
    ("Khalid", "Al-Baghdadi"),
    ("Majed", "Al-Khalifa"),
    ("Youssef", "Al-Najjar"),
    ("Ibrahim", "Al-Tikriti"),
    ("Abdullah", "Al-Saud"),
    ("Saeed", "Al-Ghamdi"),
    ("Nasser", "Al-Dosari"),
    ("Badr", "Al-Harbi"),
    ("Rashed", "Al-Otaibi"),
    ("Tariq", "Al-Juhani"),
    ("Waseem", "Al-Shamrani"),
    ("Momen", "Al-Zahrani"),
    ("Hamad", "Al-Qahtani"),
    ("Jamal", "Al-Ghamdi"),
    ("Anas", "Al-Shahrani"),
    ("Fares", "Al-Balawi"),
    ("Ziyad", "Al-Sulaimani"),
    ("Basem", "Al-Amri"),
    ("Mazen", "Al-Harbi"),
    ("Saleem", "Al-Zahrani"),
    ("Adel", "Al-Omari"),
    ("Kareem", "Al-Jabri"),
    ("Haytham", "Al-Maliki"),
    ("Rami", "Al-Asmari"),
    ("Waleed", "Al-Rashidi"),
    ("Munther", "Al-Shammari"),
    ("Ghassan", "Al-Qurashi"),
    ("Saleh", "Al-Tamimi"),
    ("Othman", "Al-Nuaimi"),
    ("Moaz", "Al-Dossary"),
    ("Saad", "Al-Shammari"),
    ("Maher", "Al-Ghamdi"),
    ("Nader", "Al-Harbi"),
    ("Hani", "Al-Zahrani"),
    ("Issa", "Al-Masri"),
    ("Yahya", "Al-Qurashi"),
    ("Osama", "Al-Shahrani"),
    ("Akram", "Al-Otaibi"),
    ("Jalal", "Al-Maliki"),
    ("Sufyan", "Al-Ghamdi"),
    ("Ammar", "Al-Zahrani"),
    ("Fahad", "Al-Shammari"),
    ("Hassan", "Al-Qahtani"),
    ("Marwan", "Al-Harbi"),
    ("Salman", "Al-Sulaimani"),
    ("Hamza", "Al-Shamrani"),
    ("Fatima", "Al-Hashimi"),
    ("Aisha", "Al-Qurashi"),
    ("Maryam", "Al-Najjar"),
    ("Sarah", "Al-Tikriti"),
    ("Noura", "Al-Saud"),
    ("Lina", "Al-Ghamdi"),
    ("Reem", "Al-Dosari"),
    ("Yasmin", "Al-Harbi"),
    ("Dania", "Al-Otaibi"),
    ("Lama", "Al-Juhani"),
    ("Joud", "Al-Shamrani"),
    ("Aya", "Al-Zahrani"),
    ("Noor", "Al-Qahtani"),
    ("Rawan", "Al-Ghamdi"),
    ("Saja", "Al-Shahrani"),
    ("Malak", "Al-Balawi"),
    ("Huda", "Al-Sulaimani"),
    ("Zainab", "Al-Amri"),
    ("Shaimaa", "Al-Harbi"),
    ("Arwa", "Al-Zahrani"),
    ("Batool", "Al-Omari"),
    ("Hoor", "Al-Jabri"),
    ("Raghad", "Al-Maliki"),
    ("Mayar", "Al-Asmari"),
    ("Jana", "Al-Rashidi"),
    ("Marah", "Al-Shammari"),
    ("Salwa", "Al-Qurashi"),
    ("Doha", "Al-Tamimi"),
    ("Iman", "Al-Nuaimi"),
    ("Amal", "Al-Dossary"),
    ("Inas", "Al-Shammari"),
    ("Ghada", "Al-Ghamdi"),
    ("Hiba", "Al-Harbi"),
    ("May", "Al-Zahrani"),
    ("Nadeen", "Al-Masri"),
    ("Ritaj", "Al-Qurashi"),
    ("Tala", "Al-Shahrani"),
    ("Wijdan", "Al-Otaibi"),
    ("Sadan", "Al-Maliki"),
    ("Mays", "Al-Ghamdi"),
    ("Renad", "Al-Zahrani"),
    ("Lujain", "Al-Shammari"),
    ("Judy", "Al-Qahtani"),
    ("Mayar", "Al-Harbi"),
    ("Layan", "Al-Sulaimani"),
    ("Roz", "Al-Amri"),
    ("Miral", "Al-Harbi"),
    ("Rama", "Al-Zahrani"),
    ("Selene", "Al-Omari"),
    ("Yara", "Al-Jabri")
]
# Load the pre-trained model (do this once at startup)

model =""
loaded_object =""
try:
    model = joblib.load("C:\\Users\\manal_qckxaa\\Desktop\\amad_pay_github\\hog_svm_model.pkl")
    # Load the file and inspect its type
    loaded_object = joblib.load("C:\\Users\\manal_qckxaa\\Desktop\\amad_pay_github\\hog_svm_model.pkl")
    print("Type of loaded object:", type(loaded_object))
    #model = model['model']  # Adjust key based on your save format
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {str(e)}")

def preprocess_image(image_bytes: bytes, target_size=(128, 128), visualize=True) -> np.ndarray:
    """Convert uploaded image to HOG features with optional visualization"""
    try:
        # Convert bytes to PIL Image
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to grayscale and resize
        img = img.convert('L')
        img = img.resize(target_size)
        img_array = np.array(img)
        
        # Vein-specific preprocessing
        img_array = cv2.equalizeHist(img_array)  # CLAHE alternative
        img_array = cv2.GaussianBlur(img_array, (3,3), 0)
        
        # Compute HOG features
        features, hog_image = hog(
            img_array,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,  # Always calculate for optional vis
            transform_sqrt=True,  # Better for dark veins
            feature_vector=True
        )
        
        if visualize:
            # Enhanced visualization
            hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 0.05))
            
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(img_array, cmap='gray')
            plt.title('Preprocessed Vein Image')
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(hog_image, cmap='inferno')
            plt.title('HOG Features')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig("test.png")
            plt.show()
        
        return features.reshape(1, -1)  # Shape (1, n_features)

    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

def classify_palm_vein_image(image_bytes):
    # Verify file is an image
    try:
        # If the output shows: <class 'tuple'>
        clf = loaded_object[0]  # Typically the model is first element

        hog_features = preprocess_image(image_bytes)

        # Make prediction
        prediction = clf.predict(hog_features)
        first , last = arabic_names[prediction[0]]
        print(f"{first} {last}")

        return {
            "prediction": str(prediction[0]),
            "name" : f"{first} {last}",
        }
    
    except Exception as e:
        print(f"Classification error: {str(e)}")