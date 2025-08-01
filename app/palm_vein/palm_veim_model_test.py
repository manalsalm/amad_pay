
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# 📂 Dataset path (update as needed)
img_dir = 'C:\\Users\\manal_qckxaa\\Desktop\\amad_pay_github\\multimodal pp and pv\\DB_Vein\\'

# ⚙️ Parameters
subject_count = 98
images_per_subject = 140
img_size = (48,96)  # (width, height) for OpenCV resize

# 🖼️ Step 1: Load and preprocess images
images = []
labels = []
image_paths = []

for i in range(1, subject_count + 1):
    subject_label = f"{i:03d}"
    subject_folder_path = os.path.join(img_dir, subject_label)

    if not os.path.isdir(subject_folder_path):
        continue

    for j in range(1, images_per_subject + 1):
        image_name = f"{subject_label}I_VL{j:02d}.bmp"
        image_path = os.path.join(subject_folder_path, image_name)

        if os.path.exists(image_path):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, img_size)
            flat = img_resized.flatten()
            images.append(flat)
            labels.append(i)
            image_paths.append(image_path)

images = np.array(images)
labels = np.array(labels)
# 🔃 Step 2: Standardize
scaler = StandardScaler()
images_std = scaler.fit_transform(images)

# 🔻 Step 3: PCA
pca = PCA(n_components=50)
pca_features = pca.fit_transform(images_std)

# 📊 Step 4: Visualize PCA
plt.figure(figsize=(8, 6))
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title("PCA - 2D Projection of Image Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Subject Label")
plt.show()

# 🧑‍💻 Step 5: LDA
lda = LDA(n_components=2)
lda_features = lda.fit_transform(images_std, labels)

# 📊 Step 6: Visualize LDA
plt.figure(figsize=(8, 6))
plt.scatter(lda_features[:, 0], lda_features[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title("LDA - 2D Projection of Image Features")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.colorbar(label="Subject Label")
plt.show()

# 📏 Step 7: Euclidean distance in PCA space
dists = euclidean_distances(pca_features)
print("Euclidean Distance Matrix (rounded):")
print(np.round(dists, 2))

# 🧠 Step 8: Feature Extraction Function
def extract_features(img_path, resize_shape=(60, 80), pca_model=None, scaler_model=None):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    img_resized = cv2.resize(img, resize_shape)
    flat = img_resized.flatten().reshape(1, -1)
    standardized = scaler_model.transform(flat)
    features = pca_model.transform(standardized)
    return features

# 🖼️ Step 9: Show Sample Image
sample_image = images[0].reshape(img_size[1], img_size[0])  # (height, width)
plt.imshow(sample_image, cmap='gray')
plt.title(f"Sample Image (Subject {labels[0]})")
plt.axis('off')
plt.show()

# 📌 Step 10: Extract and Show Sample Feature Vector
sample_path = image_paths[0]
sample_feature = extract_features(sample_path, resize_shape=img_size, pca_model=pca, scaler_model=scaler)

print("Extracted PCA Feature Vector:")
print(np.round(sample_feature, 2))