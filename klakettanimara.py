import numpy as np
import cv2
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog, graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier 

def hough_circle_features(gray):
    img_h, img_w = gray.shape[:2]
    blurred = cv2.medianBlur(gray, 11)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.array([0.0, 0.0, 0.0])

    largest_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)

    max_pixel_radius = min(img_h, img_w) // 2
    safe_radius = min(radius, max_pixel_radius)

    if safe_radius <= 5:
        return np.array([0.0, 0.0, 0.0])

    center = np.array([x, y], dtype=float)
    image_center = np.array([img_w / 2.0, img_h / 2.0], dtype=float)
    center_dist = float(np.linalg.norm(center - image_center))

    return np.array([1.0, float(safe_radius), center_dist])

def extract_features_from_image(img_matrix):

        img_rgb = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        gauss1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gauss2 = cv2.GaussianBlur(gray, (0, 0), 2.0)
        dog = cv2.absdiff(gauss1, gauss2)
        dog_mean = float(np.mean(dog))
        dog_std = float(np.std(dog))
        dog_hist = cv2.calcHist([dog], [0], None, [16], [0, 256]).flatten()
        dog_hist = dog_hist / np.sum(dog_hist) if np.sum(dog_hist) > 0 else dog_hist
        dog_features = np.concatenate(([dog_mean, dog_std], dog_hist))

        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        glcm = graycomatrix(gray, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        texture_features = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])


        hog_features = hog(gray, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=False)

        sift = cv2.SIFT_create(nfeatures=100)
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is not None:
            desc_features = descriptors.flatten()
            if len(desc_features) < 100 * 128:
                desc_features = np.pad(desc_features, (0, 100 * 128 - len(desc_features)))
            else:
                desc_features = desc_features[:100 * 128]
        else:
            desc_features = np.zeros(100 * 128)

        hist = cv2.calcHist([img_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_features = hist.flatten()

        hist_gray= cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_gray1 = hist_gray.flatten()


        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
            contour_features = np.array([len(contours), area, circularity])
        else:
            contour_features = np.zeros(3)

        circle_features = hough_circle_features(gray)

        k = 3
        pixel_values = img_resized.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixel_values, K=k, bestLabels=None, criteria=criteria, attempts=3, flags=cv2.KMEANS_RANDOM_CENTERS)

        centers_features = centers.flatten() / 255.0 

        # Proportion of each segment (k = 3 values)
        labels_flat = labels.flatten()
        proportions = np.array([np.sum(labels_flat == i) / len(labels_flat) for i in range(k)])

        kmeans_features = np.concatenate((centers_features, proportions))

        return np.concatenate((hist_features, hist_gray1,texture_features, desc_features, hog_features, contour_features, circle_features, dog_features, kmeans_features))





datadir = 'dataset-balls/train/'
testdir = 'dataset-balls/test/'
target = []
flat_data = []
test_target = []
test_flat_data = []
categories = ['american_football', 'baseball', 'basketball', 'football', 'golf_ball', 'tennis_ball']



print("lahza wahda ma3alesh..")
for category in categories:
    class_index = categories.index(category)
    path = os.path.join(datadir, category)
    print(path)
    if not os.path.exists(path):
        print(f"mesh mawgood ya basha - {path}")
        continue

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img_matrix = cv2.imread(img_path)
        
        if img_matrix is None:
            print(f"feh moshkela - {img_path}")
            continue

        combined_features = extract_features_from_image(img_matrix)
        flat_data.append(combined_features)
        target.append(class_index)

flat_data = np.array(flat_data)
target = np.array(target)
df = pd.DataFrame(flat_data)
df['target'] = target
print("tamm.")




scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df.drop('target', axis=1))

print("estany bat3alem ahoo ya basha, shwaya shwaya")

model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, df['target'])
print(" tamam ya basha!")






print("\n ya mosahel!")

for category in categories:
    class_index = categories.index(category)
    path = os.path.join(testdir, category)
    print(path)
    if not os.path.exists(path):
        print(f"mesh mawgood ya basha - {path}")
        continue

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img_matrix = cv2.imread(img_path)
        
        if img_matrix is None:
            print(f"feh moshkela - {img_path}")
            continue

        combined_features = extract_features_from_image(img_matrix)
        test_flat_data.append(combined_features)
        test_target.append(class_index)





X_test_scaled = scaler.transform(np.array(test_flat_data))
y_pred = model.predict(X_test_scaled)
print(f"\nFinal Accuracy with Random Forest: {accuracy_score(test_target, y_pred) * 100:.2f}")
print(classification_report(test_target, y_pred, target_names=categories))