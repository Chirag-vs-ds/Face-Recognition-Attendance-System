import cv2
import os
import numpy as np

def train_model():
    dataset_path = "dataset"
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    ids = []

    print("📂 Scanning dataset folder...")

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)

        # ✅ Only allow folders like user_<number>
        if not os.path.isdir(folder_path):
            continue

        if not folder.startswith("user_"):
            print(f"⚠️ Skipping invalid folder: {folder}")
            continue

        try:
            user_id = int(folder.split("_")[1])
        except ValueError:
            print(f"⚠️ Invalid user ID in folder: {folder}")
            continue

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces.append(img)
            ids.append(user_id)

    if len(faces) == 0:
        print("❌ No valid training images found")
        return

    print(f"🧠 Training model with {len(faces)} images...")
    recognizer.train(faces, np.array(ids))

    os.makedirs("models", exist_ok=True)
    recognizer.save("models/trainer.yml")

    print("✅ Model trained successfully")
    print("📁 Saved as models/trainer.yml")

# 🔥 REQUIRED
if __name__ == "__main__":
    train_model()
