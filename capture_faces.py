import cv2
import os

def capture_faces():
    # ---------- INPUT ----------
    student_id = input("Enter student ID: ")

    # ---------- CAMERA ----------
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Camera not accessible")
        return

    # ---------- HAAR CASCADE ----------
    BASE_DIR = os.path.dirname(__file__)
    CASCADE_PATH = os.path.join(
        BASE_DIR,
        "haarcascade",
        "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(CASCADE_PATH)

    if detector.empty():
        print("❌ Haar Cascade file not loaded")
        return

    print("✅ Haar Cascade loaded")
    print("📸 Camera started. Press ESC to stop.")

    # ---------- DATASET FOLDER ----------
    path = os.path.join("dataset", f"user_{student_id}")
    os.makedirs(path, exist_ok=True)

    count = 0

    # ---------- CAPTURE LOOP ----------
    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(
                os.path.join(path, f"{count}.jpg"),
                gray[y:y+h, x:x+w]
            )
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Capturing Faces", frame)

        if cv2.waitKey(1) == 27 or count >= 100:
            break

    # ---------- CLEANUP ----------
    cam.release()
    cv2.destroyAllWindows()
    print(f"✅ {count} images saved for student {student_id}")

# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    capture_faces()
