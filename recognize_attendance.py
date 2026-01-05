import cv2
import os
from database.db import get_connection
from datetime import date, datetime

def mark_attendance(student_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO attendance (student_id, date, time, status)
            VALUES (%s, %s, %s, 'Present')
        """, (student_id, date.today(), datetime.now().time()))
        conn.commit()
    except:
        pass
    conn.close()

def recognize_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("models/trainer.yml")

    detector = cv2.CascadeClassifier(
        "haarcascade/haarcascade_frontalface_default.xml"
    )

    cam = cv2.VideoCapture(0)
    print("🎥 Attendance system started")

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 70:
                mark_attendance(id_)
                label = f"User {id_}"
            else:
                label = "Unknown"

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,label,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

        cv2.imshow("Face Attendance", frame)

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_attendance()
