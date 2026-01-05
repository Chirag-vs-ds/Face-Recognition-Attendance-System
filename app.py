from flask import Flask, render_template, Response, request, redirect
import cv2
import os
from datetime import date, datetime
from database.db import get_connection

app = Flask(__name__)

camera = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists("models/trainer.yml"):
    recognizer.read("models/trainer.yml")

# ------------------ VIDEO STREAM ------------------
def gen_frames():
    while True:
        success, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.2, 5)

        for (x,y,w,h) in faces:
            id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 70:
                mark_attendance(id)
                label = f"ID {id}"
            else:
                label = "Unknown"

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,label,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ------------------ ROUTES ------------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        roll = request.form['roll']
        dept = request.form['dept']
        student_id = register_student(name, roll, dept)
        capture_faces(student_id)
        train_model()
        return redirect('/')
    return render_template("register.html")

@app.route('/attendance')
def attendance():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.name, s.roll_no, a.date, a.time
        FROM attendance a
        JOIN students s ON a.student_id=s.student_id
    """)
    data = cur.fetchall()
    conn.close()
    return render_template("attendance.html", data=data)

# ------------------ FUNCTIONS ------------------
def register_student(name, roll, dept):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO students (name, roll_no, department) VALUES (%s,%s,%s)",
        (name, roll, dept)
    )
    conn.commit()
    student_id = cur.lastrowid
    conn.close()
    return student_id

def capture_faces(student_id):
    path = f"dataset/user_{student_id}"
    os.makedirs(path, exist_ok=True)
    count = 0

    while count < 100:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            count += 1
            cv2.imwrite(f"{path}/{count}.jpg", gray[y:y+h,x:x+w])

def train_model():
    faces, ids = [], []
    for folder in os.listdir("dataset"):
        sid = int(folder.split("_")[1])
        for img in os.listdir(f"dataset/{folder}"):
            img_path = f"dataset/{folder}/{img}"
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(image)
            ids.append(sid)

    recognizer.train(faces, ids)
    os.makedirs("models", exist_ok=True)
    recognizer.save("models/trainer.yml")

def mark_attendance(student_id):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO attendance (student_id,date,time,status)
            VALUES (%s,%s,%s,'Present')
        """, (student_id, date.today(), datetime.now().time()))
        conn.commit()
    except:
        pass
    conn.close()

if __name__ == '__main__':
    app.run(debug=True)
