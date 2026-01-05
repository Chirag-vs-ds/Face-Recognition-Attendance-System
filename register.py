from database.db import get_connection

def register_student():
    name = input("Enter Name: ")
    roll = input("Enter Roll No: ")
    dept = input("Enter Department: ")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO students (name, roll_no, department) VALUES (%s,%s,%s)",
        (name, roll, dept)
    )

    conn.commit()
    conn.close()

    print("✅ Student registered successfully")

# 🔥 THIS LINE IS IMPORTANT
if __name__ == "__main__":
    register_student()
