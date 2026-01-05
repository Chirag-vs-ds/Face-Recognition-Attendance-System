from database.db import get_connection

try:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()

    print("✅ Python successfully connected to MySQL")
    print("📋 Tables:")

    for table in tables:
        print(table)

    conn.close()

except Exception as e:
    print("❌ Connection failed")
    print(e)
