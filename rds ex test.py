import mysql.connector

try:
    conn = mysql.connector.connect(
        host="cogo-insurance.cjuawrphqolo.us-east-1.rds.amazonaws.com",
        user="cogoinsurance",
        password="9k7ofPkde7qR",
        database="cogoinsurance",
        port=3306
    )

    print("Connected successfully!")
except mysql.connector.Error as err:
    print("Error:", err)

cursor = conn.cursor()
cursor.execute("SHOW TABLES;")
for table in cursor.fetchall():
    print(table)

cursor.close()
conn.close()
