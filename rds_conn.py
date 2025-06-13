import mysql.connector

# Step 1: Connect to the RDS MySQL database
conn = mysql.connector.connect(
        host="cogo-insurance.cjuawrphqolo.us-east-1.rds.amazonaws.com",        
        user="cogoinsurance",    
        password="9k7ofPkde7qR",
        database="cogoinsurance" 
    )

cursor = conn.cursor()

# Step 2: Get the list of tables
cursor.execute("SHOW TABLES;")
tables = [t[0] for t in cursor.fetchall()]  # Now 'tables' is defined

# Step 3: Print table structure and top 5 rows from each
for table in tables:
    print(f"\nTable: {table}")
    
    # Get structure
    cursor.execute(f"DESCRIBE {table};")
    print("Columns:")
    for column in cursor.fetchall():
        print(column)
    
    # Get sample data
    cursor.execute(f"SELECT * FROM {table} LIMIT 5;")
    rows = cursor.fetchall()
    print("Sample Rows:")
    for row in rows:
        print(row)

# Close connection
cursor.close()
conn.close()