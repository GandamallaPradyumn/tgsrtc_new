import pandas as pd
from sqlalchemy import create_engine

# Step 1: Load the CSV file into a Pandas DataFrame
csv_file_path ='lsa2025.csv'
df = pd.read_csv(csv_file_path)
print("Original data shape:", df.shape)

# Optional: Show how many nulls are present in each column
print("Null values in each column:\n", df.isnull().sum())

# Step 1.1: Drop fully empty rows or rows with missing critical fields
required_fields = ['employee_id', 'depot', 'leave_type', 'date']
df = df.dropna(how='all')  # Drop completely empty rows
df = df.dropna(subset=required_fields)  # Drop rows where any required field is null
print("Filtered data shape:", df.shape)

# Step 2: Create a connection to MySQL
user_name = 'root'
password = ''
database = 'tgsrtc_new'

try:
    engine = create_engine(f'mysql+pymysql://{user_name}:{password}@localhost/{database}')
    print("Connection successful!")  # If connection is successful
except Exception as e:
    print(f"Error occurred while connecting to the database: {e}")

# Step 3: Upload the DataFrame to MySQL
try:
    print("Using engine:", engine)
    df.to_sql('driver_absenteeism', con=engine, if_exists='append', index=False)
    print("Data uploaded successfully!")
except Exception as e:
    print(f"Error occurred while uploading to the database: {e}")
