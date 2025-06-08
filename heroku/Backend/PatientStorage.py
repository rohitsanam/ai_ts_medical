import os
import pandas as pd

CSV_FILE = "patient_data.csv"

# ✅ Create CSV file if it doesn't exist
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["PatientID", "PatientName", "PatientSex", "PatientWeight", "Age", "Diagnosis"])
    df.to_csv(CSV_FILE, index=False)