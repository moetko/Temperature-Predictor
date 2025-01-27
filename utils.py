import pandas as pd
import re

# configuration
WINDOW_SIZE = 3
RANDOM_STATE = 42
TEST_SIZE = 0.2

# preprocess data
def preprocess(file_paths):
    datasets = {}
    for label, path in file_paths.items():
        df = pd.read_csv(path).set_index('Time (s)').drop(columns=['NO.'], errors='ignore').dropna()
        df.columns = [extract_voltage(col) for col in df.columns]
        datasets[label] = df

    return datasets

# extracts voltage value from column name
def extract_voltage(column_name):
    match = re.search(r'\((\d+)V\)', column_name)
    return float(match.group(1)) if match else None

