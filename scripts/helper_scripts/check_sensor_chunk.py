import pandas as pd
import glob
import sys
from pathlib import Path

def check_all_files(folder, start_time, end_time):
    """
    Checks all sensor CSV files in a folder for data within a specified time range.
    Reports row counts, missing timestamps, and duplicates for each file.
    """
    files = sorted(glob.glob(str(Path(folder) / "*sensor.csv")))
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    total_rows = 0

    for f in files:
        df = pd.read_csv(f)
        # Rename timestamp column if needed
        if 'HEADER_TIMESTAMP' in df.columns:
            df = df.rename(columns={'HEADER_TIMESTAMP': 'timestamp'})
        # Parse timestamps with explicit format for consistency
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        # Filter rows within the specified time range
        chunk = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
        if len(chunk) > 0:
            print(f"{Path(f).name}:")
            print(f"  Rows in range: {len(chunk)}")
            print(f"  Missing timestamps: {chunk['timestamp'].isna().sum()}")
            print(f"  Duplicate timestamps: {chunk['timestamp'].duplicated().sum()}")
            print(f"  Min timestamp: {chunk['timestamp'].min()}")
            print(f"  Max timestamp: {chunk['timestamp'].max()}")
            print(f"  Any NaN in X/Y/Z: {chunk[['X','Y','Z']].isna().any().any()}")

            # 1) Any X/Y/Z values are 0
            zero_mask = (chunk[['X', 'Y', 'Z']] == 0).any(axis=1)
            print(f"  Rows with any X/Y/Z == 0: {zero_mask.sum()}")

            # 2) All X/Y/Z values are the same for a row
            same_mask = (chunk['X'] == chunk['Y']) & (chunk['Y'] == chunk['Z'])
            print(f"  Rows with X == Y == Z: {same_mask.sum()}")

            # 3) Same values for multiple rows
            xyz_tuples = list(zip(chunk['X'], chunk['Y'], chunk['Z']))
            from collections import Counter
            counts = Counter(xyz_tuples)
            repeated = sum(1 for v in counts.values() if v > 1)
            print(f"  Unique (X,Y,Z) value sets repeated in multiple rows: {repeated}")

            print("-" * 40)
            total_rows += len(chunk)
    print(f"Total rows in range across all files: {total_rows}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python check_sensor_chunk.py <folder> <start_time> <end_time>")
        print("Example: python check_sensor_chunk.py data/raw/2011-12/70224 '2000-01-06 15:59:30' '2000-01-06 23:59:59.988000'")
        sys.exit(1)
    check_all_files(sys.argv[1], sys.argv[2], sys.argv[3])