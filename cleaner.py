import pandas as pd

class DataCleaner:
    def clean_data(self, df):
        print("[CLEANER] Checking for data issues...")
        initial_count = len(df)
        df = df.drop_duplicates()
        new_count = len(df)
        if initial_count != new_count:
            print(f"[CLEANER] Removed {initial_count - new_count} duplicate rows.")
        if df.isnull().sum().sum() > 0:
            print("[CLEANER] Found missing values. Filling with mean...")
            df = df.fillna(df.mean())
        else:
            print("[CLEANER] No missing values found.")
            
        return df