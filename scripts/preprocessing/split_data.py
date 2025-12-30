import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(input_path, output_dir):
    df = pd.read_csv(input_path)
    print(f"Total cleaned pairs: {len(df)}")
    
    # 80/10/10 split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Splits saved to {output_dir}:")
    print(f" - Train: {len(train_df)}")
    print(f" - Val: {len(val_df)}")
    print(f" - Test: {len(test_df)}")

if __name__ == "__main__":
    split_data("data/processed/cleaned_pairs.csv", "data/processed/splits")
