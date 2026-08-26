# -*- coding: utf-8 -*-
"""
Train/validation/test split script for Amazon CTR dataset.
Processes all domain CSV files, sorts chronologically.
Chronological split: last 10% as test, remaining 90% split 8:1 for train/valid, yields exact 80/10/10.
reviewText, user_hist_id, user_hist_title are kept only in train set; removed from validation and test sets.
"""
import os
import traceback
import pandas as pd

def make_train_valid_dfs(data_path):
    df = pd.read_csv(data_path)
    if 'unixReviewTime' in df.columns:
        df['_sort_time'] = pd.to_datetime(df['unixReviewTime'], unit='s', errors='coerce')
        valid_mask = df['_sort_time'].notna()
        unknown_count = (~valid_mask).sum()
        if unknown_count > 0:
            print(f"Dropped {unknown_count} records with invalid unixReviewTime")
            df = df[valid_mask].copy()
        df = df.sort_values(by='_sort_time', ascending=True, kind='mergesort').reset_index(drop=True)
        df = df.drop(columns=['_sort_time'])
        print("Sorted by unixReviewTime in ascending order")
    else:
        print(f"Warning: 'unixReviewTime' column not found in {os.path.basename(data_path)}, will not sort by time")
    return df

def split_dataset(df):
    total = len(df)
    if total == 0:
        raise ValueError("Dataset is empty, cannot perform split")
    test_split = round(total * 0.9)
    remaining = test_split
    valid_split = round(remaining * (8 / 9))

    train = df.iloc[:valid_split].copy()
    valid = df.iloc[valid_split:test_split].copy()
    test = df.iloc[test_split:].copy()

    if len(train)==0 or len(valid)==0 or len(test)==0:
        raise ValueError("Split result contains empty subset, dataset size too small")
    return train, valid, test

def save_split_data(data, save_dir, base_name, data_type):
    filename = f"{base_name}_{data_type}.csv"
    file_path = os.path.join(save_dir, filename)
    if data_type in ["valid","test"]:
        drop_cols = [col for col in ["reviewText", "user_hist_id", "user_hist_title"] if col in data.columns]
        if drop_cols:
            data = data.drop(columns=drop_cols)
    data.to_csv(file_path, index=False, encoding='utf-8-sig')
    return file_path

def process_all_datasets():
    DATA_PATHS = [
        r"./datasets/amazon_review_data/processed_data/All_Beauty_processed.csv",
        r"./datasets/amazon_review_data/processed_data/Amazon_Fashion_processed.csv",
        r"./datasets/amazon_review_data/processed_data/Digital_Music_processed.csv",
        r"./datasets/amazon_review_data/processed_data/Gift_Cards_processed.csv",
        r"./datasets/amazon_review_data/processed_data/Musical_Instruments_processed.csv"
    ]
    save_root = r"./datasets/amazon_review_data/data_splits"
    os.makedirs(save_root, exist_ok=True)
    print(f"Results will be saved to: {save_root}")
    for path in DATA_PATHS:
        try:
            full_filename = os.path.splitext(os.path.basename(path))[0]
            base_name = full_filename.replace("_processed", "")
            print(f"\nProcessing dataset: {base_name}")
            df = make_train_valid_dfs(path)
            print(f"Data loaded, total {len(df)} records")
            data_save_dir = os.path.join(save_root, base_name)
            os.makedirs(data_save_dir, exist_ok=True)
            train, valid, test = split_dataset(df)
            train_file = save_split_data(train, data_save_dir, base_name, "train")
            valid_file = save_split_data(valid, data_save_dir, base_name, "valid")
            test_file = save_split_data(test, data_save_dir, base_name, "test")
            print(f"  Saved files:")
            print(f"    Train set: {os.path.basename(train_file)} ({len(train)} records)")
            print(f"    Validation set: {os.path.basename(valid_file)} ({len(valid)} records)")
            print(f"    Test set: {os.path.basename(test_file)} ({len(test)} records)")
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    process_all_datasets()
