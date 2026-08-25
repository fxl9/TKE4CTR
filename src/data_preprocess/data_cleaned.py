# -*- coding: utf-8 -*-
"""
Batch cleaning script for Amazon CTR dataset CSV files.
Handles HTML cleaning, field standardization, price normalization, and format validation.
"""
import re
import html
import pandas as pd
import os
from bs4 import BeautifulSoup
from tqdm import tqdm
def clean_complex_html(content):
    if pd.isna(content) or content == 'unknown' or str(content).strip() == '':
        return 'unknown'
    content_str = str(content).strip()
    content_str = html.unescape(content_str)
    content_str = re.sub(r'<script[^>]*?>.*?</script>', '', content_str, flags=re.DOTALL)
    content_str = re.sub(r'<style[^>]*?>.*?</style>', '', content_str, flags=re.DOTALL)
    soup = BeautifulSoup(content_str, 'html.parser')
    clean_text = soup.get_text(separator=' ', strip=True)
    clean_text = re.sub(r'https?://\S+', '', clean_text)
    clean_text = re.sub(r'www\.\S+', '', clean_text)
    clean_text = re.sub(r'/gp/[^ ]+', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = re.sub(r'([.,;!?()])(?=\w)', r'\1 ', clean_text)
    clean_text = re.sub(r'(\w)(?=[.,;!?()])', r'\1 ', clean_text)
    clean_text = re.sub(r'  ', ' ', clean_text)
    clean_text = re.sub(r'[^\w\s.,;!?()\-#&*]', '', clean_text)
    clean_text = clean_text.strip().capitalize()
    if ' and ' in clean_text:
        items = [item.strip().capitalize() for item in clean_text.split(' and ')]
        clean_text = ' and '.join(items)
    return clean_text if clean_text else 'unknown'
def clean_price(price_value):
    if pd.isna(price_value) or str(price_value).strip().lower() in ['unknown', 'nan', 'none', '']:
        return None
    price_str = str(price_value).strip()
    price_str = clean_complex_html(price_str)
    price_match = re.search(r'(\d{1,3}(?:,\d{3})*\.\d{2}|\d{1,3}(?:,\d{3})*|\.\d{2}|\d+\.\d{2}|\d+)', price_str)
    if price_match:
        try:
            return float(price_match.group(1).replace(',', ''))
        except ValueError:
            return None
    return None
def extract_target_attributes(text):
    if pd.isna(text) or text == 'unknown' or str(text).strip() == '':
        return 'unknown'
    text_str = str(text).strip()
    TARGET_ATTRIBUTES = ['Product Dimensions', 'Shipping Weight', 'Item Weight']
    DELETE_ATTRIBUTES = ['ASIN', 'UPC', 'Item model number']
    SEPARATOR_PATTERN = r'::?'
    attr_pattern = r'(' + '|'.join([re.escape(attr) for attr in
                                    TARGET_ATTRIBUTES + DELETE_ATTRIBUTES]) + r')' + SEPARATOR_PATTERN + r'\s*([^&(and)]+?)(?=\s+(and\s+)?(' + '|'.join(
        [re.escape(attr) for attr in TARGET_ATTRIBUTES + DELETE_ATTRIBUTES]) + r')::?|\s*\(|\s*$)'
    matches = re.findall(attr_pattern, text_str, flags=re.IGNORECASE | re.DOTALL)
    target_pairs = []
    for match in matches:
        attr_name = match[0].strip().capitalize()
        attr_value = match[1].strip().lower()
        if attr_name in TARGET_ATTRIBUTES and attr_value:
            target_pairs.append(f"{attr_name}: {attr_value}")
    if target_pairs:
        return '; '.join(target_pairs)
    clean_text = text_str
    for delete_attr in DELETE_ATTRIBUTES:
        clean_text = re.sub(
            rf'{re.escape(delete_attr)}{SEPARATOR_PATTERN}\s*[^\s&(and)]+',
            '',
            clean_text,
            flags=re.IGNORECASE
        )
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    clean_text = clean_text.capitalize()
    return clean_text if clean_text else 'unknown'
def standardize_list_fields(text):
    if text == 'unknown' or not text:
        return 'unknown'
    text_str = str(text).strip()
    separators = [r' and ', r' & ', r', ', r'、', r'; ']
    for sep in separators:
        text_str = re.sub(sep, ', ', text_str)
    items = [item.strip() for item in text_str.split(', ') if item.strip()]
    unique_items = list(dict.fromkeys(items))
    return ', '.join(unique_items) if unique_items else 'unknown'
def validate_format_consistency(cleaned_df, target_fields):
    print("\n" + "=" * 50)
    print("Format Consistency Validation")
    print("=" * 50)
    format_issues = {}
    for field in target_fields:
        if field not in cleaned_df.columns:
            continue
        issues = []
        sample_values = cleaned_df[field].drop_duplicates().head(5).tolist()
        if field in ['also_buy', 'also_view', 'similar_item']:
            for val in sample_values:
                if val != 'unknown' and not re.match(r'^[\w\s_,]+$', val):
                    issues.append(f"Invalid format: {val[:50]}...")
        elif field in ['details']:
            for val in sample_values:
                if val != 'unknown' and '; ' in val and not re.match(r'^[A-Z][^:]+: [^;]+(; [A-Z][^:]+: [^;]+)*$', val):
                    issues.append(f"Invalid attribute format: {val[:50]}...")
        if issues:
            format_issues[field] = issues[:3]
    if not format_issues:
        print("All fields have consistent format")
    else:
        print("Format issues found:")
        for field, issues in format_issues.items():
            print(f"\nField {field}:")
            for issue in issues:
                print(f"- {issue}")
def batch_clean_amazon_csv(csv_paths, backup_dir=None, target_fields=None):
    default_target_fields = {
        'text': [
            'description', 'feature', 'category', 'style',
            'user_hist_title', 'title', 'summary', 'reviewText'
        ],
        'list': [
            'also_buy', 'also_view', 'similar_item'
        ],
        'attribute': [
            'details'
        ],
        'special': [
            'rank'
        ]
    }
    if backup_dir and not os.path.exists(backup_dir):
        os.makedirs(backup_dir, exist_ok=True)
        print(f"Backup directory created: {backup_dir}")
    for csv_path in tqdm(csv_paths, desc="Total Progress: Batch Cleaning CSV Files"):
        file_name = os.path.basename(csv_path)
        print(f"\n" + "=" * 60)
        print(f"Processing file: {file_name}")
        print(f"File path: {csv_path}")
        print("=" * 60)
        if not os.path.exists(csv_path):
            print(f"File not found, skipping: {file_name}")
            continue
        original_df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if backup_dir:
            backup_path = os.path.join(backup_dir, file_name)
            original_df.to_csv(backup_path, index=False, encoding='utf-8-sig')
            print(f"Original file backed up to: {backup_path}")
        cleaned_df = original_df.copy()
        all_fields = []
        for field_type, fields in default_target_fields.items():
            valid_fields = [f for f in fields if f in cleaned_df.columns]
            if valid_fields:
                all_fields.extend(valid_fields)
                print(f"Processing {field_type} fields: {valid_fields}")
        if not all_fields and 'price' not in cleaned_df.columns:
            print(f"No fields to clean, skipping file")
            continue
        print("\n===== Starting Field Cleaning =====")
        text_fields = [f for f in default_target_fields['text'] if f in cleaned_df.columns]
        for field in text_fields:
            print(f"\nCleaning text field: {field}")
            cleaned_df[field] = [
                clean_complex_html(val)
                for val in tqdm(cleaned_df[field], desc=f"Processing {field}", leave=False)
            ]
        list_fields = [f for f in default_target_fields['list'] if f in cleaned_df.columns]
        for field in list_fields:
            print(f"\nStandardizing list field: {field}")
            cleaned_df[field] = [clean_complex_html(val) for val in cleaned_df[field]]
            cleaned_df[field] = [
                standardize_list_fields(val)
                for val in tqdm(cleaned_df[field], desc=f"Processing {field}", leave=False)
            ]
        attr_fields = [f for f in default_target_fields['attribute'] if f in cleaned_df.columns]
        for field in attr_fields:
            print(f"\nExtracting and standardizing attribute field: {field}")
            cleaned_df[field] = [clean_complex_html(val) for val in cleaned_df[field]]
            cleaned_df[field] = [
                extract_target_attributes(val)
                for val in tqdm(cleaned_df[field], desc=f"Processing {field}", leave=False)
            ]
        if 'rank' in cleaned_df.columns:
            print(f"\nProcessing special field: rank")
            cleaned_df['rank'] = [
                re.sub(r'[^\d,]', '', str(val))
                if val != 'unknown' else 'unknown'
                for val in tqdm(cleaned_df['rank'], desc="Processing rank field", leave=False)
            ]
        if 'price' in cleaned_df.columns:
            print("\n===== Processing price field =====")
            cleaned_df['price'] = [
                clean_price(val)
                for val in tqdm(cleaned_df['price'], desc="Cleaning price", leave=False)
            ]
            price_mean = cleaned_df['price'].dropna().mean() if not cleaned_df['price'].dropna().empty else 0
            cleaned_df['price'] = cleaned_df['price'].fillna(round(price_mean, 2))
        validate_format_consistency(cleaned_df, all_fields + (['price'] if 'price' in cleaned_df.columns else []))
        cleaned_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\nCleaned file saved to original path: {csv_path}")
    print(f"\n" + "=" * 60)
    print("All CSV files batch cleaning completed!")
    print("=" * 60)
if __name__ == '__main__':
    CSV_PATHS = [
        r"./datasets/amazon_review_data/processed_data/All_Beauty_processed.csv",
        r"./datasets/amazon_review_data/processed_data/Amazon_Fashion_processed.csv",
        r"./datasets/amazon_review_data/processed_data/Digital_Music_processed.csv",
        r"./datasets/amazon_review_data/processed_data/Gift_Cards_processed.csv",
        r"./datasets/amazon_review_data/processed_data/Musical_Instruments_processed.csv"
    ]
    BACKUP_DIR = "./datasets/amazon_review_data/processed_data_backup"
    CUSTOM_TARGET_FIELDS = None
    batch_clean_amazon_csv(
        csv_paths=CSV_PATHS,
        backup_dir=BACKUP_DIR,
        target_fields=CUSTOM_TARGET_FIELDS
    )
