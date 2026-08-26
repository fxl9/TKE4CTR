# -*- coding: utf-8 -*-
"""
Data preprocessing for Amazon multi‑domain CTR dataset.
Raw json.gz files should be manually downloaded from https://nijianmo.github.io/amazon/index.html.
Used domains: All_Beauty, Amazon_Fashion, Digital_Music, Gift_Cards, Musical_Instruments.
"""

import pandas as pd
import re
import joblib
import warnings
import numpy as np
import os
import json
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
from tqdm import tqdm
warnings.filterwarnings('ignore')

SCENARIO_MAPPING = {
    0: "Amazon Fashion",
    1: "Digital Music",
    2: "Musical Instruments",
    3: "Gift Cards",
    4: "All_Beauty"
}
EXCLUDE_ATTRIBUTES = {'vote', 'imageURL', 'imageURLHighRes', 'image', 'main_cat', 'fit', 'verified', 'tech1', 'tech2'}
FINAL_COLS = [
    "Domain",
    "scenario_id",
    "user_id",
    "user_name",
    "asin",
    "price",
    "brand",
    "title",
    "similar_item",
    "rank",
    "category",
    "reviewText",
    "label",
    "user_hist_id",
    "user_hist_title",
    "unixReviewTime",
    "also_buy",
    "also_view",
    "description",
    "details",
    "feature",
    "style"
]

def clean_text_blank_lines(text):
    if isinstance(text, (list, np.ndarray)):
        non_empty_items = [str(item) for item in text if not pd.isna(item) and str(item).strip() != '']
        text = ' and '.join(non_empty_items) if non_empty_items else 'unknown'
    if pd.isna(text) or str(text).strip() == 'unknown':
        return 'unknown'
    clean_str = str(text)
    clean_str = clean_str.replace('\n', '').replace('\r', '').replace('\t', '')
    clean_str = re.sub(r'\s+', ' ', clean_str)
    clean_str = clean_str.strip()
    return clean_str if clean_str else 'unknown'

def flatten_nested_structure(x):
    flat_list = []
    stack = [(x,)]
    while stack:
        node, = stack.pop()
        if node is None or pd.isna(node):
            continue
        if isinstance(node, np.ndarray):
            node = node.tolist()
        if isinstance(node, list):
            for elem in reversed(node):
                stack.append((elem,))
        elif isinstance(node, dict):
            for key, value in reversed(list(node.items())):
                clean_key = clean_text_blank_lines(key)
                if not clean_key or clean_key == 'unknown':
                    continue
                stack.append(({"__kv": (clean_key, value)},))
        elif isinstance(node, dict) and "__kv" in node:
            k, v = node["__kv"]
            sub_flat = []
            sub_stack = [(v,)]
            while sub_stack:
                sub_node, = sub_stack.pop()
                if sub_node is None or pd.isna(sub_node):
                    continue
                if isinstance(sub_node, np.ndarray):
                    sub_node = sub_node.tolist()
                if isinstance(sub_node, list):
                    for e in reversed(sub_node):
                        sub_stack.append((e,))
                elif isinstance(sub_node, bool):
                    sub_flat.append(str(sub_node))
                elif isinstance(sub_node, str):
                    s = clean_text_blank_lines(sub_node)
                    if s and s != 'unknown':
                        sub_flat.append(s)
                elif not pd.isna(sub_node):
                    s = clean_text_blank_lines(str(sub_node))
                    if s and s != 'unknown':
                        sub_flat.append(s)
            if sub_flat:
                flat_list.append(f"{k}: {' and '.join(sub_flat)}")
        elif isinstance(node, bool):
            flat_list.append(str(node))
        elif isinstance(node, str):
            clean_str = clean_text_blank_lines(node)
            if clean_str and clean_str != 'unknown':
                flat_list.append(clean_str)
        elif not pd.isna(node):
            clean_str = clean_text_blank_lines(str(node))
            if clean_str and clean_str != 'unknown':
                flat_list.append(clean_str)
    return flat_list

def process_nested_field(x):
    if isinstance(x, (list, np.ndarray)) and len(x) == 0:
        return 'unknown'
    flat_list = flatten_nested_structure(x)
    non_empty = [item for item in flat_list if item.strip() != '' and item != 'unknown']
    return ' and '.join(non_empty) if non_empty else 'unknown'

def clean_rank(rank_str):
    if pd.isna(rank_str) or str(rank_str).strip() == '':
        return 'unknown'
    rank_str = str(rank_str).strip()
    cleaned_rank = re.sub(r'\(.*$', '', rank_str).strip()
    return cleaned_rank if cleaned_rank else 'unknown'

def process_style_field(style_value):
    if pd.isna(style_value) or str(style_value).strip().lower() in ['unknown', 'nan', 'none', '']:
        return 'unknown'
    if isinstance(style_value, dict):
        items = []
        for key, value in style_value.items():
            clean_key = key.strip().rstrip(':').strip()
            clean_value = str(value).strip()
            if clean_key and clean_value:
                items.append(f"{clean_key}: {clean_value}")
        return ' and '.join(items) if items else 'unknown'
    style_str = str(style_value).strip()
    processed = style_str.replace('{', '').replace('}', '').replace("'", '').replace('"', '')
    if ':' in processed:
        parts = processed.split(':', 1)
        if len(parts) == 2:
            key = parts[0].strip().rstrip(':').strip()
            value = parts[1].strip()
            if key and value:
                return f"{key}: {value}"
    return processed if processed.strip() else 'unknown'

def clean_related_html_content(content):
    if not content:
        return ''
    content_str = str(content).strip()
    soup = BeautifulSoup(content_str, 'html.parser')
    clean_text = soup.get_text(separator=' ', strip=True)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = re.sub(r'http\S+', '', clean_text)
    clean_text = re.sub(r'[^a-zA-Z0-9\s&:]', '', clean_text)
    return clean_text.strip()

def process_related_fields(data, asin_encoder):
    tqdm.write("\nProcessing related item fields, stripping html...")
    try:
        unknown_idx = list(asin_encoder.classes_).index('unknown_asin')
    except ValueError:
        new_classes = list(asin_encoder.classes_) + ['unknown_asin']
        asin_encoder.classes_ = np.array(new_classes)
        unknown_idx = len(new_classes) - 1
    unknown_product = f"product_{unknown_idx}"
    asin_to_encoded = {asin: idx for idx, asin in enumerate(asin_encoder.classes_)}

    def batch_process(field):
        if field not in data.columns:
            return
        all_items = []
        lengths = []
        empty_mask = []
        for items in data[field]:
            if not items or (isinstance(items, str) and items.strip() == ''):
                empty_mask.append(True)
                lengths.append(0)
                continue
            empty_mask.append(False)
            cleaned_content = clean_related_html_content(items)
            if isinstance(items, str):
                item_list = [item.strip() for item in re.split(r';|and', cleaned_content) if item.strip()]
            elif isinstance(items, list):
                item_list = [clean_related_html_content(str(item)).strip() for item in items if clean_related_html_content(str(item)).strip()]
            else:
                item_list = []
            if field in ['also_buy', 'also_view'] and len(item_list) > 5:
                item_list = item_list[:5]
            lengths.append(len(item_list))
            all_items.extend(item_list)
        encoded_results = []
        for item in all_items:
            encoded_results.append(asin_to_encoded.get(item, unknown_idx))
        processed = []
        current = 0
        for i in range(len(empty_mask)):
            if empty_mask[i]:
                processed.append(unknown_product)
            else:
                end = current + lengths[i]
                batch = encoded_results[current:end]
                current = end
                processed_str = ' and '.join([f'product_{code}' for code in batch])
                processed.append(processed_str if processed_str else unknown_product)
        data[field] = processed
        if field in ['also_buy', 'also_view']:
            tqdm.write(f"   - {field} finished: {len(data)} records, keep top‑5 items")
        else:
            tqdm.write(f"   - {field} finished: {len(data)} records")
        if len(data) > 0:
            tqdm.write(f"   - sample: {str(data[field].iloc[0])[:100]}...")
    related_fields = ['also_buy', 'also_view', 'similar_item']
    existing_related_fields = [f for f in related_fields if f in data.columns]
    for field in existing_related_fields:
        batch_process(field)
    return data

def process_timestamps(data):
    tqdm.write("\nProcessing timestamps...")
    if 'unixReviewTime' in data.columns:
        data['_raw_unixReviewTime'] = pd.to_numeric(data['unixReviewTime'], errors='coerce')
        data['unixReviewTime'] = pd.to_datetime(data['unixReviewTime'], unit='s', errors='coerce').dt.strftime('%Y‑%m‑%d %H:%M:%S').fillna('unknown')
        invalid_count = (data['unixReviewTime'] == 'unknown').sum()
        if invalid_count > 0:
            tqdm.write(f"   - {invalid_count} invalid unixReviewTime")
    if 'reviewTime' in data.columns:
        data['reviewTime'] = pd.to_datetime(data['reviewTime'], errors='coerce').dt.strftime('%Y‑%m‑%d').fillna('unknown')
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce').dt.strftime('%Y‑%m‑%d').fillna('unknown')
    return data

def is_valid_boolean_value(x):
    try:
        if isinstance(x, np.ndarray) and x.dtype == bool:
            return x.any() if len(x) > 0 else False
        if isinstance(x, list):
            bool_values = [v for v in x if isinstance(v, bool)]
            return any(bool_values) if bool_values else False
        if isinstance(x, bool):
            return x
        return bool(x)
    except:
        return False

def process_nan(data):
    tqdm.write("Starting missing value handling...")
    if 'style' in data.columns:
        data['style'] = data['style'].apply(process_style_field)
        unknown_count = (data['style'] == 'unknown').sum()
        unknown_rate = round(unknown_count / len(data['style']) * 100, 2)
        tqdm.write(f"   - style: unknown {unknown_count}/{len(data['style'])} ({unknown_rate}%)")
        non_unknown = data[data['style'] != 'unknown']
        if len(non_unknown) > 0:
            tqdm.write(f"   - style sample: {non_unknown['style'].iloc[0][:100]}")
    bool_fields = []
    for col in data.columns:
        try:
            if pd.api.types.is_bool_dtype(data[col]):
                bool_fields.append(col)
        except:
            continue
    for field in bool_fields:
        try:
            mask = data[field].isna()
            if not pd.api.types.is_bool_dtype(data[field]):
                valid_mask = data[field].apply(is_valid_boolean_value)
                mask = mask | (~valid_mask)
            data.loc[mask, field] = False
            tqdm.write(f"   - bool field [{field}]: fill na with False")
        except Exception as e:
            tqdm.write(f"   - error processing bool field [{field}]: {str(e)}, fallback fill False")
            data[field] = data[field].fillna(False)
    known_nested_fields = ['description', 'details', 'feature', 'style', 'category', 'also_buy', 'also_view', 'similar_item']
    nested_fields = [col for col in data.columns if col in known_nested_fields]
    for field in nested_fields:
        try:
            results = []
            for idx, val in enumerate(tqdm(data[field], desc=f"Process {field}")):
                try:
                    processed = process_nested_field(val)
                    results.append(processed)
                except Exception as e:
                    tqdm.write(f"   - record {idx} field {field} error: {str(e)}")
                    results.append('unknown')
            data[field] = results
            unknown_count = (data[field] == 'unknown').sum()
            total_count = len(data[field])
            unknown_rate = round(unknown_count / total_count * 100, 2)
            tqdm.write(f"   - nested field [{field}]: unknown {unknown_count}/{total_count} ({unknown_rate}%)")
        except Exception as e:
            tqdm.write(f"   - error processing nested field [{field}]: {str(e)}, fallback fill unknown")
            data[field] = 'unknown'
    text_fields = [col for col in data.columns if col not in bool_fields + nested_fields and data[col].dtype == object]
    for field in text_fields:
        tqdm.write(f"   - start processing text field [{field}]")
        try:
            processed_values = []
            for idx, val in enumerate(tqdm(data[field], desc=f"Clean {field}")):
                try:
                    processed = clean_text_blank_lines(val)
                    processed_values.append(processed)
                except Exception as e:
                    tqdm.write(f"   - record {idx} field {field} error: {str(e)} | preview: {str(val)[:50]}")
                    processed_values.append('unknown')
            data[field] = processed_values
            unknown_count = (data[field] == 'unknown').sum()
            total_count = len(data[field])
            unknown_rate = round(unknown_count / total_count * 100, 2)
            tqdm.write(f"   - text field [{field}]: unknown {unknown_count}/{total_count} ({unknown_rate}%)")
        except Exception as e:
            tqdm.write(f"   - error processing text field [{field}]: {str(e)}, fallback fill unknown")
            data[field] = 'unknown'
    if 'rank' in data.columns:
        data['rank'] = data['rank'].apply(clean_rank)
        unknown_count = (data['rank'] == 'unknown').sum()
        total_count = len(data['rank'])
        unknown_rate = round(unknown_count / total_count * 100, 2)
        tqdm.write(f"   - rank: unknown {unknown_count}/{total_count} ({unknown_rate}%)")
    tqdm.write("Missing value handling finished")
    return data

def price_process(price_str):
    if price_str == 'unknown' or pd.isna(price_str):
        return None
    price_str = str(price_str).strip()
    price_nums = re.findall(r"\d+\.?\d*", price_str)
    if not price_nums:
        return None
    if '-' in price_str and len(price_nums) >= 2:
        price = (float(price_nums[0]) + float(price_nums[1])) / 2
    else:
        price = float(price_nums[0])
    return round(price, 2)

def get_user_history_feature(data, time_window):
    tqdm.write("\nGenerating user history features...")
    if '_raw_unixReviewTime' in data.columns:
        data = data.sort_values(by=['user_id', '_raw_unixReviewTime']).reset_index(drop=True)
    else:
        data = data.sort_values(by=['user_id']).reset_index(drop=True)
        tqdm.write("Warning: missing valid timestamp, sort only by user_id")
    if 'rating' not in data.columns:
        raise ValueError("Missing 'rating' column, cannot generate user history")
    if data['rating'].sum() == 0:
        tqdm.write("Warning: no positive samples, skip user history generation")
        data['user_hist_id'] = 'unknown'
        data['user_hist_title'] = 'unknown'
        return data
    tqdm.write("   - Build asin‑title mapping...")
    id_title_map = {}
    title_groups = data.groupby('asin')['title'].apply(lambda x: next((t for t in x if t not in ['unknown', '']), 'unknown'))
    for asin_encoded, title in title_groups.items():
        cleaned_title = title.replace(';', ',').replace('and', 'and').strip()
        id_title_map[asin_encoded] = cleaned_title
    tqdm.write("   - Extract user positive item sequences...")
    pos_data = data[data['rating'] == 1][['user_id', 'asin', '_raw_unixReviewTime']].copy()
    user_pos_history = pos_data.groupby('user_id').apply(lambda x: x.sort_values('_raw_unixReviewTime')['asin'].tolist()).reset_index(name='pos_items')
    user_pos_dict = dict(zip(user_pos_history['user_id'], user_pos_history['pos_items']))
    data['user_hist_id'] = 'unknown'
    data['user_hist_title'] = 'unknown'
    unique_users = data['user_id'].unique()
    user_progress = tqdm(total=len(unique_users), desc="Process users")
    for user_id in unique_users:
        user_mask = data['user_id'] == user_id
        if '_raw_unixReviewTime' in data.columns:
            user_records = data[user_mask].sort_values('_raw_unixReviewTime').reset_index(drop=True)
        else:
            user_records = data[user_mask].reset_index(drop=True)
        if len(user_records) <= 1:
            user_progress.update(1)
            continue
        user_pos_items = user_pos_dict.get(user_id, [])
        if not user_pos_items:
            user_progress.update(1)
            continue
        encoded_id_buffer = []
        title_buffer = []
        for idx, row in user_records.iterrows():
            if idx == 0:
                continue
            prev_idx = user_records.index.get_loc(idx) - 1
            prev_row = user_records.iloc[prev_idx]
            if prev_row['rating'] == 1:
                encoded_id_buffer.append(prev_row['asin'])
                title_buffer.append(id_title_map.get(prev_row['asin'], 'unknown'))
                if len(encoded_id_buffer) > time_window:
                    encoded_id_buffer = encoded_id_buffer[-time_window:]
                    title_buffer = title_buffer[-time_window:]
            if encoded_id_buffer:
                user_records.loc[idx, 'user_hist_id'] = ' and '.join([f'product_{num}' for num in encoded_id_buffer])
                user_records.loc[idx, 'user_hist_title'] = ' and '.join([f"product '{title}'" for title in title_buffer])
            else:
                prev_hist_id = user_records.iloc[prev_idx]['user_hist_id']
                prev_hist_title = user_records.iloc[prev_idx]['user_hist_title']
                if prev_hist_id != 'unknown':
                    id_list = prev_hist_id.split(' and ')
                    title_list = prev_hist_title.split(' and ')
                    id_list = id_list[-time_window:]
                    title_list = title_list[-time_window:]
                    user_records.loc[idx, 'user_hist_id'] = ' and '.join(id_list)
                    user_records.loc[idx, 'user_hist_title'] = ' and '.join(title_list)
                else:
                    user_records.loc[idx, 'user_hist_id'] = 'unknown'
                    user_records.loc[idx, 'user_hist_title'] = 'unknown'
        data.loc[user_mask, ['user_hist_id', 'user_hist_title']] = user_records[['user_hist_id', 'user_hist_title']].values
        user_progress.update(1)
    user_progress.close()
    valid_hist_count = (data['user_hist_id'] != 'unknown').sum()
    valid_rate = round(valid_hist_count / len(data) * 100, 2)
    tqdm.write(f"User history finished (valid ratio: {valid_rate}%)")
    return data

def create_global_encoders(all_review_data, all_meta_data, encoder_dir):
    print("\n" + "=" * 50)
    print("Create global user and asin encoders")
    print("=" * 50)
    try:
        os.makedirs(encoder_dir, exist_ok=True)
        print(f"Encoder dir created/exists: {encoder_dir}")
    except Exception as e:
        print(f"Failed to create encoder dir: {str(e)}")
        raise
    all_user_ids = []
    for df in all_review_data:
        if 'reviewerID' in df.columns:
            all_user_ids.extend(df['reviewerID'].unique())
    all_user_ids = list(set(all_user_ids))
    if 'unknown_user' not in all_user_ids:
        all_user_ids.append('unknown_user')
    all_asin = []
    for df in all_review_data:
        if 'asin' in df.columns:
            all_asin.extend(df['asin'].unique())
    for df in all_meta_data:
        if 'asin' in df.columns:
            all_asin.extend(df['asin'].unique())
    all_asin = list(set(all_asin))
    if 'unknown_asin' not in all_asin:
        all_asin.append('unknown_asin')
    try:
        user_encoder = LabelEncoder()
        user_encoder.fit(all_user_ids)
        joblib.dump(user_encoder, os.path.join(encoder_dir, "global_user_encoder.pkl"))
        if os.path.exists(os.path.join(encoder_dir, "global_user_encoder.pkl")):
            print(f"User encoder saved ({len(user_encoder.classes_)} unique users)")
        else:
            raise FileNotFoundError("User encoder dumped but file missing")
    except Exception as e:
        print(f"Failed saving user encoder: {str(e)}")
        raise
    try:
        asin_encoder = LabelEncoder()
        asin_encoder.fit(all_asin)
        joblib.dump(asin_encoder, os.path.join(encoder_dir, "global_asin_encoder.pkl"))
        if os.path.exists(os.path.join(encoder_dir, "global_asin_encoder.pkl")):
            print(f"Asin encoder saved ({len(asin_encoder.classes_)} unique items)")
        else:
            raise FileNotFoundError("Asin encoder dumped but file missing")
    except Exception as e:
        print(f"Failed saving asin encoder: {str(e)}")
        raise
    return user_encoder, asin_encoder

def process_single_scenario(
        review_path: str,
        meta_path: str,
        encoder_dir: str,
        save_path: str,
        scenario_id: int,
        user_encoder=None,
        asin_encoder=None,
        time_window: int = 3,
        sample_size: int = None,
        skip_if_exists: bool = False
):
    if skip_if_exists and os.path.exists(save_path):
        print("=" * 50)
        scenario_name = SCENARIO_MAPPING.get(scenario_id, f"Unknown(ID:{scenario_id})")
        print(f"Skip {scenario_name} (ID: {scenario_id}) - file already exists")
        print(f"Existing path: {save_path}")
        print("=" * 50)
        return None, None
    print("=" * 50)
    scenario_name = SCENARIO_MAPPING.get(scenario_id, f"Unknown(ID:{scenario_id})")
    print(f"Start processing scenario: {scenario_name} (ID: {scenario_id})")
    print(f"Output path: {save_path}")
    if sample_size:
        print(f"Sample mode: only load first {sample_size} records")
    print("=" * 50)
    print(f"\n1. Load review file: {os.path.basename(review_path)}")
    try:
        if not os.path.exists(review_path):
            raise FileNotFoundError(f"Review file not found: {review_path}")
        review_records = []
        with open(review_path, 'r', encoding='utf‑8') as f:
            for idx, line in enumerate(tqdm(f, desc="Load review json")):
                if sample_size and idx >= sample_size:
                    break
                try:
                    review_records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        review_data = pd.DataFrame(review_records)
        drop_cols = [col for col in review_data.columns if col in EXCLUDE_ATTRIBUTES]
        if drop_cols:
            review_data = review_data.drop(columns=drop_cols)
            tqdm.write(f"   - Drop excluded columns: {drop_cols}")
        review_data['scenario_id'] = scenario_id
        review_data['Domain'] = scenario_name
        print(f"   Review rows: {len(review_data)}")
    except Exception as e:
        print(f"   Failed load review: {str(e)}")
        return review_data, None
    print(f"2. Load meta file: {os.path.basename(meta_path)}")
    try:
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        meta_records = []
        with open(meta_path, 'r', encoding='utf‑8') as f:
            for idx, line in enumerate(tqdm(f, desc="Load meta json")):
                if sample_size and idx >= sample_size:
                    break
                try:
                    meta_records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        meta_data = pd.DataFrame(meta_records)
        drop_cols = [col for col in meta_data.columns if col in EXCLUDE_ATTRIBUTES]
        if drop_cols:
            meta_data = meta_data.drop(columns=drop_cols)
            tqdm.write(f"   - Drop excluded columns: {drop_cols}")
        print(f"   Meta rows: {len(meta_data)}")
    except Exception as e:
        print(f"   Failed load meta: {str(e)}")
        return review_data, None
    print(f"\n3. Run preprocessing")
    try:
        print(f"   Process meta data...")
        meta_data = process_nan(meta_data)
        meta_data = meta_data.drop_duplicates(subset=['asin'], keep='first')
        print(f"   Meta after dedup: {len(meta_data)}")
        review_data = process_nan(review_data)
        print(f"   Review preprocessing finished (rows: {len(review_data)})")
    except Exception as e:
        print(f"   Preprocessing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return review_data, meta_data
    print(f"\n4. Merge review and meta")
    try:
        join_data = pd.merge(left=review_data, right=meta_data, on='asin', how='left')
        join_data = process_nan(join_data)
        print(f"   Merged rows: {len(join_data)}")
        print(f"   Merged columns: {len(join_data.columns)}")
    except Exception as e:
        print(f"   Merge error: {str(e)}")
        return review_data, meta_data
    print(f"5. Price cleaning and fill")
    try:
        if 'price' in join_data.columns:
            join_data['price'] = join_data['price'].apply(price_process)
            if join_data['price'].dropna().empty:
                join_data['price'] = 0.0
                print(f"   All price invalid, fill with 0.0")
            else:
                price_mean = round(join_data['price'].dropna().mean(), 2)
                join_data['price'] = join_data['price'].fillna(price_mean)
                print(f"   Price processed (mean: {price_mean})")
        else:
            print(f"   No price column, fill price=0.0")
            join_data['price'] = 0.0
    except Exception as e:
        print(f"   Price processing error: {str(e)}")
        join_data['price'] = 0.0
    print(f"6. Process timestamps")
    try:
        join_data = process_timestamps(join_data)
        print(f"   Timestamp processing finished")
    except Exception as e:
        print(f"   Timestamp error: {str(e)}")
        return review_data, meta_data
    print(f"7. User‑item encoding")
    try:
        if user_encoder is None:
            user_encoder = joblib.load(os.path.join(encoder_dir, "global_user_encoder.pkl"))
        if asin_encoder is None:
            asin_encoder = joblib.load(os.path.join(encoder_dir, "global_asin_encoder.pkl"))
        join_data['reviewerID'] = join_data['reviewerID'].fillna('unknown_user')
        unknown_users_mask = ~join_data['reviewerID'].isin(user_encoder.classes_)
        if unknown_users_mask.sum() > 0:
            tqdm.write(f"   {unknown_users_mask.sum()} unseen users, map to unknown_user")
            join_data.loc[unknown_users_mask, 'reviewerID'] = 'unknown_user'
        join_data['user_id'] = user_encoder.transform(join_data['reviewerID'])
        join_data['user_name'] = join_data.get('reviewerName', 'unknown')
        join_data['asin_original'] = join_data['asin']
        join_data['asin'] = join_data['asin'].fillna('unknown_asin')
        unknown_asin_mask = ~join_data['asin'].isin(asin_encoder.classes_)
        if unknown_asin_mask.sum() > 0:
            tqdm.write(f"   {unknown_asin_mask.sum()} unseen asins, map to unknown_asin")
            join_data.loc[unknown_asin_mask, 'asin'] = 'unknown_asin'
        join_data['asin'] = asin_encoder.transform(join_data['asin'])
        for drop_col in ['reviewerID', 'reviewerName']:
            if drop_col in join_data.columns:
                join_data = join_data.drop(columns=[drop_col])
        print(f"   Encoding done (users:{len(user_encoder.classes_)}, items:{len(asin_encoder.classes_)})")
    except Exception as e:
        print(f"   Encoding error: {str(e)}")
        return review_data, meta_data
    print(f"\n8. Process related‑item fields")
    try:
        join_data = process_related_fields(join_data, asin_encoder)
    except Exception as e:
        print(f"   Related‑item fields error: {str(e)}")
        try:
            unknown_idx = list(asin_encoder.classes_).index('unknown_asin')
            for field in ['also_buy', 'also_view', 'similar_item']:
                if field not in join_data.columns:
                    join_data[field] = f"product_{unknown_idx}"
        except:
            pass
    print(f"\n9. Generate positive label")
    if 'overall' in join_data.columns:
        join_data['rating'] = join_data['overall'].apply(lambda x: 1 if x > 3 else 0)
        join_data['label'] = join_data['rating']
        neutral_mask = join_data['overall'] == 3
        neutral_cnt = neutral_mask.sum()
        if neutral_cnt > 0:
            tqdm.write(f"   Drop {neutral_cnt} neutral samples (overall=3)")
            join_data = join_data[~neutral_mask].copy()
        positive_ratio = round(join_data['rating'].mean() * 100, 2)
        print(f"   Positive sample ratio(after drop neutral): {positive_ratio}%")
    else:
        print("   Missing 'overall', cannot generate rating label")
        join_data['rating'] = 0
        join_data['label'] = 0
    print(f"10. Generate user history sequence (window={time_window})")
    try:
        join_data = get_user_history_feature(join_data, time_window=time_window)
        non_unknown_hist = join_data[join_data['user_hist_id'] != 'unknown']
        if len(non_unknown_hist) > 0:
            print(f"   history_id sample: {non_unknown_hist['user_hist_id'].iloc[0][:100]}...")
            print(f"   history_title sample: {non_unknown_hist['user_hist_title'].iloc[0][:100]}...")
    except Exception as e:
        print(f"   User history generation error: {str(e)}")
        join_data['user_hist_id'] = 'unknown'
        join_data['user_hist_title'] = 'unknown'
    for col in FINAL_COLS:
        if col not in join_data.columns:
            if col == "price":
                join_data[col] = 0.0
            elif col == "scenario_id":
                join_data[col] = -1
            else:
                join_data[col] = "unknown"
    join_data = join_data[FINAL_COLS].copy()
    print(f"\n11. Save processed dataset")
    try:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"   Create output dir: {save_dir}")
        join_data.to_csv(save_path, index=False, encoding='utf‑8‑sig')
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            print(f"   Saved successfully! Path: {save_path}")
            print(f"   File size: {os.path.getsize(save_path)/1024:.2f} KB")
            print(f"   Final rows: {len(join_data)}")
            print(f"   Final columns: {list(join_data.columns)}")
        else:
            raise Exception("Saved file is empty or missing")
    except Exception as e:
        print(f"   Save error: {str(e)}")
        return review_data, meta_data
    print("\n" + "=" * 50)
    print(f"{scenario_name} finished!")
    print("=" * 50)
    return review_data, meta_data

def process_all_scenarios(
        datasets_config,
        encoder_dir,
        output_base_dir,
        time_window=3,
        regenerate_encoders=False,
        sample_size=None,
        skip_existing=True
):
    print("=" * 80)
    print("Start processing all amazon multi‑domain datasets")
    if sample_size:
        print(f"Sample mode: load first {sample_size} records per domain")
    print("=" * 80)
    all_review_data = []
    all_meta_data = []
    if regenerate_encoders or (not os.path.exists(os.path.join(encoder_dir, "global_user_encoder.pkl"))) or (not os.path.exists(os.path.join(encoder_dir, "global_asin_encoder.pkl"))):
        print("\n" + "=" * 50)
        print("Stage 1/2: load raw data to build global encoders")
        print("=" * 50)
        for scenario_id, config in datasets_config.items():
            print(f"\nLoad {SCENARIO_MAPPING[scenario_id]} (ID:{scenario_id}) ...")
            try:
                review_records = []
                with open(config["review_path"], 'r', encoding='utf‑8') as f:
                    for idx, line in enumerate(tqdm(f, desc="Load review for encoder")):
                        if sample_size and idx >= sample_size:
                            break
                        try:
                            review_records.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
                df_rev = pd.DataFrame(review_records)
                keep_cols = ['reviewerID', 'asin']
                df_rev = df_rev[[c for c in keep_cols if c in df_rev.columns]]
                all_review_data.append(df_rev)
                print(f"   Review loaded ({len(df_rev)} rows)")
            except Exception as e:
                print(f"   Failed load review: {str(e)}")
            try:
                meta_records = []
                with open(config["meta_path"], 'r', encoding='utf‑8') as f:
                    for idx, line in enumerate(tqdm(f, desc="Load meta for encoder")):
                        if sample_size and idx >= sample_size:
                            break
                        try:
                            meta_records.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
                df_meta = pd.DataFrame(meta_records)
                if 'asin' in df_meta.columns:
                    df_meta = df_meta[['asin']]
                    all_meta_data.append(df_meta)
                    print(f"   Meta loaded ({len(df_meta)} rows)")
            except Exception as e:
                print(f"   Failed load meta: {str(e)}")
        user_encoder, asin_encoder = create_global_encoders(all_review_data, all_meta_data, encoder_dir)
    else:
        print("\n" + "=" * 50)
        print("Stage 1/2: reuse existing global encoders")
        print("=" * 50)
        user_encoder = joblib.load(os.path.join(encoder_dir, "global_user_encoder.pkl"))
        asin_encoder = joblib.load(os.path.join(encoder_dir, "global_asin_encoder.pkl"))
        print(f"Load user encoder ({len(user_encoder.classes_)} classes)")
        print(f"Load asin encoder ({len(asin_encoder.classes_)} classes)")
    print("\n" + "=" * 50)
    print("Stage 2/2: process each domain")
    print("=" * 50)
    for scenario_id, config in datasets_config.items():
        scenario_name = SCENARIO_MAPPING[scenario_id]
        save_path = os.path.join(output_base_dir, f"{config['name']}_processed.csv")
        process_single_scenario(
            review_path=config["review_path"],
            meta_path=config["meta_path"],
            encoder_dir=encoder_dir,
            save_path=save_path,
            scenario_id=scenario_id,
            user_encoder=user_encoder,
            asin_encoder=asin_encoder,
            time_window=time_window,
            sample_size=sample_size,
            skip_if_exists=skip_existing
        )
    print("\n" + "=" * 80)
    print("All domains preprocessing finished!")
    print("=" * 80)

if __name__ == '__main__':
    DATASETS_CONFIG = {
        0: {
            "name": "Amazon_Fashion",
            "review_path": "./datasets/amazon_review/raw/reviews_Amazon_Fashion.json",
            "meta_path": "./datasets/amazon_review/raw/meta_Amazon_Fashion.json",
        },
        1: {
            "name": "Digital_Music",
            "review_path": "./datasets/amazon_review/raw/reviews_Digital_Music.json",
            "meta_path": "./datasets/amazon_review/raw/meta_Digital_Music.json",
        },
        2: {
            "name": "Musical_Instruments",
            "review_path": "./datasets/amazon_review/raw/reviews_Musical_Instruments.json",
            "meta_path": "./datasets/amazon_review/raw/meta_Musical_Instruments.json",
        },
        3: {
            "name": "Gift_Cards",
            "review_path": "./datasets/amazon_review/raw/reviews_Gift_Cards.json",
            "meta_path": "./datasets/amazon_review/raw/meta_Gift_Cards.json",
        },
        4: {
            "name": "All_Beauty",
            "review_path": "./datasets/amazon_review/raw/reviews_All_Beauty.json",
            "meta_path": "./datasets/amazon_review/raw/meta_All_Beauty.json",
        }
    }
    ENCODER_DIR = "./datasets/amazon_review/global_encoders"
    OUTPUT_BASE_DIR = "./datasets/amazon_review/processed_data"
    TIME_WINDOW = 3
    REGENERATE_ENCODERS = True
    SAMPLE_SIZE = None
    SKIP_EXISTING = False
    process_all_scenarios(
        datasets_config=DATASETS_CONFIG,
        encoder_dir=ENCODER_DIR,
        output_base_dir=OUTPUT_BASE_DIR,
        time_window=TIME_WINDOW,
        regenerate_encoders=REGENERATE_ENCODERS,
        sample_size=SAMPLE_SIZE,
        skip_existing=SKIP_EXISTING
    )
