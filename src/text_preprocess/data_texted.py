# -*- coding: utf-8 -*-
"""
Build structured prompt text from amazon review csv splits.
Input:  ./datasets/amazon_review_data/data_splits/{domain}/{domain}_{split}.csv
Output: ./datasets/amazon_review_data/data_text/{domain}/{domain}_{split}_text.txt
Relative‑path only. Handle nan/empty values, sanitize line breaks and quotes.
"""
import pandas as pd
import os
import time

PROMPT_USE_COLS = [
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

def click_process(user_hist_str):
    if pd.isna(user_hist_str) or str(user_hist_str).strip().lower() in ["", "nan", "none", "unknown"]:
        return "nothing"
    else:
        return str(user_hist_str)

def clean_line(s: str):
    if pd.isna(s):
        return ""
    return str(s).replace("\n", " ").replace("\r", " ")

def build_prompt(row, columns):
    scenario = "Unknown Scenario"
    if "Domain" in columns and not pd.isna(row["Domain"]):
        scenario_val = clean_line(str(row["Domain"]).strip())
        if scenario_val.lower() not in ["", "nan", "none", "unknown"]:
            scenario = scenario_val
    elif 'scenario_id' in columns and not pd.isna(row['scenario_id']):
        scenario_id_mapping = {
            0: "Amazon Fashion",
            1: "Digital Music",
            2: "Musical Instruments",
            3: "Gift Cards",
            4: "All Beauty"
        }
        scenario_id = int(row['scenario_id'])
        scenario = scenario_id_mapping.get(scenario_id, f"Scenario {scenario_id}")

    user_id = clean_line(str(row['user_id'])) if 'user_id' in columns else "unknown"

    username = row['user_name'] if 'user_name' in columns else None
    processed_username = click_process(username)
    if processed_username == "nothing":
        processed_username = "anonymous"
    processed_username = clean_line(processed_username)

    hist_id = row['user_hist_id'] if "user_hist_id" in columns else None
    hist_title = row['user_hist_title'] if "user_hist_title" in columns else None
    processed_hist_id = clean_line(click_process(hist_id))
    processed_hist_title = clean_line(click_process(hist_title))

    asin = clean_line(str(row['asin'])) if 'asin' in columns else "unknown"

    item_info_parts = [f"The ID of current product is product_{asin}"]

    item_attributes = [
        ("title", "the title is '{}'"),
        ("brand", "the brand is '{}'"),
        ("price", "the price is ${:.2f}"),
        ("similar_item", "similar items are {}"),
        ("rank", "the rank is '{}'"),
        ("category", "the category is '{}'"),
        ("reviewText", "the review text is '{}'"),
        ("unixReviewTime", "the Unix review time is '{}'"),
        ("also_buy", "customers also bought {}"),
        ("also_view", "customers also viewed {}"),
        ("description", "the description is '{}'"),
        ("details", "the details are '{}'"),
        ("feature", "the feature is '{}'"),
        ("style", "the style is '{}'")
    ]

    for attr in item_attributes:
        field_name = attr[0]
        if field_name not in columns or field_name not in PROMPT_USE_COLS:
            continue
        value = row[field_name]

        if field_name == 'price':
            try:
                price_val = float(value)
                if pd.isna(price_val):
                    continue
            except (ValueError, TypeError):
                continue
        else:
            if pd.isna(value) or str(value).strip().lower() in ["", "nan", "none", "unknown"]:
                continue

        str_value = clean_line(str(value)).replace("'", "\"")
        if field_name == 'price':
            item_info_parts.append(attr[1].format(float(value)))
        else:
            item_info_parts.append(attr[1].format(str_value))

    scenario_info = f"{scenario}: "
    base_user_info = f"The user is {processed_username} (ID: user_{user_id})"
    if processed_hist_id == "nothing" and processed_hist_title == "nothing":
        user_info_combined = f"{base_user_info}. "
    else:
        user_info_combined = f"{base_user_info}, who clicked IDs {processed_hist_id} and titles {processed_hist_title} recently. "
    item_info = ", ".join(item_info_parts) + ". "

    full_text = scenario_info + user_info_combined + item_info
    return full_text


def data_process_amazon_ctr(data_path, data_source, split_type):
    start_time = time.time()
    output_dir = os.path.join("./datasets/amazon_review_data/data_text", data_source)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    columns = df.columns.tolist()
    combined_file_path = os.path.join(output_dir, f"{data_source}_{split_type}_text.txt")

    fail_count = 0
    with open(combined_file_path, 'w+', encoding='utf-8') as fout:
        for _, row in df.iterrows():
            try:
                text_line = build_prompt(row, columns)
                fout.write(text_line + "\n")
            except Exception:
                fail_count += 1
    end_time = time.time()
    print(f"Total records: {len(df)}, skip corrupted lines: {fail_count}")
    print(f"Output file: {combined_file_path}, elapsed {end_time - start_time:.2f}s")


def process_all_split_datasets():
    dataset_names = [
        "All_Beauty",
        "Amazon_Fashion",
        "Digital_Music",
        "Gift_Cards",
        "Musical_Instruments"
    ]
    split_types = ["train", "valid", "test"]
    data_root = "./datasets/amazon_review_data/data_splits"

    for dataset in dataset_names:
        print(f"\n===== Process {dataset} =====")
        dataset_dir = os.path.join(data_root, dataset)
        for split_type in split_types:
            print(f"\n-- {split_type} --")
            csv_path = os.path.join(dataset_dir, f"{dataset}_{split_type}.csv")
            if not os.path.exists(csv_path):
                print(f"skip, file not found: {csv_path}")
                continue
            data_process_amazon_ctr(csv_path, dataset, split_type)


if __name__ == "__main__":
    process_all_split_datasets()
    print("\nAll done.")
