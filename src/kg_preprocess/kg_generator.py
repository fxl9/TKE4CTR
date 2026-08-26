# -*- coding: utf-8 -*-
"""
Two‑stage KG triple generation with hallucination suppression.
Stage1: Structured constrained few‑shot prompt extract KG triples via local Ollama LLM.
Stage2: LLM‑based triple validation, max regenerate 3 times for invalid candidates.
"""
import re
import json
import os
import sys
from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

try:
    import csv
    csv.field_size_limit(100 * 1024 * 1024)
except OverflowError:
    csv.field_size_limit(sys.maxsize)


class StrictFormatKGGenerator:
    def __init__(self, category: str, dataset_type: str, model_name: str = "llama3:latest", max_retry: int = 3):
        self.category = category
        self.dataset_type = dataset_type
        self.max_retry = max_retry
        self.model = self._init_llm(model_name)
        self.line_triples_map: Dict[int, List[Dict]] = {}
        self.total_records = 0

        self.stats = {
            "total_text_blocks": 0,
            "blocks_with_triples": 0,
            "blocks_with_empty_triples": 0,
            "total_triples_generated": 0,
            "relation_type_count": {},
            "validation_stats": {
                "total_candidate_triples": 0,
                "valid_triples": 0,
                "invalid_triples": 0,
                "regenerate_attempts": 0
            }
        }

        self.generate_prompt = self._create_generation_prompt()
        self.validate_prompt = self._create_validation_prompt()

    def _init_llm(self, model_name: str) -> ChatOllama:
        llm = ChatOllama(
            model=model_name,
            temperature=0.0,
            max_tokens=4000,
            timeout=300,
            base_url="http://localhost:11434"
        )
        print(f"✅ LLM instance init done, model: {model_name}")
        return llm

    def _create_generation_prompt(self) -> PromptTemplate:
        test_restriction = "Test datasets are forbidden to extract click_history_id. Do NOT generate triples using review‑text‑related relations for test set." if self.dataset_type == "test" else ""
        template = f"""Task: Extract knowledge graph triples from the text as a JSON array.
Output Rules (MANDATORY, NO EXCEPTIONS):
1. Output ONLY a valid JSON array, no other content, no explanations, no comments, no markdown code blocks.
2. If no triples can be extracted, output an empty JSON array: [].
3. Each element in JSON array must be object with exactly three keys: "head", "relation", "tail".
4. Each key value must be non‑empty string.

Structured Data Relations:
- User: has_username, belongs_to_scenario
- Product: has_price, has_brand, has_title, belongs_to_category, has_style, has_rank
- Click: click_history_id (train ONLY, valid/test FORBIDDEN)

Review Text Relations:
- expresses_sentiment: user→sentiment→product
- compares_to: product→other_product/category
- suggests_preference: user→preference_type

Important Constraint: Review texts are only available in training set. Do NOT generate triples using these review‑text‑related relations for validation and test sets.
Mandatory KG Rules: {test_restriction}

Entity Rules:
- User ID: user_xxx (use actual ID from input text)
- Product ID: product_xxx (use actual ID from input text)
- Category: category_xxx (inferred from product title)
- Scenario: original name (e.g., Amazon Fashion)
- Sentiment: positive / negative / neutral
- Preference: value_focused / quality_focused / etc.

Few‑shot Example:
Input Text: The user is Julie (ID: user_83), who clicked IDs product_22, clicked titles "3‑Pack Sleep N Play (Terry)". The ID of the current product is product_68, the title is "5‑pack bodysuits", the brand is "Belocia", the price is "$27.00", it has similar items product_93. The review text is "I don't see the difference between these bodysuits and the more expensive ones."

Expected Output:
[
    {{"head":"user_83", "relation":"has_username", "tail":"Julie"}},
    {{"head":"user_83", "relation":"click_history_id", "tail":"product_22"}},
    {{"head":"user_83", "relation":"click_history_title", "tail":"3‑Pack Sleep N Play (Terry)"}},
    {{"head":"user_83", "relation":"expresses_sentiment", "tail":"neutral"}},
    {{"head":"user_83", "relation":"suggests_preference", "tail":"value_focused"}},
    {{"head":"product_68", "relation":"has_title", "tail":"5‑pack bodysuits"}},
    {{"head":"product_68", "relation":"has_price", "tail":"$27.00"}},
    {{"head":"product_68", "relation":"has_brand", "tail":"Belocia"}},
    {{"head":"product_68", "relation":"has_similar_item", "tail":"product_93"}},
    {{"head":"product_68", "relation":"compares_to", "tail":"premium_bodysuits"}}
]

Text:
{{text_chunk}}"""
        return PromptTemplate(input_variables=["text_chunk"], template=template)

    def _create_validation_prompt(self) -> PromptTemplate:
        """Prompt from tab:kg_validation_prompt for triple secondary validation."""
        template = """Task: Strictly validate the factual correctness of knowledge graph triples grounded only in the given source text. Reject over‑inference, semantic deviation and ungrounded extra information.
Input:
Source Text: {source_text}
Triple to Validate: (head:{head}, relation:{relation}, tail:{tail})

Validation Rules:
1. Check whether all abstract entities such as user_xxx and product_xxx exactly appear in the source text.
2. Verify whether the relation strictly belongs to the predefined allowed relation list.
3. The triple must be explicitly stated or strictly entailed by the source text.
4. Any triple with over‑inference, unmentioned extra facts or semantic inconsistency should be marked invalid.

Output: Only return JSON object in format {{"valid": true/false, "reason": "explanation", "confidence": 0.0‑1.0}}.
"""
        return PromptTemplate(input_variables=["source_text", "head", "relation", "tail"], template=template)

    def _validate_single_triple(self, source_text: str, triple: Dict) -> Dict:
        """Call LLM to validate one triple, return parsed validation dict."""
        head = triple.get("head", "")
        rel = triple.get("relation", "")
        tail = triple.get("tail", "")
        prompt = self.validate_prompt.format(source_text=source_text, head=head, relation=rel, tail=tail)
        try:
            resp = self.model.invoke(prompt)
            raw = resp.content.strip()
            raw = re.sub(r'^```(json)?\s*', '', raw, flags=re.IGNORECASE)
            raw = re.sub(r'\s*```$', '', raw)
            val_result = json.loads(raw)
            return val_result
        except Exception:
            return {"valid": False, "reason": "validation parse error", "confidence": 0.0}

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return ""
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        return cleaned_text

    def _generate_raw_triples(self, cleaned_text: str) -> List[Dict]:
        prompt = self.generate_prompt.format(text_chunk=cleaned_text)
        response = self.model.invoke(prompt)
        raw_output_original = response.content.strip() if hasattr(response, 'content') else ""
        raw_output = raw_output_original
        if not raw_output:
            return []
        raw_output = re.sub(r'^```(json)?\s*', '', raw_output, flags=re.IGNORECASE)
        raw_output = re.sub(r'\s*```$', '', raw_output)
        raw_output = re.sub(r',\s*([}\]])', r'\1', raw_output)
        raw_output = re.sub(r'\s+', ' ', raw_output)
        try:
            triples = json.loads(raw_output)
        except (json.JSONDecodeError, ValueError):
            raw_output = re.sub(r'[^[\]{}":,.\w\s]', '', raw_output)
            try:
                triples = json.loads(raw_output)
            except (json.JSONDecodeError, ValueError):
                return []
        if not isinstance(triples, list):
            triples = []
        return triples

    def extract_triples_from_chunk(self, text_chunk: str, line_number: int) -> None:
        self.stats["total_text_blocks"] += 1
        final_valid_triples = []
        cleaned_text = self.clean_text(text_chunk)
        if len(cleaned_text) > 8000:
            truncate_pos = cleaned_text.rfind(' ', 0, 8000)
            if truncate_pos == -1:
                truncate_pos = 8000
            cleaned_text = cleaned_text[:truncate_pos].strip()

        if line_number == 0:
            print(f"\n{'='*60} Line {line_number} raw input {'='*60}")
            print(cleaned_text[:1200])
            print(f"{'='*120}")

        # Two‑stage loop: generate‑validate‑regenerate up to max_retry times
        attempt = 0
        candidate_triples = []
        while attempt < self.max_retry:
            attempt += 1
            self.stats["validation_stats"]["regenerate_attempts"] += 1
            candidate_triples = self._generate_raw_triples(cleaned_text)
            if self.dataset_type == "test":
                forbidden_relations = {"click_history_id"}
                candidate_triples = [t for t in candidate_triples if isinstance(t, dict) and t.get("relation") not in forbidden_relations]
            if not candidate_triples:
                continue
            # validate each candidate triple
            round_valid = []
            for t in candidate_triples:
                if not isinstance(t, dict):
                    continue
                self.stats["validation_stats"]["total_candidate_triples"] += 1
                val_res = self._validate_single_triple(cleaned_text, t)
                if val_res.get("valid", False):
                    round_valid.append(t)
                    self.stats["validation_stats"]["valid_triples"] += 1
                else:
                    self.stats["validation_stats"]["invalid_triples"] += 1
            if len(round_valid) > 0:
                final_valid_triples = round_valid
                break

        # post‑process: auto‑complete belongs_to_scenario
        validated_triples = []
        user_ids = set()
        product_ids = set()
        click_product_map = {}
        for t in final_valid_triples:
            if not isinstance(t, dict):
                continue
            head = t.get("head", "")
            tail = t.get("tail", "")
            relation = t.get("relation", "")
            if isinstance(head, str) and head.startswith("user_"):
                user_ids.add(head)
            if isinstance(tail, str) and tail.startswith("user_"):
                user_ids.add(tail)
            if isinstance(head, str) and head.startswith("product_"):
                product_ids.add(head)
            if isinstance(tail, str) and tail.startswith("product_"):
                product_ids.add(tail)
            if relation == "click_history_id":
                click_product_map[tail] = head
            validated_triples.append(t)

        scenario_name = f"scenario_{self.category}"
        for user_id in user_ids:
            has_scenario = any(
                isinstance(t, dict) and t.get("head") == user_id and t.get("relation") == "belongs_to_scenario"
                for t in validated_triples
            )
            if not has_scenario:
                validated_triples.append({
                    "head": user_id,
                    "relation": "belongs_to_scenario",
                    "tail": scenario_name
                })

        product_title_map = {}
        for t in validated_triples:
            if not isinstance(t, dict):
                continue
            if t.get("relation") == "has_title" and isinstance(t.get("head"), str) and t["head"].startswith("product_"):
                product_title_map[t["head"]] = t.get("tail", "")

        for product_id in click_product_map.keys():
            if product_id not in product_title_map and isinstance(text_chunk, str):
                title_match = re.search(fr"{product_id}\s+title[:：](.+?)(\s|$)", text_chunk, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
                    validated_triples.append({
                        "head": product_id,
                        "relation": "has_title",
                        "tail": title
                    })
                    product_title_map[product_id] = title

        triples = validated_triples
        self.line_triples_map[line_number] = triples

        if len(triples) > 0:
            self.stats["blocks_with_triples"] += 1
            self.stats["total_triples_generated"] += len(triples)
            for t in triples:
                if not isinstance(t, dict):
                    continue
                rel_type = t.get("relation", "Unknown")
                self.stats["relation_type_count"][rel_type] = self.stats["relation_type_count"].get(rel_type, 0) + 1
        else:
            self.stats["blocks_with_empty_triples"] += 1

        if line_number % 50 == 0 and line_number > 0:
            print(f"Processed {line_number} chunks | valid triples:{self.stats['validation_stats']['valid_triples']} | total triples:{self.stats['total_triples_generated']}")

    def process_text_file(self, file_path: str, is_csv: bool = False, csv_text_column: str = "text") -> None:
        lines = []
        try:
            if is_csv:
                with open(file_path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    if csv_text_column not in reader.fieldnames and "reviewText" not in reader.fieldnames:
                        raise ValueError(f"CSV file missing required column: {csv_text_column} or reviewText")
                    for row in reader:
                        text_content = row.get(csv_text_column, row.get("reviewText", ""))
                        lines.append(self.clean_text(text_content))
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [self.clean_text(line) for line in f]
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return
        except Exception as e:
            print(f"Error: Failed to read file {file_path} - {str(e)}")
            return

        self.total_records = len(lines)
        print(f"Starting processing {self.category}-{self.dataset_type}: {self.total_records} text chunks, max retry={self.max_retry}")
        if self.dataset_type == "test":
            print(f"⚠️ TEST DATASET: Strictly no click_history_id & review‑based relations!")
        for line_num, text_chunk in enumerate(lines):
            self.extract_triples_from_chunk(text_chunk, line_num)

    def save_triples(self, output_dir: str) -> None:
        try:
            category_dir = os.path.join(output_dir, self.category)
            os.makedirs(category_dir, exist_ok=True)
            extracted_users = set()
            for line_triples in self.line_triples_map.values():
                for t in line_triples:
                    if isinstance(t, dict) and isinstance(t.get("head"), str) and t["head"].startswith("user_"):
                        extracted_users.add(t["head"])

            output_data = {
                "total_records": self.total_records,
                "triples_list": [],
                "user_statistics": {
                    "total_users_extracted": len(extracted_users),
                    "user_relation_breakdown": {
                        rt: self.stats["relation_type_count"].get(f"user_{rt}", 0)
                        for rt in ["has_username", "belongs_to_scenario", "click_history_id"]
                    },
                    "is_test_dataset": self.dataset_type == "test",
                    "test_set_click_restriction_applied": self.dataset_type == "test",
                    "standard_kg_structure": "user→product_id→title (no direct user→title)"
                },
                "two_stage_hal_suppression": {
                    "max_regenerate_retry": self.max_retry,
                    "validation_stats": self.stats["validation_stats"]
                }
            }
            for line_num in range(self.total_records):
                output_data["triples_list"].append({
                    "line_number": line_num,
                    "triples": self.line_triples_map.get(line_num, [])
                })

            output_file = os.path.join(category_dir, f"{self.category}_{self.dataset_type}_triples.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            stats_file = os.path.join(category_dir, f"{self.category}_{self.dataset_type}_stats.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)

            print(f"\n=== {self.category}-{self.dataset_type} Enhanced Summary ===")
            print(f"Total text chunks: {self.total_records}")
            print(f"Chunks with triples: {self.stats['blocks_with_triples']}")
            print(f"Total triples generated: {self.stats['total_triples_generated']}")
            print(f"Validation: total_candidate={self.stats['validation_stats']['total_candidate_triples']}, valid={self.stats['validation_stats']['valid_triples']}, invalid={self.stats['validation_stats']['invalid_triples']}")
            print(f"Triple file saved to: {output_file}")
            print("=" * 60)
        except Exception as e:
            print(f"Error: Failed to save triples - {str(e)}")


def process_all_datasets(text_root: str, kg_output_root: str, model_name: str = "llama3:latest",
                         test_mode: bool = False, max_test_chunks: int = 100, process_csv: bool = False, max_retry=3):
    dataset_names = ["All_Beauty", "Amazon_Fashion", "Digital_Music", "Gift_Cards", "Musical_Instruments"]
    split_types = ["train", "valid", "test"]
    global_user_stats = {
        "total_users_across_datasets": 0,
        "dataset_user_breakdown": {},
        "total_user_triples": 0,
        "train_valid_click_relations": 0,
        "total_product_title_relations": 0,
        "kg_structure_compliance": {"direct_user_title_relations": 0, "title_on_product_nodes": 0},
        "global_two_stage_stats": {"total_candidate":0, "total_valid":0, "total_invalid":0}
    }

    for dataset in dataset_names:
        print(f"\n{'=' * 70}")
        print(f"Processing Dataset: {dataset}, two‑stage hallucination suppression, max_retry={max_retry}")
        print(f"{'=' * 70}")
        dataset_user_count = 0
        dataset_user_triples = 0
        dataset_train_valid_clicks = 0
        dataset_product_titles = 0
        dataset_direct_user_titles = 0
        for split_type in split_types:
            if process_csv:
                text_file_name = f"{dataset}_{split_type}_text.csv"
            else:
                text_file_name = f"{dataset}_{split_type}_text.txt"
            text_file_path = os.path.join(text_root, dataset, text_file_name)
            if not os.path.exists(text_file_path):
                print(f"⚠️ File not found: {text_file_path} - Skipping")
                continue
            kg_generator = StrictFormatKGGenerator(category=dataset, dataset_type=split_type, model_name=model_name, max_retry=max_retry)
            if test_mode:
                print(f"\n🔹 TEST MODE: {dataset}-{split_type} (max {max_test_chunks} chunks)")
                lines = []
                try:
                    with open(text_file_path, 'r', encoding='utf-8') as f:
                        for idx, line in enumerate(f):
                            if idx >= max_test_chunks:
                                break
                            lines.append(kg_generator.clean_text(line))
                except Exception as e:
                    print(f"Error read test file: {str(e)}")
                    continue
                kg_generator.total_records = len(lines)
                for line_num, text_chunk in enumerate(lines):
                    kg_generator.extract_triples_from_chunk(text_chunk, line_num)
                kg_generator.save_triples(kg_output_root)
            else:
                print(f"\n🔹 FULL MODE: {dataset}-{split_type}")
                kg_generator.process_text_file(text_file_path, is_csv=process_csv)
                kg_generator.save_triples(kg_output_root)

            stats_file = os.path.join(kg_output_root, dataset, f"{dataset}_{split_type}_stats.json")
            if os.path.exists(stats_file):
                try:
                    with open(stats_file, 'r', encoding='utf-8') as f:
                        stats = json.load(f)
                    global_user_stats["global_two_stage_stats"]["total_candidate"] += stats["validation_stats"]["total_candidate_triples"]
                    global_user_stats["global_two_stage_stats"]["total_valid"] += stats["validation_stats"]["valid_triples"]
                    global_user_stats["global_two_stage_stats"]["total_invalid"] += stats["validation_stats"]["invalid_triples"]
                except Exception as e:
                    print(f"read stats warning: {e}")
                    continue

        global_stats_file = os.path.join(kg_output_root, "global_user_statistics.json")
        with open(global_stats_file, 'w', encoding='utf-8') as f:
            json.dump(global_user_stats, f, ensure_ascii=False, indent=2)
    print(f"\n✅ All datasets done. Global two‑stage stats: {global_user_stats['global_two_stage_stats']}")


if __name__ == "__main__":
    TEXT_ROOT = r"D:\CTR\Text2KG_CTR\datasets\amazon_review_data\data_text"
    KG_OUTPUT_ROOT = r"D:\CTR\Text2KG_CTR\datasets\amazon_review_data\kg_triples"
    MODEL_NAME = "llama3:latest"
    TEST_MODE = True
    MAX_TEST_CHUNKS = 100
    PROCESS_CSV = False
    MAX_RETRY = 3

    print(f"Two‑stage KG generation with hallucination suppression. max_retry={MAX_RETRY}")
    print(f"Test Mode: {TEST_MODE} | Max Test Chunks: {MAX_TEST_CHUNKS}")
    print(f"Model: {MODEL_NAME} (Local Ollama 11434)")
    process_all_datasets(
        text_root=TEXT_ROOT,
        kg_output_root=KG_OUTPUT_ROOT,
        model_name=MODEL_NAME,
        test_mode=TEST_MODE,
        max_test_chunks=MAX_TEST_CHUNKS,
        process_csv=PROCESS_CSV,
        max_retry=MAX_RETRY
    )
