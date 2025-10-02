# train_bert.py
import json
import os
import difflib
import bisect
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, concatenate_datasets
import evaluate
from sklearn.model_selection import train_test_split

# --- Data Preparation ---

def generate_bio_labels(tokens, corrected):
    """Dynamically assign BIO labels by matching corrected values to tokens."""
    token_texts = [t["text"] for t in tokens]
    full_text = " ".join(token_texts)
    labels = ["O"] * len(tokens)

    # Build cumulative char positions for quick token lookup
    cum_chars = [0]
    for t in token_texts:
        cum_chars.append(cum_chars[-1] + len(t) + 1)

    def assign_labels(entity_type, value):
        if not isinstance(value, str):
            value = str(value)
        if not value.strip():
            return # Skip empty values

        matcher = difflib.SequenceMatcher(None, full_text, value, autojunk=False)
        match = matcher.find_longest_match(0, len(full_text), 0, len(value))

        # Use a threshold to ensure the match is good
        if match.size < len(value) * 0.8:
            return

        match_start = match.a
        match_end = match.a + match.size
        
        # Find token indices corresponding to the character match
        token_start = bisect.bisect_right(cum_chars, match_start) - 1
        token_end = bisect.bisect_left(cum_chars, match_end) - 1
        
        if token_start < 0: token_start = 0
        if token_end >= len(tokens): token_end = len(tokens) - 1

        if token_start <= token_end:
            labels[token_start] = f"B-{entity_type}"
            for i in range(token_start + 1, token_end + 1):
                labels[i] = f"I-{entity_type}"

    # Patient entities
    for key, row in corrected.get("patient", {}).items():
        entity_type = key.upper().replace(" ", "_")
        assign_labels(entity_type, row.get("Value", ""))

    # Tests entities
    for test in corrected.get("tests", []):
        if isinstance(test, dict):
            for col, val in test.items():
                # **FIX 1: Added "TESTS_" prefix to avoid label ambiguity (e.g., NAME vs TESTS_NAME)**
                col_type = f"TESTS_{col.upper().replace(' ', '_')}"
                assign_labels(col_type, val)

    # Additional sections
    for section_name, records in corrected.get("additional_sections", {}).items():
        section_type = section_name.upper().replace(" ", "_")
        for record in records:
            if isinstance(record, dict):
                for col, val in record.items():
                    col_type = f"{section_type}_{col.upper().replace(' ', '_')}"
                    assign_labels(col_type, val)
            elif isinstance(record, str):
                assign_labels(section_type, record)

    return labels

def prepare_dataset(corrections_dir="data/corrections"):
    """Load all correction JSONs and generate a Hugging Face Dataset."""
    data = []
    print(f"Loading data from: {corrections_dir}")
    json_files = [f for f in os.listdir(corrections_dir) if f.endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"No .json files found in '{corrections_dir}'. Please check the path.")
        
    print(f"Found {len(json_files)} correction files.")
    
    for file in json_files:
        with open(os.path.join(corrections_dir, file), "r") as f:
            corr = json.load(f)
            tokens = corr.get("original_tokens", [])
            if not tokens: continue

            labels = generate_bio_labels(tokens, corr["corrected_structured"])
            item = {
                "tokens": [t["text"] for t in tokens],
                "ner_tags": labels,
            }
            data.append(item)
            
    # Create a comprehensive label-to-ID mapping
    all_labels = sorted(list(set(tag for item in data for tag in item["ner_tags"])))
    label2id = {label: i for i, label in enumerate(all_labels)}
    
    # Convert string labels to integer IDs
    for item in data:
        item["ner_tags"] = [label2id[tag] for tag in item["ner_tags"]]
        
    return Dataset.from_list(data), label2id, {v: k for k, v in label2id.items()}

# --- Training ---

def train_model(dataset=None, label2id=None, id2label=None):
    """Initializes and runs the model training process."""
    if dataset is None:
        try:
            # Assuming your 26 files are in a directory named 'data/corrections'
            dataset, label2id, id2label = prepare_dataset(corrections_dir="data/corrections")
        except FileNotFoundError as e:
            print(e)
            return None, None, None

    if len(dataset) < 2:
        print("Error: Need at least 2 files for training and evaluation.")
        return None, None, None

    # Split dataset into training and evaluation sets
    train_ds, eval_ds = dataset.train_test_split(test_size=0.2, seed=42).values()

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def tokenize_and_align(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=512,
            padding="max_length"
        )
        labels = []
        for i, label_list in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_list[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    train_ds = train_ds.map(tokenize_and_align, batched=True)
    eval_ds = eval_ds.map(tokenize_and_align, batched=True)

    print(f"Number of labels: {len(label2id)}")
    print("Labels:", list(label2id.keys()))
    
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels=len(label2id), 
        id2label=id2label, 
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir="models/bert_lab_extractor",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4, # Increased batch size slightly
        per_device_eval_batch_size=4,
        num_train_epochs=20, # Increased epochs for small datasets
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        gradient_accumulation_steps=1,
    )

    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        """
        **FIX 2: Correctly align predictions and labels for seqeval.**
        This is the main fix for the ValueError you encountered.
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (-100) and convert IDs to label strings
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "f1": results["overall_f1"],
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "accuracy": results["overall_accuracy"],
        }

    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=train_ds, 
        eval_dataset=eval_ds, 
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    print("Starting training...")
    trainer.train()
    trainer.save_model("models/bert_lab_extractor/final")
    print("Training complete and model saved.")
    return model, tokenizer, id2label

# --- Inference to JSON ---
# Note: The inference part is for after you've trained a model.

def infer_and_structure_to_json(tokens, model_path="models/bert_lab_extractor/final"):
    """Infer labels from tokens and structure the output into a JSON object."""
    if not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        return {}
        
    model = BertForTokenClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    id2label = model.config.id2label

    inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True, truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    confidences = torch.softmax(logits, dim=2).max(dim=2).values

    # Align predictions with original tokens
    word_ids = inputs.word_ids()
    pred_labels = []
    pred_confs = []
    prev_word_id = None
    for i, word_id in enumerate(word_ids):
        if word_id is not None and word_id != prev_word_id:
            label_id = predictions[0][i].item()
            pred_labels.append(id2label[label_id])
            pred_confs.append(confidences[0][i].item())
        prev_word_id = word_id

    # Group BIO labels into structured entities
    structured = {"patient": {}, "tests": [], "additional_sections": {}}
    current_entity = []
    current_type = None
    current_conf = []

    for token, label, conf in zip(tokens, pred_labels, pred_confs):
        if label.startswith("B-"):
            if current_entity:
                avg_conf = sum(current_conf) / len(current_conf) if current_conf else 0.0
                add_to_structured(structured, current_type, " ".join(current_entity), avg_conf)
            current_type = label[2:]
            current_entity = [token]
            current_conf = [conf]
        elif label.startswith("I-") and current_type == label[2:]:
            current_entity.append(token)
            current_conf.append(conf)
        else:
            if current_entity:
                avg_conf = sum(current_conf) / len(current_conf) if current_conf else 0.0
                add_to_structured(structured, current_type, " ".join(current_entity), avg_conf)
            current_entity = None
            current_type = None
    
    if current_entity: # Add the last entity if it exists
        avg_conf = sum(current_conf) / len(current_conf) if current_conf else 0.0
        add_to_structured(structured, current_type, " ".join(current_entity), avg_conf)
        
    return structured

def add_to_structured(structured, entity_type, value, conf):
    """Dynamically add an extracted entity to the JSON structure."""
    # **FIX 3: Changed split delimiter from "" to "_" to correctly parse entity types.**
    parts = entity_type.split("_")
    
    if not parts: return

    base_type = parts[0]
    sub_type = "_".join(parts[1:]).lower()

    if base_type == "PATIENT":
        structured["patient"][sub_type] = {"value": value, "confidence": f"{conf:.2f}"}
    elif base_type == "TESTS":
        if not structured["tests"] or sub_type in structured["tests"][-1]:
            structured["tests"].append({})
        structured["tests"][-1][sub_type] = {"value": value, "confidence": f"{conf:.2f}"}
    else: # Handles all other sections like LAB_INFO
        section_name = base_type.lower()
        if section_name not in structured["additional_sections"]:
            structured["additional_sections"][section_name] = []
        
        if not structured["additional_sections"][section_name] or sub_type in structured["additional_sections"][section_name][-1]:
            structured["additional_sections"][section_name].append({})
        structured["additional_sections"][section_name][-1][sub_type] = {"value": value, "confidence": f"{conf:.2f}"}

if __name__ == "__main__":
    # To train the model, ensure your 26 JSON files are in a folder, e.g., 'data/corrections'
    train_model()

    # --- Example Inference Usage (after training) ---
    try:
        with open("data/ocr_tokens/doc_3_page_01.json", "r") as f:
            # This assumes a different JSON format for inference (only tokens)
            ocr_data = json.load(f)
            test_tokens = [t["text"] for t in ocr_data]
        json_output = infer_and_structure_to_json(test_tokens)
        print("\n--- Sample Inference Output ---")
        print(json.dumps(json_output, indent=2))
    except FileNotFoundError:
        print("\nSkipping inference example: 'data/ocr_tokens/doc_3_page_01.json' not found.")
    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")