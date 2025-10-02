import json
import os
import difflib
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import evaluate

# --- Data Preparation ---

def generate_bio_labels(tokens, corrected):
    """Assign BIO labels from corrected JSON, handling nested sections like subtests and additional_sections."""
    token_texts = [t["text"] for t in tokens]
    labels = ["O"] * len(tokens)

    def assign_labels_bbox_aware(entity_type, value):
        """Match value text to tokens and assign BIO labels using fuzzy matching."""
        if not isinstance(value, str):
            value = str(value)
        if not value.strip():
            return

        target_tokens = value.split()
        best_match_indices, best_score = None, -1

        for i in range(len(token_texts)):
            if token_texts[i].lower() == target_tokens[0].lower():
                match_indices = [i]
                match_text_parts = [token_texts[i]]

                for j in range(1, len(target_tokens)):
                    if i + j < len(token_texts) and token_texts[i+j].lower() == target_tokens[j].lower():
                        match_indices.append(i + j)
                        match_text_parts.append(token_texts[i+j])
                    else:
                        break

                score = difflib.SequenceMatcher(None, " ".join(match_text_parts), value).ratio()

                if score > 0.8 and len(match_indices) >= len(target_tokens) * 0.7:
                    if score > best_score:
                        best_score, best_match_indices = score, match_indices

        if best_match_indices:
            for k, idx in enumerate(best_match_indices):
                if labels[idx] == "O":
                    labels[idx] = f"B-{entity_type}" if k == 0 else f"I-{entity_type}"

    # --- Patient ---
    for key, row in corrected.get("patient", {}).items():
        if isinstance(row, dict):
            if "Value" in row and row["Value"]:
                assign_labels_bbox_aware(f"PATIENT_{key.upper().replace(' ', '_')}", row["Value"])
            for col, val in row.items():
                if col not in ["Value", "Conf (%)"] and val:
                    assign_labels_bbox_aware(f"PATIENT_{key.upper().replace(' ', '_')}_{col.upper().replace(' ', '_')}", val)
        elif isinstance(row, str):
            assign_labels_bbox_aware(f"PATIENT_{key.upper().replace(' ', '_')}", row)

    # --- Tests (with subtests) ---
    for test in corrected.get("tests", []):
        for col, val in test.items():
            if col != "subtests" and val:
                assign_labels_bbox_aware(f"TEST_{col.upper().replace(' ', '_')}", val)

        if "subtests" in test and isinstance(test["subtests"], list):
            for sub in test["subtests"]:
                for col, val in sub.items():
                    if val:
                        assign_labels_bbox_aware(f"TEST_SUB_{col.upper().replace(' ', '_')}", val)

    # --- Additional Sections ---
    for section_name, records in corrected.get("additional_sections", {}).items():
        section_prefix = section_name.upper().replace(" ", "_")
        for record in records:
            if isinstance(record, dict):
                for col, val in record.items():
                    if isinstance(val, str) and val:
                        assign_labels_bbox_aware(f"{section_prefix}_{col.upper().replace(' ', '_')}", val)
                    elif isinstance(val, list):
                        for subval in val:
                            if isinstance(subval, str) and subval:
                                assign_labels_bbox_aware(f"{section_prefix}_{col.upper().replace(' ', '_')}", subval)
                            elif isinstance(subval, dict):
                                for subcol, subval2 in subval.items():
                                    if subval2:
                                        assign_labels_bbox_aware(f"{section_prefix}_{col.upper().replace(' ', '_')}_{subcol.upper().replace(' ', '_')}", subval2)

    return labels

def prepare_dataset(corrections_dir="data/corrections", normalize_bboxes=False):
    """Load corrections and generate dataset with BIO labels."""
    data = []
    all_possible_labels = set()

    # Pass 1: Collect all unique labels
    for file in os.listdir(corrections_dir):
        if file.endswith(".json"):
            try:
                with open(os.path.join(corrections_dir, file), "r") as f:
                    corr = json.load(f)
                temp_labels = generate_bio_labels(corr["original_tokens"], corr["corrected_structured"])
                all_possible_labels.update(temp_labels)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

    # Create label mappings
    all_labels_sorted = sorted(list(all_possible_labels))
    label2id = {label: i for i, label in enumerate(all_labels_sorted)}
    id2label = {i: label for label, i in label2id.items()}

    # Pass 2: Generate dataset
    for file in os.listdir(corrections_dir):
        if file.endswith(".json"):
            try:
                with open(os.path.join(corrections_dir, file), "r") as f:
                    corr = json.load(f)
                tokens = corr["original_tokens"]
                labels = generate_bio_labels(tokens, corr["corrected_structured"])

                item = {
                    "tokens": [t["text"] for t in tokens],
                    "ner_tags": [label2id.get(tag, label2id["O"]) for tag in labels],
                }
                if normalize_bboxes:
                    max_x = max(t["left"] + t["width"] for t in tokens) if tokens else 1
                    max_y = max(t["top"] + t["height"] for t in tokens) if tokens else 1
                    item["bboxes"] = [[
                        int(1000 * t["left"] / max_x),
                        int(1000 * t["top"] / max_y),
                        int(1000 * (t["left"] + t["width"]) / max_x),
                        int(1000 * (t["top"] + t["height"]) / max_y)
                    ] for t in tokens]
                data.append(item)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

    if not data:
        raise ValueError("No valid correction files found in data/corrections")
    return Dataset.from_list(data), label2id, id2label

# --- Training ---

def train_model(dataset=None, label2id=None, id2label=None):
    """Train BERT model for token classification."""
    if dataset is None:
        dataset, label2id, id2label = prepare_dataset()
    if label2id is None or id2label is None:
        print("Error: label2id or id2label not properly generated.")
        return None, None, None
    if len(dataset) < 2:
        print("Need at least 2 corrections for training to perform a train/eval split.")
        return None, None, None

    train_data, eval_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def tokenize_and_align(examples):
        tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != prev_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                prev_word_idx = word_idx
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    # Preserve bboxes if present
    columns_to_remove = [col for col in train_ds.column_names if col not in ['bboxes']]
    train_ds = train_ds.map(tokenize_and_align, batched=True, remove_columns=columns_to_remove)
    eval_ds = eval_ds.map(tokenize_and_align, batched=True, remove_columns=columns_to_remove)

    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir="models/bert_lab_extractor",
        eval_strategy="epoch",  # Correct parameter name
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",  # Align with eval_strategy
    )

    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = torch.argmax(torch.from_numpy(predictions), dim=2)
        true_preds = [[id2label[p.item()] for (p, l) in zip(pred, label) if l != -100]
                      for pred, label in zip(predictions, labels)]
        true_labels = [[id2label[l.item()] for (p, l) in zip(pred, label) if l != -100]
                       for pred, label in zip(predictions, labels)]
        results = metric.compute(predictions=true_preds, references=true_labels, zero_division=0)
        return {
            "f1": results["overall_f1"],
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "accuracy": results["overall_accuracy"]
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model("models/bert_lab_extractor/final")

    return model, tokenizer, id2label

def retrain_from_new_corrections():
    """Retrain model with all corrections in data/corrections."""
    print("Retraining model with all available corrections...")
    dataset, label2id, id2label = prepare_dataset()
    train_model(dataset, label2id, id2label)

# --- Inference to JSON ---

def infer_and_structure_to_json(tokens_list_of_text, model_path="models/bert_lab_extractor/final"):
    """Infer labels and structure into JSON with confidence."""
    try:
        model = BertForTokenClassification.from_pretrained(model_path)
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        id2label = model.config.id2label
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return {}

    inputs = tokenizer(tokens_list_of_text, return_tensors="pt", is_split_into_words=True, truncation=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    confidences = torch.softmax(logits, dim=2).max(dim=2).values

    word_ids = inputs.word_ids(batch_index=0)
    pred_labels_aligned = []
    pred_confs_aligned = []

    current_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != current_word_idx:
            pred_labels_aligned.append(id2label[predictions[0, i].item()])
            pred_confs_aligned.append(confidences[0, i].item())
            current_word_idx = word_idx

    structured = {"patient": {}, "tests": [], "additional_sections": {}}
    current_entity_tokens = []
    current_type = None
    current_confs = []

    for token_text, label, conf in zip(tokens_list_of_text, pred_labels_aligned, pred_confs_aligned):
        if label.startswith("B-"):
            if current_entity_tokens:
                avg_conf = sum(current_confs) / len(current_confs) if current_confs else 0.0
                add_to_structured(structured, current_type, " ".join(current_entity_tokens), avg_conf)
            current_type = label[2:]
            current_entity_tokens = [token_text]
            current_confs = [conf]
        elif label.startswith("I-") and current_type == label[2:]:
            current_entity_tokens.append(token_text)
            current_confs.append(conf)
        else:
            if current_entity_tokens:
                avg_conf = sum(current_confs) / len(current_confs) if current_confs else 0.0
                add_to_structured(structured, current_type, " ".join(current_entity_tokens), avg_conf)
            current_entity_tokens = []
            current_type = None
            current_confs = []

    if current_entity_tokens:
        avg_conf = sum(current_confs) / len(current_confs) if current_confs else 0.0
        add_to_structured(structured, current_type, " ".join(current_entity_tokens), avg_conf)

    return structured

def add_to_structured(structured, entity_type, value, conf):
    """Dynamically add to JSON structure based on type."""
    if entity_type is None:
        return
    parts = entity_type.split("_")

    if entity_type.startswith("PATIENT_"):
        base_category = "PATIENT"
        sub_type_parts = parts[1:]
    elif entity_type.startswith("TEST_"):
        base_category = "TEST"
        sub_type_parts = parts[1:]
    else:
        base_category = "ADDITIONAL"
        sub_type_parts = parts

    if base_category == "PATIENT":
        key_name = "_".join(sub_type_parts).lower()
        structured["patient"][key_name] = {"Value": value, "Confidence": conf}
    elif base_category == "TEST":
        if sub_type_parts and sub_type_parts[0] == "SUB":
            sub_column = "_".join(sub_type_parts[1:]).lower()
            if not structured["tests"]:
                structured["tests"].append({})
            if "subtests" not in structured["tests"][-1]:
                structured["tests"][-1]["subtests"] = []
            if not structured["tests"][-1]["subtests"] or sub_column in structured["tests"][-1]["subtests"][-1]:
                structured["tests"][-1]["subtests"].append({})
            structured["tests"][-1]["subtests"][-1][sub_column] = value
        else:
            column_name = "_".join(sub_type_parts).lower()
            if not structured["tests"] or (column_name in structured["tests"][-1] and column_name != "name"):
                structured["tests"].append({})
            if structured["tests"]:
                structured["tests"][-1][column_name] = value
    else:
        if not parts:
            return
        section_name = parts[0].lower()
        column_name = "_".join(parts[1:]).lower() if len(parts) > 1 else "value"
        if section_name not in structured["additional_sections"]:
            structured["additional_sections"][section_name] = []
        if not structured["additional_sections"][section_name] or column_name in structured["additional_sections"][section_name][-1]:
            structured["additional_sections"][section_name].append({})
        if structured["additional_sections"][section_name]:
            structured["additional_sections"][section_name][-1][column_name] = value

if __name__ == "__main__":
    print("Starting BERT Lab Extractor training/inference script...")
    try:
        model, tokenizer, id2label = train_model()
        print("Model training complete and saved.")
    except Exception as e:
        print(f"Error during training: {e}")
        model, tokenizer, id2label = None, None, None

    if model and tokenizer and id2label:
        test_ocr_token_file = "data/ocr_tokens/doc_1_page_01.json"
        if os.path.exists(test_ocr_token_file):
            with open(test_ocr_token_file, "r") as f:
                test_tokens_data = json.load(f)
                test_tokens_list_of_text = [t["text"] for t in test_tokens_data]
            print(f"\nPerforming inference on {test_ocr_token_file}...")
            json_output = infer_and_structure_to_json(test_tokens_list_of_text)
            print("Sample JSON Output:")
            print(json.dumps(json_output, indent=2))
        else:
            print(f"Test OCR token file not found: {test_ocr_token_file}. Cannot run inference example.")
    else:
        print("Model or tokenizer not available for inference.")