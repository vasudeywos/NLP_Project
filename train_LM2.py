import json
import os
import difflib
import bisect
import torch
import numpy as np
from PIL import Image
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split

def generate_bio_labels(tokens, corrected):
    token_texts = [t["text"] for t in tokens]
    full_text = " ".join(token_texts)
    labels = ["O"] * len(tokens)
    cum_chars = [0]
    for t in token_texts:
        cum_chars.append(cum_chars[-1] + len(t) + 1)
    def assign_labels(entity_type, value):
        if not isinstance(value, str): value = str(value)
        if not value.strip(): return
        matcher = difflib.SequenceMatcher(None, full_text, value, autojunk=False)
        match = matcher.find_longest_match(0, len(full_text), 0, len(value))
        if match.size < len(value) * 0.8: return
        match_start, match_end = match.a, match.a + match.size
        token_start = bisect.bisect_right(cum_chars, match_start) - 1
        token_end = bisect.bisect_left(cum_chars, match_end) - 1
        if token_start <= token_end:
            labels[token_start] = f"B-{entity_type}"
            for i in range(token_start + 1, token_end + 1):
                labels[i] = f"I-{entity_type}"
    for key, row in corrected.get("patient", {}).items():
        assign_labels(f"PATIENT_{key.upper()}", row.get("Value", ""))
    for test in corrected.get("tests", []):
        if isinstance(test, dict):
            for col, val in test.items():
                assign_labels(f"TESTS_{col.upper()}", val)
    for section_name, records in corrected.get("additional_sections", {}).items():
        section_type = section_name.upper()
        for record in records:
            if isinstance(record, dict):
                for col, val in record.items():
                    assign_labels(f"{section_type}_{col.upper()}", val)
            elif isinstance(record, str):
                assign_labels(section_type, record)
    return labels

def prepare_dataset(corrections_dir="data/corrections", image_dir="data/preprocessed"):
    data = []
    print(f"Loading data from: {corrections_dir} and {image_dir}")
    for file_name in os.listdir(corrections_dir):
        if not file_name.endswith(".json"): continue
        base_name = os.path.splitext(file_name)[0].replace('_correction', '_proc')
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        if not image_path:
            print(f"Warning: No image found for {file_name}")
            continue
        json_path = os.path.join(corrections_dir, file_name)
        with open(json_path, "r") as f, Image.open(image_path).convert("RGB") as image:
            corr = json.load(f)
            tokens = corr.get("original_tokens", [])
            if not tokens: continue
            page_width, page_height = image.size
            def clamp(value, min_val=0, max_val=1000):
                return max(min_val, min(value, max_val))
            bboxes = []
            for t in tokens:
                x1 = (t['left'] / page_width) * 1000
                y1 = (t['top'] / page_height) * 1000
                x2 = ((t['left'] + t['width']) / page_width) * 1000
                y2 = ((t['top'] + t['height']) / page_height) * 1000
                box = [
                    clamp(int(x1)),
                    clamp(int(y1)),
                    clamp(int(x2)),
                    clamp(int(y2))
                ]
                bboxes.append(box)
            labels = generate_bio_labels(tokens, corr["corrected_structured"])
            item = {"image_path": image_path, "tokens": [t["text"] for t in tokens], "ner_tags": labels, "bboxes": bboxes}
            data.append(item)
    all_labels = sorted(list(set(tag for item in data for tag in item["ner_tags"])))
    label2id = {label: i for i, label in enumerate(all_labels)}
    dataset_data = {'image_path': [], 'tokens': [], 'ner_tags': [], 'bboxes': []}
    for item in data:
        dataset_data['image_path'].append(item['image_path'])
        dataset_data['tokens'].append(item['tokens'])
        dataset_data['ner_tags'].append([label2id[tag] for tag in item['ner_tags']])
        dataset_data['bboxes'].append(item['bboxes'])
    return Dataset.from_dict(dataset_data), label2id, {v: k for k, v in label2id.items()}

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

def tokenize_and_align(examples):
    images = [Image.open(path).convert("RGB") for path in examples['image']]
    tokens = examples['tokens']
    boxes = examples['bboxes']
    word_labels = examples['ner_tags']
    encoded_inputs = processor(images, tokens, boxes=boxes, word_labels=word_labels,
                               padding="max_length", truncation=True)
    return encoded_inputs

def train_model():
    dataset, label2id, id2label = prepare_dataset()
    dataset = dataset.rename_column("image_path", "image")
    train_ds, eval_ds = dataset.train_test_split(test_size=0.2, seed=42).values()
    train_ds.set_transform(tokenize_and_align)
    eval_ds.set_transform(tokenize_and_align)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    args = TrainingArguments(
        output_dir="models/layoutlmv3_lab_extractor",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=40,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        remove_unused_columns=False,
    )
    metric = evaluate.load("seqeval")
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [[id2label[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[id2label[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        results = metric.compute(predictions=true_predictions, references=true_labels, scheme="IOB2")
        return {"f1": results["overall_f1"], "precision": results["overall_precision"], "recall": results["overall_recall"]}
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, compute_metrics=compute_metrics)
    print("Starting LayoutLMv3 training...")
    trainer.train()
    trainer.save_model("models/layoutlmv3_lab_extractor/final")
    print("Training complete and LayoutLMv3 model saved.")

if __name__ == "__main__":
    train_model()
