import streamlit as st
import json
import os
import difflib
import pandas as pd
from PIL import Image

def load_data(ocr_path, baseline_path, image_path):
    if not os.path.exists(ocr_path):
        st.error(f"OCR file not found: {ocr_path}")
        return None, None, None
    if not os.path.exists(baseline_path):
        st.error(f"Baseline file not found: {baseline_path}")
        return None, None, None
    if not os.path.exists(image_path):
        st.warning(f"Image not found: {image_path} (visual reference skipped)")
        img = None
    else:
        img = Image.open(image_path)
    
    with open(ocr_path, "r") as f:
        tokens = json.load(f)
    with open(baseline_path, "r") as f:
        baseline = json.load(f)
    
    return tokens, baseline, img

def compute_confidence(tokens, value):
    full_text = " ".join([t["text"] for t in tokens])
    matcher = difflib.SequenceMatcher(None, full_text, value)
    match = matcher.find_longest_match(0, len(full_text), 0, len(value))
    if match.size < len(value) * 0.8:
        return 0.0
    token_texts = [t["text"] for t in tokens]
    cum_chars = [0]
    for t in token_texts:
        cum_chars.append(cum_chars[-1] + len(t) + 1)
    token_start = next((i for i, pos in enumerate(cum_chars) if pos > match.a), len(tokens)) - 1
    token_end = next((i for i, pos in enumerate(cum_chars) if pos > match.a + match.size), len(tokens)) - 1
    matched_tokens = tokens[token_start:token_end + 1]
    if not matched_tokens:
        return 0.0
    return sum(t["conf"] for t in matched_tokens) / len(matched_tokens)

st.title("Human-in-the-Loop Review for Lab Report Extraction")

pdf_base = st.selectbox("Select PDF", ["doc_1", "doc_2"])
page_num = st.number_input("Page Number", min_value=1, value=1)
ocr_path = f"data/ocr_tokens/{pdf_base}_page_{page_num:02d}.json"
baseline_path = f"data/baseline/{pdf_base}_page_{page_num:02d}_baseline.json"
image_path = f"data/preprocessed/{pdf_base}_page_{page_num:02d}_proc.png"

if st.button("Load Data"):
    tokens, baseline, img = load_data(ocr_path, baseline_path, image_path)
    if tokens and baseline:
        patient_data = [{"Key": k, "Value": v, "Conf (%)": compute_confidence(tokens, v)} for k, v in baseline.get("patient", {}).items()]
        st.session_state["patient_df"] = pd.DataFrame(patient_data)
        st.session_state["tests_df"] = pd.DataFrame(baseline.get("tests", []))
        st.session_state["tokens"] = tokens
        st.session_state["img"] = img
        if "additional_sections" not in st.session_state:
            st.session_state["additional_sections"] = {}
        st.success("Data loaded!")

if "img" in st.session_state and st.session_state["img"]:
    st.header("Page Image (Reference)")
    st.image(st.session_state["img"], use_column_width=True)

if "patient_df" in st.session_state:
    st.header("Patient Details (Editable Table - Add rows/columns as needed)")
    new_column_name = st.text_input("Add New Column for Patient Details", key="patient_new_col")
    if st.button("Add Column to Patient Details") and new_column_name:
        if new_column_name not in st.session_state["patient_df"].columns:
            st.session_state["patient_df"][new_column_name] = ""
            st.success(f"Added column '{new_column_name}' to Patient Details")
    edited_patient_df = st.data_editor(
        st.session_state["patient_df"],
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True
    )
    st.session_state["patient_df"] = edited_patient_df

if "tests_df" in st.session_state:
    st.header("Tests (Editable Table - Add rows/columns as needed)")
    new_column_name = st.text_input("Add New Column for Tests", key="tests_new_col")
    if st.button("Add Column to Tests") and new_column_name:
        if new_column_name not in st.session_state["tests_df"].columns:
            st.session_state["tests_df"][new_column_name] = ""
            st.success(f"Added column '{new_column_name}' to Tests")
    edited_tests_df = st.data_editor(
        st.session_state["tests_df"],
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True
    )
    st.session_state["tests_df"] = edited_tests_df

st.header("Additional Sections (e.g., Treatment Goals)")
new_section_name = st.text_input("New Section Name (e.g., treatment_goals)")
if st.button("Add New Section") and new_section_name:
    if new_section_name not in st.session_state["additional_sections"]:
        st.session_state["additional_sections"][new_section_name] = pd.DataFrame(columns=["Category", "Value"])
    st.success(f"Added section: {new_section_name}")

if "additional_sections" not in st.session_state:
    st.session_state["additional_sections"] = {}

for section_name, df in list(st.session_state["additional_sections"].items()):
    st.subheader(f"{section_name} (Editable Table - Add/Delete rows/columns as needed)")
    new_column_name = st.text_input(f"Add New Column for {section_name}", key=f"add_col_{section_name}")
    if st.button(f"Add Column to {section_name}") and new_column_name:
        if new_column_name not in df.columns:
            st.session_state["additional_sections"][section_name][new_column_name] = ""
            st.success(f"Added column '{new_column_name}' to {section_name}")
    if len(df.columns) > 0:
        col_to_delete = st.selectbox(
            f"Select Column to Delete from {section_name}",
            df.columns,
            key=f"del_col_{section_name}"
        )
        if st.button(f"Delete Column {col_to_delete} from {section_name}"):
            st.session_state["additional_sections"][section_name].drop(columns=[col_to_delete], inplace=True)
            st.success(f"Deleted column '{col_to_delete}' from {section_name}")
            st.rerun()
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key=f"editor_{section_name}"
    )
    st.session_state["additional_sections"][section_name] = edited_df
    if st.button(f"Remove {section_name}"):
        del st.session_state["additional_sections"][section_name]
        st.rerun()

if st.button("Confirm and Save"):
    if "tokens" in st.session_state and "patient_df" in st.session_state and "tests_df" in st.session_state:
        patient_records = st.session_state["patient_df"].to_dict(orient="records")
        patient = {row.get("Key", f"unknown_{i}"): {k: v for k, v in row.items() if k != "Key" and k != "Conf (%)"} for i, row in enumerate(patient_records)} if patient_records else {}
        tests = st.session_state["tests_df"].to_dict(orient="records")
        additional = {name: df.to_dict(orient="records") for name, df in st.session_state["additional_sections"].items()}
        corrected = {
            "patient": patient,
            "tests": tests,
            "additional_sections": additional
        }
        confirmed_dir = "data/confirmed"
        corrections_dir = "data/corrections"
        os.makedirs(confirmed_dir, exist_ok=True)
        os.makedirs(corrections_dir, exist_ok=True)
        confirmed_path = f"{confirmed_dir}/{pdf_base}_page_{page_num:02d}_confirmed.json"
        correction_path = f"{corrections_dir}/{pdf_base}_page_{page_num:02d}_correction.json"
        with open(confirmed_path, "w") as f:
            json.dump(corrected, f, indent=2)
        correction_data = {
            "original_tokens": st.session_state["tokens"],
            "corrected_structured": corrected
        }
        with open(correction_path, "w") as f:
            json.dump(correction_data, f, indent=2)
        st.success(f"Saved confirmed to {confirmed_path} and correction to {correction_path}")
