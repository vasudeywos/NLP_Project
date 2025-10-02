# rules.py
import re
import json
import numpy as np # Used implicitly for coordinate calculations in a more complex setup, but kept here for potential future use

# --- Utility Functions ---

def group_tokens_into_lines(tokens, y_tol=5):
    """
    Group tokens into lines based on y-coordinate proximity.
    y_tol is reduced to 5 for tight vertical spacing common in lab reports.
    """
    # Sort primarily by 'top' (y-coordinate) and secondarily by 'left' (x-coordinate)
    tokens_sorted = sorted(tokens, key=lambda x: (x['top'], x['left']))
    lines = []
    current_line = []
    last_y = None
    for t in tokens_sorted:
        if last_y is None or abs(t['top'] - last_y) <= y_tol:
            current_line.append(t)
            last_y = t['top']
        else:
            lines.append(current_line)
            current_line = [t]
            last_y = t['top']
    if current_line:
        lines.append(current_line)
    return lines


def get_next_key_start_left(line, start_index, key_start_left, max_left_coord=10000):
    """
    Helper function to find the left coordinate of the next potential key on the line.
    This sets the boundary for the current value, making value extraction non-greedy.
    """
    
    # Heuristic column starts for common lab report layouts (approximate)
    # These are used as a hint for when a new key might start.
    HINT_COLUMNS = [1200, 1750] 
    
    for j in range(start_index, len(line)):
        tok = line[j]
        
        # Condition 1: Token is a major capitalized word and significantly distant from the current key.
        is_distant = tok["left"] > key_start_left + 300
        is_title = tok["text"][0].isupper() and len(tok["text"]) > 1
        
        # Condition 2: Token's left coordinate aligns with a known column start hint.
        is_aligned = any(abs(tok["left"] - col) < 50 for col in HINT_COLUMNS)
        
        if (is_distant and is_title) or (is_aligned and is_title):
            # Check if this potential key is followed by a colon or separator soon after.
            for k in range(j, min(j + 5, len(line))): # Search up to 5 tokens ahead
                if re.match(r'[:\u2014_]', line[k]["text"]):
                    # Found a new key starting at line[j]["left"]
                    return line[j]["left"]
    
    # If no next key is found, set the boundary very far to the right (end of line)
    return max_left_coord


# --- Core Extraction Logic ---

def extract_key_values(lines):
    """
    Extract key-value pairs using ':' or other separators.
    Value collection stops dynamically at the start of the next key column.
    This function is now dynamic, capturing any field it finds.
    """
    fields = {}
    for line in lines:
        i = 0
        while i < len(line):
            tok = line[i]
            key_text = ""
            colon_index = -1
            
            # --- 1. Dynamic Key Identification and Colon Index ---
            
            # Case A: Single Token Key (e.g., "Name:" or "Gender:")
            if tok["text"].strip().endswith(":") or re.match(r'[A-Za-z]+[\u2019]?:', tok["text"]):
                key_text = tok["text"].strip(":\u2019").strip()
                colon_index = i
            
            # Case B: Two Token Key (e.g., "Lab ID" then ":")
            elif i + 1 < len(line) and re.match(r'[:\u2014_-]', line[i+1]["text"]):
                key_text = tok["text"].strip()
                colon_index = i + 1
            
            # Case C: Multi-word Key (e.g., 'Reg Date and Time' with separator later)
            elif tok["text"][0].isupper() and len(tok["text"]) > 1:
                j = i + 1
                potential_key_words = [tok["text"]]
                
                # Look ahead for a separator (colon or similar)
                while j < len(line) and j < i + 6 and not re.match(r'[:\u2014_-]', line[j]["text"]):
                    # Only include words that aren't too far away horizontally
                    if line[j]["left"] - line[j-1]["left"] < 250: 
                        potential_key_words.append(line[j]["text"])
                    else:
                        break
                    j += 1
                
                if j < len(line) and re.match(r'[:\u2014_-]', line[j]["text"]):
                    colon_index = j
                    key_text = " ".join(potential_key_words)
                    i = j - 1 # Adjust i to ensure the loop continues correctly
            
            
            if key_text and colon_index != -1:
                # --- 2. Value Collection with Dynamic Stop Condition ---
                
                start_index = colon_index + 1
                value_tokens = []
                
                # Safely get the left coordinate of the key's first token (at index i)
                # This fixes the IndexError from the previous attempt.
                key_start_left = line[i]["left"]
                
                # Determine the left boundary for the next key field on this line.
                next_key_left = get_next_key_start_left(line, start_index, key_start_left)
                
                k = start_index
                while k < len(line):
                    current_tok = line[k]
                    
                    # Stop Condition: If the token's left coordinate is close to the next key's start
                    if current_tok["left"] >= next_key_left - 50:
                        break
                    
                    value_tokens.append(current_tok["text"])
                    k += 1
                
                value = " ".join(value_tokens).strip()
                
                # Normalize key and cleanup value
                dynamic_key = re.sub(r'[\u2014_-]', '', key_text).lower().replace(' ', '_').strip('._:-')
                value = re.sub(r'^[:\s\u2014_-]+', '', value).strip()
                
                if dynamic_key and value:
                    fields[dynamic_key] = value
                
                i = k - 1 # Continue search from where value extraction stopped or colon was found

            i += 1
            
    return fields


def extract_tests(lines):
    """
    Extract test rows with value + unit from the test table.
    """
    tests = []
    # Expanded common units
    unit_regex = r'(mg/dL|mEq/L|g/dL|IU/L|U/L|mmol/L|ng/mL|pg/mL|%|fl|kU/L|pg/ml|ug/dL|uIU/ml)' 
    
    for line in lines:
        texts = [tok["text"] for tok in line]
        ln = " ".join(texts)
        
        # Exclude header lines
        if ln.startswith("TEST") or ln.startswith("RESULTS") or ln.startswith("Bio. Ref."): 
            continue
        
        # Pattern: [Test Name] [Value (digits/decimal)] [Unit]
        # Use a non-greedy approach for the name part up to the value/unit
        m = re.search(r'([A-Za-z\s\-\.\/\(\),]+?)\s+([\d\.]+)\s*'+unit_regex, ln)
        
        if m:
            tests.append({
                "name": m.group(1).strip().replace('-', ' ').strip(),
                "value": m.group(2),
                "unit": m.group(3)
            })
    return tests


def run_rule_extraction(tokens):
    """
    Run full rule-based extraction pipeline.
    """
    lines = group_tokens_into_lines(tokens, y_tol=5) 
    
    fields = extract_key_values(lines)
    tests = extract_tests(lines)

    # Post-processing: Apply simple standardizations to the dynamically extracted keys
    cleaned_fields = {}
    
    # Dynamic Key Mapping/Cleaning (applied after extraction)
    for k, v in fields.items():
        # Clean up common OCR errors in keys
        if 'name' in k: k = 'name'
        elif 'gender' in k: k = 'gender'
        elif 'labid' in k or 'lab_no' in k: k = 'lab_id'
        elif 'age' in k: k = 'age'
        elif 'collected' in k and 'at' in k: k = 'sample_collected_at'
        elif 'collected' in k: k = 'collected_on'
        elif 'reported' in k: k = 'reported_on'
        elif 'status' in k: k = 'att_status'
        elif 'ref_by' in k: k = 'referred_by'
        elif 'processed' in k: k = 'processed_at'
        
        # Clean up values (e.g., removing trailing colons or excess space)
        cleaned_fields[k] = v.strip(':- ')

    return {
        "patient": cleaned_fields,
        "tests": tests
    }