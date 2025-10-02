import pytesseract
from pytesseract import Output
import json
import os

def ocr_image_to_tokens(image_path, out_json_path=None):
    data = pytesseract.image_to_data(image_path, output_type=Output.DICT)

    tokens = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        if text == "":
            continue
        tokens.append({
            "text": text,
            "left": int(data['left'][i]),
            "top": int(data['top'][i]),
            "width": int(data['width'][i]),
            "height": int(data['height'][i]),
            "conf": float(data['conf'][i])
        })

    if out_json_path:
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(tokens, f, indent=2)

    return tokens
