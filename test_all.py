import os

def prepare_dataset(corrections_dir="data/corrections", image_dir="data/preprocessed"):
    """
    Collect all (image, correction JSON) pairs.
    """
    data = []
    print(f"Loading data from: {corrections_dir} and {image_dir}")
    
    for file_name in os.listdir(corrections_dir):
        if not file_name.endswith(".json"):
            continue
        
        # doc_5_page_01_correction.json -> doc_5_page_01_proc.png
        base_name = os.path.splitext(file_name)[0].replace("_correction", "_proc")
        json_path = os.path.join(corrections_dir, file_name)
        image_path = os.path.join(image_dir, base_name + ".png")

        if os.path.exists(image_path):
            data.append((image_path, json_path))
        else:
            print(f"⚠️ Missing image for {file_name}: expected {image_path}")
    
    return data


def process_all(model_path, corrections_dir="data/corrections", image_dir="data/preprocessed", output_dir="data/outputs"):
    """
    Run inference for all paired inputs and save outputs as JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = prepare_dataset(corrections_dir, image_dir)

    for image_path, json_path in dataset:
        print(f"\n🔍 Processing {image_path} with {json_path}")
        result = run_inference(model_path, image_path, json_path)

        out_name = os.path.splitext(os.path.basename(json_path))[0].replace("_correction", "_output.json")
        out_path = os.path.join(output_dir, out_name)

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"✅ Saved output to {out_path}")


if __name__ == "__main__":
    process_all(MODEL_PATH)
