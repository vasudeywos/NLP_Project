# NLP Project

This project implements an end-to-end pipeline to extract structured information from PDF lab reports. The system converts unstructured documents into a clean, queryable JSON format by leveraging a combination of image processing, rule-based methods, and a sophisticated multimodal deep learning model. A Human-in-the-Loop (HITL) web interface is included for data validation and progressive model improvement.

## 📋 Table of Contents

  - [Project Overview](https://www.google.com/search?q=%23-project-overview)
  - [System Architecture](https://www.google.com/search?q=%23-system-architecture)
  - [Implementation Details](https://www.google.com/search?q=%23-implementation-details)
      - [1. Preprocessing and OCR](https://www.google.com/search?q=%231-preprocessing-and-ocr)
      - [2. Rule-Based Extraction (Baseline)](https://www.google.com/search?q=%232-rule-based-extraction-baseline)
      - [3. Human-in-the-Loop (HITL) for Data Labeling](https://www.google.com/search?q=%233-human-in-the-loop-hitl-for-data-labeling)
      - [4. Supervised Learning with Transformer Models](https://www.google.com/search?q=%234-supervised-learning-with-transformer-models)
  - [How to Run the Project](https://www.google.com/search?q=%23-how-to-run-the-project)
  - [Model Performance and Future Work](https://www.google.com/search?q=%23-model-performance-and-future-work)

-----

## 🔭 Project Overview

The primary goal of this project is to automate the extraction of key-value pairs and tabular data from scanned lab reports. These documents, often in PDF format, contain critical information in a semi-structured layout that is challenging for traditional parsers. This project builds a supervised learning pipeline to convert the OCR-extracted text into a structured JSON object.

**Input**: A PDF lab report.
**Output**: A structured JSON file containing patient details, test results, and other relevant sections.

```json
{
  "patient": {
    "name": {"Value": "Mr. John Doe"},
    "age": {"Value": "45 Years"}
  },
  "tests": [
    {
      "name": "Hemoglobin",
      "value": "14.5",
      "unit": "g/dL"
    }
  ]
}
```

-----

## 🏛️ System Architecture

The project follows a multi-stage pipeline, ensuring robustness and continuous improvement.

1.  **File Input & Preprocessing**: PDFs are converted into clean, machine-readable images. (`preprocess.py`)
2.  **OCR & Tokenization**: Text and its coordinates are extracted from the images. (`ocr.py`)
3.  **Rule-Based Baseline**: A regex and heuristic-based system provides initial, structured output. (`rules.py`)
4.  **Human-in-the-Loop (HITL)**: A Streamlit web app allows users to correct the baseline output, creating a high-quality labeled dataset. (`hitl.py`)
5.  **Supervised Model Training**: A Transformer-based model is trained on the human-corrected data for Named Entity Recognition (NER). (`train_layoutlmv3.py`)
6.  **Inference**: The trained model is used to extract information from new documents.

-----

## ⚙️ Implementation Details

### 1\. Preprocessing and OCR

This initial stage is handled by `preprocess.py` and `ocr.py`. It ensures that the text extraction process is as accurate as possible.

  * **PDF to Image Conversion**: The `pdf2image` library converts each page of a source PDF into a high-resolution PNG image.
  * **Image Cleaning**: **OpenCV** is used to perform critical image enhancement tasks:
      * **Auto-Orientation**: Detects and corrects the orientation of the scanned page using `pytesseract.image_to_osd`.
      * **Deskewing**: Straightens pages that were scanned at a slight angle.
      * **Denoising & Binarization**: A median blur removes noise, and Otsu's thresholding converts the image to a clean black-and-white format, which is optimal for OCR.
  * **Optical Character Recognition (OCR)**: The cleaned image is processed by **Tesseract OCR** (`pytesseract.image_to_data`). This step extracts not just the text but a list of "tokens," where each token contains the word, its confidence score, and its precise bounding box coordinates (`left`, `top`, `width`, `height`) on the page. This positional information is crucial for later stages.

### 2\. Rule-Based Extraction (Baseline)

Before deploying a complex model, we establish a strong baseline using `rules.py`. This module uses regular expressions and positional heuristics to perform an initial extraction. This provides immediate value and generates data for the HITL correction phase.

  * **Line Grouping**: Tokens are grouped into lines based on their vertical (`top`) coordinates.
  * **Dynamic Key-Value Extraction**: The script dynamically identifies key-value pairs by looking for separators like colons (`:`). It intelligently determines the value boundary by detecting the start of the next potential key on the same line, preventing greedy matching.
  * **Test Table Extraction**: A separate function uses regex to identify common patterns for test results, such as `(Test Name) (Value) (Unit)`.

### 3\. Human-in-the-Loop (HITL) for Data Labeling

This is the most critical step for creating a supervised learning dataset. The `hitl.py` script launches a **Streamlit** web application that allows a human to review and correct the output from the rule-based system.

  * **Interactive UI**: The app displays the document image for visual reference alongside editable tables (`st.data_editor`) for patient details, tests, and any additional sections.
  * **Dynamic Schema**: The user can dynamically add or remove rows and columns, allowing for flexible labeling of varied document structures.
  * **Saving Corrections**: When the user confirms their changes, the app saves a `correction.json` file. This file contains both the original OCR tokens (with coordinates) and the user-verified structured JSON, creating a perfectly labeled example for model training.

### 4\. Supervised Learning with Transformer Models

The corrected data is used to train a powerful Transformer model to perform token classification (NER). The goal is to assign a BIO (Beginning, Inside, Outside) tag to each token, such as `B-PATIENT_NAME` or `I-TESTS_VALUE`.

#### Model Evolution

  * **Attempt 1: BERT (`bert.py`)**: An initial attempt was made using `bert-base-uncased`. BERT is a powerful text-based model, but it treats the document as a one-dimensional sequence of words. This approach failed to achieve high accuracy because it is **unaware of the 2D document layout**. It struggled to differentiate entities based on their position, such as which column a value belongs to.

  * **Attempt 2: LayoutLMv3 (`train_layoutlmv3.py`)**: To overcome BERT's limitations, the project was upgraded to **LayoutLMv3**. This is a multimodal model that processes three types of information simultaneously:

    1.  **Text**: The meaning of the words.
    2.  **Layout**: The 2D bounding box coordinates of each word.
    3.  **Visual**: The actual pixels of the document image.

    By understanding text, position, and visual cues (like lines and tables), LayoutLMv3 can learn the spatial structure of the lab report, leading to significantly better performance. The model is trained to assign the correct entity tag to each token from the human-labeled data.

-----

## 🚀 How to Run the Project

1.  **Setup**:
      * Clone the repository.
      * Install dependencies: `pip install -r requirements.txt`. This includes `transformers`, `torch`, `streamlit`, `opencv-python`, `pdf2image`, `pytesseract`, etc.
      * Install Tesseract OCR on your system.
2.  **Preprocessing & OCR**:
      * Place your PDFs in a `data/raw_pdfs` folder.
      * Run the preprocessing and OCR scripts to generate cleaned images and token JSON files.
3.  **Rule-Based Extraction**:
      * Run `rules.py` on the OCR token files to generate baseline JSON outputs.
4.  **Data Labeling**:
      * Launch the HITL app: `streamlit run hitl.py`.
      * Open the web interface, load documents, and correct the baseline extractions to generate `correction.json` files.
5.  **Model Training**:
      * Run the training script: `python train_layoutlmv3.py`.
      * The trained model will be saved to the `models/` directory.
6.  **Inference**:
      * Use the provided `test_layoutlmv3.py` script to run the trained model on new documents.

-----

## 📊 Model Performance and Future Work

The **LayoutLMv3 model performs significantly better** than the text-only BERT model, successfully identifying and structuring most patient and test information. It demonstrates a strong ability to use layout cues to disambiguate fields.

However, the model's performance is still constrained by the limited size of the training dataset. Current work is focused on:

  * **Data Augmentation**: Programmatically creating more training examples by swapping entities (names, dates) and simulating OCR errors to improve model robustness.
  * **Hyperparameter Tuning**: Fine-tuning parameters like the learning rate and number of epochs to prevent overfitting on the small dataset.
  * **Confidence Scoring**: Implementing a confidence score for the model's predictions to flag low-certainty extractions for human review.
