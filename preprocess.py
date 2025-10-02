import os
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path


def pdf_to_images(pdf_path, dpi=300, out_dir='data/preprocessed'):
    pages = convert_from_path(pdf_path, dpi=dpi)
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    paths = []

    for i, page in enumerate(pages):
        filename = f"{base}_page_{i+1:02d}.png"
        path = os.path.join(out_dir, filename)
        page.save(path, "PNG")
        paths.append(path)

    return paths





def auto_orient(img):
    osd = pytesseract.image_to_osd(img, config="--psm 0")
    rotate_line = [l for l in osd.split("\n") if "Rotate:" in l][0]
    angle = int(rotate_line.split(":")[-1].strip())
    if angle != 0:
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return img


def deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    angle = 0.0
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            ang = (theta * 180 / np.pi) - 90
            if -45 < ang < 45:
                angles.append(ang)
        if angles:
            angle = np.median(angles)
    if abs(angle) > 0.5:
        (h, w) = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return img


def denoise_and_binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    _, th = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def preprocess_document(image_path, out_path):
    img = cv2.imread(image_path)
    img = auto_orient(img)
    img = deskew(img)
    img = denoise_and_binarize(img)
    cv2.imwrite(out_path, img)
    return out_path


def preprocess_pdf(pdf_path, out_dir='data/preprocessed'):
    raw_image_paths = pdf_to_images(pdf_path, out_dir=out_dir)
    processed_paths = []

    for path in raw_image_paths:
        filename = os.path.splitext(os.path.basename(path))[0]  
        out_path = os.path.join(out_dir, f"{filename}_proc.png")
        preprocess_document(path, out_path)
        processed_paths.append(out_path)

    return processed_paths
