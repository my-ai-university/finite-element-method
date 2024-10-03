
"""
This module provides tools for processing images of textbook pages, focusing on extracting 
text and mathematical content. It includes functions for OCR, LaTeX extraction, and image segmentation
"""

"""
pip install Pillow  # For image processing (PIL)
pip install pix2tex  # For LaTeX OCR (pix2tex)
pip install pytesseract  # For OCR (Tesseract)
"""

import os
import json
import base64
import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from pathlib import Path
from pix2tex.cli import LatexOCR
import pytesseract


# Load environment variables from .env file
load_dotenv()

# Load OpenAI API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the LaTeX OCR model for process_image_latex_ocr function (see below)
latex_ocr_model = LatexOCR()


def encode_image(image_path):
    """
    Encode an image in base64 format.
    """
    image_path = Path(image_path)
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def process_image_openai(image_path):
    """
    Process an image using OpenAI's API for text and equation extraction.

    This gives the best result, but it is slow and cannot be run in a loop (not sure why)
    """

    base64_image = encode_image(image_path)

    system_message = {
        "role": "system",
        "content": """You are an AI assistant specialized in processing images of textbook pages. 
        Your task is to accurately extract and format the content, paying special attention to mathematical equations."""
    }

    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            },
            {
                "type": "text",
                "text": """
                Analyze the image and extract the following information:

                1. **Header**: Extract the one line of text at the top of the page, including all numbers.
                2. **Content**: Extract the main content of the page, starting from the first word after the header and ending with the very last word on the page, preserving the order.

                **IMPORTANT**:
                - Use LaTeX for all mathematical symbols and equations.
                - Ensure complete accuracy in representing equations, including superscripts, subscripts, hats, bars, Greek symbols, boldness, and special characters.
                - Preserve the original formatting as much as possible.
                - Include question numbers.
                - Use dollar signs for equations: inline `$a=b$` and display `$$f(x) = x^2$$` for others.
                """
            }
        ]
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_message, user_message],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "page_extraction_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer"},
                        "chapter": {"type": "integer"},
                        "header": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["page", "chapter", "header", "content"],
                    "additionalProperties": False
                }
            }
        },
        temperature=0.15,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    generated_content = response.choices[0].message.content

    try:
        return json.loads(generated_content)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print("Generated content was:")
        print(generated_content)
        return None


def process_image_latex_ocr(image_path):
    """
    Process an image using LaTeX OCR to extract mathematical content and equations.

    The image should be small containing mainly euqations.
    """

    img = Image.open(image_path)
    latex = latex_ocr_model(img)
    return latex


def process_image_tesseract(image_path):
    """
    Process an image using Tesseract OCR to extract text.

    Works better if the entire page given as the input image.
    """

    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text


def slice_image_into_sections(image_path, output_folder, threshold=128):
    """
    Slices a grayscale image into sections horizontally  
    and saves each slice as a separate image in the specified output folder.
    """

    # Load the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Threshold the image to binary (black and white)
    _, thresh_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)

    # Find horizontal projections of the text to detect empty spaces
    horizontal_projection = np.sum(thresh_image, axis=1)

    # Identify non-empty areas (above a certain threshold)
    non_empty_rows = np.where(horizontal_projection > 0)[0]

    # Ensure non-empty rows exist, otherwise return zero slices
    if len(non_empty_rows) == 0:
        print("No non-empty rows detected.")
        return 0

    # Identify start and end of each slice by finding gaps between non-empty rows
    slices = []
    start = non_empty_rows[0]

    for i in range(1, len(non_empty_rows)):
        if non_empty_rows[i] != non_empty_rows[i - 1] + 1:
            end = non_empty_rows[i - 1]
            slices.append((start, end))
            start = non_empty_rows[i]

    # Add the last slice
    slices.append((start, non_empty_rows[-1]))


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save each slice as a separate image
    for i, (start, end) in enumerate(slices):
        slice_image = image[start:end, :]
        output_path = os.path.join(output_folder, f'slice_{i + 1}.png')
        cv2.imwrite(output_path, slice_image)

