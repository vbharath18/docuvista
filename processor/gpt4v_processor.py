import os
import io
import tempfile
import logging
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import requests
import concurrent.futures
import fitz  # PyMuPDF
from .utils import ensure_data_dir, save_uploaded_file, cleanup_file

_log = logging.getLogger(__name__)

def process_uploaded_pdf_with_gpt4v(uploaded_file):
    """
    Process uploaded PDF using Azure OpenAI GPT-4 Vision for OCR, outputting markdown and searchable PDF.
    """
    ensure_data_dir()

    # Check if output files already exist
    if os.path.exists("./data/ocr.md") and os.path.exists("./data/ocr_searchable.pdf"):
        print("Output files already exist. Skipping document processing.")
        return True

    temp_pdf_path = save_uploaded_file(uploaded_file)

    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_path(
            pdf_path=temp_pdf_path,
            output_folder=temp_dir,
            fmt='png',
            dpi=300
        )
        md_output = [None] * len(images)
        img_text_pairs = [None] * len(images)

        def ocr_and_text(idx_img):
            i, image = idx_img
            try:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                text = call_gpt4v_ocr(img_byte_arr.getvalue())
                return (i, image, text)
            except Exception as e:
                _log.error(f"Error during GPT-4V OCR processing on page {i+1}: {e}")
                return (i, image, "[Error processing page]")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(ocr_and_text, enumerate(images)))
        results.sort(key=lambda x: x[0])
        for i, image, text in results:
            md_output[i] = f"\n\n## Page {i+1}\n\n{text.strip()}"
            img_text_pairs[i] = (image, text)

        # Create searchable PDF with fitz
        doc = fitz.open()
        for i, (image, text) in enumerate(img_text_pairs):
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            img_pdf = fitz.open(stream=img_byte_arr.read(), filetype="png")
            rect = img_pdf[0].rect
            page = doc.new_page(width=rect.width, height=rect.height)
            page.insert_image(rect, stream=img_byte_arr.getvalue())
            # Overlay text as invisible layer
            if text and text != "[Error processing page]":
                page.insert_textbox(
                    rect,
                    text,
                    fontsize=12,
                    fontname="helv",
                    color=(1, 1, 1, 0),  # invisible
                    overlay=True,
                    render_mode=3  # invisible text
                )
        doc.save("./data/ocr_searchable.pdf")
        doc.close()
        # Write the markdown output
        with open("./data/ocr.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md_output))
    cleanup_file(temp_pdf_path)
    return True

def call_gpt4v_ocr(image_bytes):
    """
    Calls Azure OpenAI GPT-4 Vision API to perform OCR on the given image bytes.
    Returns the recognized text.
    """
    import base64
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.environ.get("AZURE_OPENAI_VERSION", "2024-02-15-preview")
    if not endpoint or not api_key or not deployment:
        raise RuntimeError("Azure OpenAI Vision environment variables are not set.")
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image as accurately as possible. Return only the plain text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False
    }
    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]
