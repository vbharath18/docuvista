import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
import subprocess

# Load environment variables
load_dotenv()

def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF using Tesseract OCR"""
    # Ensure the data directory exists
    os.makedirs("./data", exist_ok=True)
    
    # Save uploaded file temporarily
    temp_path = f"./data/temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Convert PDF to images
    pdf_document = fitz.open(temp_path)
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        image_path = f"./data/page_{page_num}.png"
        pix.save(image_path)
        images.append(image_path)
    
    # Process for markdown
    markdown_content = ""
    for image_path in images:
        text = subprocess.run(["tesseract", image_path, "stdout"], capture_output=True, text=True).stdout
        markdown_content += text + "\n\n"
    
    # Save markdown output
    with open("./data/ocr.md", "w") as f:
        f.write(markdown_content)
    
    # Process for searchable PDF
    searchable_pdf_path = "./data/ocr_searchable.pdf"
    # Assuming images are in the correct order
    first_image = fitz.open(images[0])
    for image_path in images[1:]:
        img_doc = fitz.open(image_path)
        first_image.insert_pdf(img_doc)
    first_image.save(searchable_pdf_path)
    
    # Clean up temp file and images
    os.remove(temp_path)
    for image_path in images:
        os.remove(image_path)
    
    return True
