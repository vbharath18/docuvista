import logging
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
import tempfile
from pypdf import PdfMerger
import io
from PIL import Image, ImageFilter, ImageEnhance

_log = logging.getLogger(__name__)

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    
    # Apply thresholding
    image = image.point(lambda x: 0 if x < 140 else 255, '1')
    
    # Denoise the image
    image = image.filter(ImageFilter.MedianFilter())
    
    # Resize image to improve OCR accuracy
    image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
    
    return image

def process_uploaded_pdf_with_tesseract(uploaded_file):
    """
    Process uploaded PDF using Tesseract OCR, outputting markdown and searchable PDF.
    """
    import os
    import tempfile
    from pdf2image import convert_from_path
    import pytesseract
    from pypdf import PdfMerger
    import io
    from PIL import Image, ImageFilter, ImageEnhance

    # Ensure the data directory exists
    os.makedirs("./data", exist_ok=True)

    # Check if output files already exist
    if os.path.exists("./data/ocr.md") and os.path.exists("./data/ocr_searchable.pdf"):
        print("Output files already exist. Skipping document processing.")
        return True

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.getbuffer())
        temp_pdf_path = temp_pdf.name

    def preprocess_image(image):
        image = image.convert('L')
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)
        image = image.point(lambda x: 0 if x < 140 else 255, '1')
        image = image.filter(ImageFilter.MedianFilter())
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
        return image

    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_path(
            pdf_path=temp_pdf_path,
            output_folder=temp_dir,
            fmt='png',
            dpi=300
        )
        merger = PdfMerger()
        md_output = []
        for i, image in enumerate(images):
            try:
                preprocessed_image = preprocess_image(image)
                tess_config = '--oem 3 --psm 3'
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                    preprocessed_image,
                    lang='eng',
                    config=tess_config,
                    extension='pdf'
                )
                text = pytesseract.image_to_string(
                    preprocessed_image,
                    lang='eng',
                    config=tess_config
                )
                pdf_file_in_memory = io.BytesIO(pdf_bytes)
                merger.append(pdf_file_in_memory)
                # Add page text to markdown output
                md_output.append(f"\n\n## Page {i+1}\n\n{text.strip()}")
            except Exception as e:
                _log.error(f"Error during OCR processing on page {i+1}: {e}")
                continue
        # Write the merged PDF to file
        with open("./data/ocr_searchable.pdf", "wb") as f:
            merger.write(f)
        merger.close()
        # Write the markdown output
        with open("./data/ocr.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md_output))
    os.remove(temp_pdf_path)
    return True

def main():
    logging.basicConfig(level=logging.INFO)

    input_doc_path = Path("./sample/pathology-report-scanned.pdf")
    output_pdf_path = Path("scratch/output.pdf")
    output_hocr_path = Path("scratch/output.hocr")
    output_txt_path = Path("scratch/output.txt")

    # Create a temporary directory to store images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Convert PDF to images
        images = convert_from_path(
            pdf_path=str(input_doc_path),
            output_folder=str(temp_dir_path),
            fmt='png',
            dpi=300
        )

        # Initialize the PDF merger
        merger = PdfMerger()

        # Initialize HOCR output
        hocr_output = []

        txt_output = []

        # Process each image with pytesseract
        for i, image in enumerate(images):
            try:
                # Preprocess the image
                preprocessed_image = preprocess_image(image)

                # tess_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
                
                tess_config = '--oem 3 --psm 3'
                # Recognize text and get PDF data
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                    preprocessed_image,
                    lang='eng',
                    config=tess_config,
                    extension='pdf'
                )       

                # Recognize text and get HOCR data
                hocr_bytes = pytesseract.image_to_pdf_or_hocr(
                    preprocessed_image,
                    lang='eng',
                    config=tess_config,
                    extension='hocr'
                )

                # Recognize text and get text data
                txt_bytes = pytesseract.image_to_string(
                    preprocessed_image,
                    lang='eng',
                    config=tess_config
                )

                # Add PDF data to the merger
                pdf_file_in_memory = io.BytesIO(pdf_bytes)
                merger.append(pdf_file_in_memory)

                # Add HOCR data to the output list
                if isinstance(hocr_bytes, bytes):
                    hocr_output.append(hocr_bytes.decode('utf-8'))
                else:
                    hocr_output.append(hocr_bytes)
                                
                # Add text data to the output list
                if isinstance(txt_bytes, bytes):
                    txt_output.append(txt_bytes.decode('utf-8'))
                else:
                    txt_output.append(txt_bytes)

                _log.info(f"OCR completed for page {i+1}/{len(images)}.")
            except Exception as e:
                _log.error(f"Error during OCR processing on page {i+1}: {e}")
                continue

        # Ensure output directory exists
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        output_hocr_path.parent.mkdir(parents=True, exist_ok=True)
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the merged PDF to file
        with output_pdf_path.open("wb") as f:
            merger.write(f)
        merger.close()

        # Write the combined HOCR to file
        with output_hocr_path.open("w", encoding='utf-8') as f:
            f.write("\n".join(hocr_output))

        with output_txt_path.open("w", encoding='utf-8') as f:
            f.write("\n".join(txt_output))

        _log.info(f"OCR processing completed. PDF output saved to {output_pdf_path}")
        _log.info(f"OCR processing completed. HOCR output saved to {output_hocr_path}")
        _log.info(f"OCR processing completed. TXT output saved to {output_txt_path}")

if __name__ == "__main__":
    main()