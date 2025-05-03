import os
import base64
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption
from .utils import ensure_data_dir, save_uploaded_file, cleanup_file

# Load environment variables
load_dotenv()

def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF using Azure Document Intelligence"""
    ensure_data_dir()
    
    # Check if output files already exist
    if os.path.exists("./data/ocr.md") and os.path.exists("./data/ocr_searchable.pdf"):
        print("Output files already exist. Skipping document processing.")
        return True
    
    temp_path = save_uploaded_file(uploaded_file)
    
    # Initialize Azure Document Intelligence client
    endpoint = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"]
    key = os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    
    # Process for markdown
    with open(temp_path, "rb") as f:
        analyze_request = {
            "base64Source": base64.b64encode(f.read()).decode("utf-8")
        }
        poller = client.begin_analyze_document(
            "prebuilt-layout", 
            body=analyze_request,
            output_content_format="markdown"
        )
    result = poller.result()
    
    # Save markdown output
    with open("./data/ocr.md", "w") as f:
        f.write(result.content)
    
    # Process for searchable PDF
    with open(temp_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-read",
            body=f,
            output=[AnalyzeOutputOption.PDF],
        )
    result = poller.result()
    operation_id = poller.details["operation_id"]
    
    response = client.get_analyze_result_pdf(model_id=result.model_id, result_id=operation_id)
    with open("./data/ocr_searchable.pdf", "wb") as writer:
        writer.writelines(response)
    
    cleanup_file(temp_path)
    return True
