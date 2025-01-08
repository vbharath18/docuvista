import os
import base64
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption

# Load environment variables
load_dotenv()

def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF using Azure Document Intelligence"""
    # Ensure the data directory exists
    os.makedirs("./data", exist_ok=True)
    
    # Save uploaded file temporarily
    temp_path = f"./data/temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
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
    
    # Clean up temp file
    os.remove(temp_path)
    return True
