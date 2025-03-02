# DocuVista

DocuVista is an advanced document processing application that combines OCR (Optical Character Recognition), RAG (Retrieval-Augmented Generation), and AI-powered document analysis to extract, process, and analyze information from PDF documents, particularly focusing on medical reports and similar document types.

## Features

- **Document Processing**: Convert scanned PDFs to searchable documents using Tesseract OCR and Azure Document Intelligence
- **Information Extraction**: Extract structured data from documents using AI frameworks (CrewAI or AutoGen)
- **Interactive Reports**: Visualize extracted data with interactive charts and tables
- **Document Search**: Search for keywords in processed documents with highlighted results
- **Q&A System**: Ask questions about document content using RAG with Azure OpenAI

## Architecture

DocuVista integrates multiple AI technologies:

- **OCR Processing**: Tesseract OCR and Azure Document Intelligence
- **Data Extraction**: 
  - CrewAI Agents and Tasks for structured data extraction
  - AutoGen framework for alternative AI processing
- **RAG System**: Azure OpenAI with FAISS vector database for document Q&A

## Installation

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- Azure API keys (for Azure OpenAI and Document Intelligence)

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/docuvista.git
cd docuvista
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Azure credentials
```
AZURE_OPENAI_VERSION="2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment-name"
AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS="azure/text-embedding-ada-002"
AZURE_OPENAI_ENDPOINT="your-azure-openai-endpoint"
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="your-document-intelligence-endpoint"
AZURE_DOCUMENT_INTELLIGENCE_KEY="your-document-intelligence-key"
```

### Docker Deployment

To run the application in a container:

```bash
docker build -t docuvista .
docker run -p 8501:8501 docuvista
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Processing Documents

1. Navigate to the "Upload" tab
2. Upload a PDF document for processing
3. Select either CrewAI or AutoGen for the AI processing framework
4. Click "Process Document"
5. Wait for processing to complete (this may take a few minutes for large documents)

### Exploring Results

After processing:

1. Use the "Report" tab to view extracted data and visualizations
2. Use the "Triage" tab to:
   - Search for keywords in the document
   - Ask questions about the document content

## Core Components

- `tesseract_processor.py`: OCR processing with Tesseract
- `azure_document_processor.py`: Document processing with Azure Document Intelligence
- `crewai_processor.py`: Data extraction with CrewAI agents
- `autogen_processor.py`: Alternative data extraction with AutoGen
- `rag_handler.py`: Q&A system using retrieval-augmented generation
- `app.py`: Streamlit web application

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Tesseract OCR for text recognition
- Azure Document Intelligence for enhanced document processing
- CrewAI and AutoGen for agent-based AI processing
- Streamlit for the web interface
- PyMuPDF (fitz) for PDF manipulation