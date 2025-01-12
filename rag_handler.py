import os
import logging
import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain import hub
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

def process_pdf_for_embeddings(file_path: str):
    """Process PDF for embedding using langchain components"""
    try:
        # Open the PDF file
        pdf_document = fitz.open(file_path)
        
        # Extract text from each page
        text_content = []
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text_content.append(page.get_text())
        
        # Combine all text
        full_text = "\n\n".join(text_content)
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Split text into chunks
        texts = text_splitter.create_documents([full_text])
        
        pdf_document.close()
        return texts
    except Exception as e:
        logging.error(f"Error processing PDF for embedding: {e}")
        return None

def setup_rag(document_splits=None):
    """Initialize RAG components with document embedding using FAISS"""
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    # Initialize embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-05-15",
        azure_endpoint=azure_endpoint,
        api_key=azure_openai_api_key,
    )
    
    # Initialize or load FAISS vector store
    if document_splits:
        vector_store = FAISS.from_documents(document_splits, embeddings)
        # Optionally save the index
        vector_store.save_local("./data/faiss_index")
    else:
        # Load existing index if available
        try:
            vector_store = FAISS.load_local("./data/faiss_index", embeddings)
        except:
            # Return None or handle the case when no index exists
            return None
    
    # Initialize retriever and LLM
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = AzureChatOpenAI(
        openai_api_version="2024-08-01-preview",
        azure_deployment="gpt-4o-mini",
        temperature=0,
    )
    
    # Setup RAG chain
    prompt = hub.pull("rlm/rag-prompt")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
