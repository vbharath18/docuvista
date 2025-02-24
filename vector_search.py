import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain import hub

def process_markdown_for_embeddings():
    """Process Markdown file for embedding using langchain components"""
    file_path = "/workspaces/docuvista/data/ocr.md"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Split text into chunks
        texts = text_splitter.create_documents([full_text])
        return texts
    except Exception as e:
        logging.error(f"Error processing Markdown for embedding: {e}")
        return None

def setup_rag(document_splits=None):
    """Initialize RAG components with document embedding using FAISS"""
    global vector_store  # Add this line to modify the global variable
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
    return vector_store

def is_vector_store_initialized():
    """Check if the vector store is initialized."""
    return vector_store is not None

def semantic_search(query, k, filter=None):
    """Perform semantic search from the vector store to retrieve relevant chunks"""
    if not is_vector_store_initialized():
        logging.error("Vector store is not initialized.")
        return None
    
    results = vector_store.similarity_search(query, k=k, filter=filter)

    # for res in results:
    #     print(f"* {res.page_content}")
    return results