import os
import logging
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain import hub
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

def process_pdf_for_embeddings(file_path: str):
    """Process markdown for embedding using langchain components"""
    try:
        # Load markdown file instead of PDF
        with open("./data/ocr.md", "r") as file:
            docs_string = file.read()
        
        # Split document into chunks
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        splits = text_splitter.split_text(docs_string)
        
        return splits
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
