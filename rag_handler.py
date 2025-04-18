import os
import logging
from pathlib import Path
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain import hub


def process_pdf_for_embeddings(file_path: str) -> list:
    """
    Extracts and splits text from a PDF file for embedding.
    Args:
        file_path (str): Path to the PDF file.
    Returns:
        list: List of langchain Document objects containing text chunks, or None on failure.
    """
    try:
        # Use context manager for resource safety
        with fitz.open(file_path) as pdf_document:
            text_content = [page.get_text() for page in pdf_document]
        full_text = "\n\n".join(text_content)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents([full_text])
        return texts
    except Exception as e:
        logging.exception(f"Error processing PDF for embedding: {e}")
        return None


def setup_rag(document_splits: list = None):
    """
    Initializes Retrieval-Augmented Generation (RAG) components with FAISS and Azure OpenAI.
    Args:
        document_splits (list, optional): List of langchain Document objects. If None, loads existing FAISS index.
    Returns:
        rag_chain: A runnable RAG chain, or None if initialization fails.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_api_version = os.getenv("AZURE_OPENAI_VERSION")
    azure_embedding_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS")

    if not azure_embedding_deployment:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS environment variable is not set or is incorrect. Please set it to your Azure OpenAI Embeddings deployment name.")

    logging.info(f"Using Azure OpenAI Embeddings deployment: {azure_embedding_deployment}")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_embedding_deployment,
        openai_api_version=azure_openai_api_version,
        azure_endpoint=azure_endpoint,
        api_key=azure_openai_api_key,
    )

    faiss_index_path = Path("./data/faiss_index")
    faiss_index_file = faiss_index_path / "index.faiss"
    try:
        if document_splits:
            vector_store = FAISS.from_documents(document_splits, embeddings)
            vector_store.save_local(str(faiss_index_path))
        else:
            if not faiss_index_file.exists():
                logging.error(f"FAISS index file not found at {faiss_index_file}. Please build the index first by providing document_splits.")
                return None
            vector_store = FAISS.load_local(str(faiss_index_path), embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logging.error(f"Error initializing FAISS vector store: {e}")
        return None

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = AzureChatOpenAI(
        openai_api_version=azure_openai_api_version,
        azure_deployment=azure_deployment,
        temperature=0,
    )
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
