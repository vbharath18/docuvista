import streamlit as st
import pandas as pd
import plotly.express as px
import logging
import fitz  # PyMuPDF
import os
import base64
from langchain import hub
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, AnalyzeOutputOption
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool, FileWriterTool

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.ERROR)

# Set the Streamlit app layout to wide format
st.set_page_config(layout="wide")

# --- Helper Functions ---
@st.cache_data 
def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        st.error(f"Data file not found: {file_path}")
        return pd.DataFrame()  

def load_pdf(file_path: str) -> fitz.Document:
    """Loads a PDF file."""
    try:
        return fitz.open(file_path)
    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        st.error(f"Error loading PDF: {e}")
        return None

def search_pdf(doc: fitz.Document, keyword: str):
    """Searches for a keyword in the PDF and returns the highlighted pages and count."""
    results = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_instances = page.search_for(keyword)
        if text_instances:
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.update()
            results.append(page_num + 1)
    return results

def process_pdf_for_embedding(file_path: str):
    """Process PDF document for embedding using Azure Document Intelligence"""
    try:
        doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        
        # Load document using Azure Document Intelligence
        loader = AzureAIDocumentIntelligenceLoader(
            file_path=file_path,
            api_key=doc_intelligence_key,
            api_endpoint=doc_intelligence_endpoint,
            api_model="prebuilt-layout"
        )
        docs = loader.load()
        
        # Split document into chunks
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        docs_string = docs[0].page_content
        splits = text_splitter.split_text(docs_string)
        
        return splits
    except Exception as e:
        logging.error(f"Error processing PDF for embedding: {e}")
        st.error(f"Error processing PDF for embedding: {e}")
        return None

# Add new helper functions for document processing
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

def process_with_crew():
    """Process the markdown file with CrewAI"""
    llm = LLM(
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        base_url=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY")
    )
    
    file_read_tool = FileReadTool()
    file_writer_tool = FileWriterTool()
    
    csv_agent = Agent(
        role="Extract, process data and record data",
        goal="Extract test results as instructed. The result MUST be valid CSV. Read right to left, treat each table as a separately.",
        backstory="You are a lab test results data extraction agent",
        tools=[file_read_tool],
        llm=llm,
    )
    
    create_CSV = Task(
        description="""
            Analyse './data/ocr.md' the data provided - it is in Markdown format.
            Your output should be in CSV format. Respond without using Markdown code fences.
            Your task is to:
               Ensure that string data is enclosed in quotes.
               Each item in the list should have its columns populated as follows. No additional columns should be added.
                    "Test type": Name of the test type is found after Patient Information,                
                    "Test": Name of the test,
                    "Result": Result of the test,
                    "Unit": Unit of the test,
                    "Interval": Biological reference interval,
                If a column is not applicable, leave it empty.
            """,
        expected_output="A correctly formatted CSV data structure with only",
        agent=csv_agent,
        output_file="./data/rp.csv",
        tools=[file_read_tool]
    )
    
    add_sentiment = Task(
        description="""
            Analyse CSV data and calculate the sentiment of each test results in
            the 'Sentiment' column. Add a new column to the CSV that records that sentiment.
            Your output should be in CSV format. Respond without using Markdown code fences.
            """,
        expected_output="A correctly formatted CSV data file",
        agent=csv_agent,
        output_file="./data/op.csv",
        tools=[file_read_tool]
    )
    
    crew = Crew(
        agents=[csv_agent, csv_agent],
        tasks=[create_CSV, add_sentiment],
        verbose=False,
    )
    
    return crew.kickoff()

# --- RAG Setup ---
def setup_rag(document_splits=None):
    """Initialize RAG components with document embedding"""
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    vector_store_address = os.getenv("AZURE_SEARCH_ENDPOINT")
    vector_store_password = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    
    # Initialize embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-05-15",
        azure_endpoint=azure_endpoint,
        api_key=azure_openai_api_key,
    )
    
    # Initialize vector store
    vector_store = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name="healthrecords",
        embedding_function=embeddings.embed_query,
    )
    
    # Add documents to vector store if provided
    if document_splits:
        vector_store.add_documents(documents=document_splits)
    
    # Initialize retriever and LLM
    retriever = vector_store.as_retriever(search_type="similarity", k=3)
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

# Add this cache decorator for RAG setup
@st.cache_resource
def get_rag_chain(file_path: str):
    """Cache the RAG chain setup to avoid reprocessing"""
    document_splits = process_pdf_for_embedding(file_path)
    return setup_rag(document_splits)

# Add this after the existing helper functions
def reload_data():
    """Reload all data sources"""
    df = load_data("./data/op.csv")
    pdf_doc = load_pdf("./data/ocr_searchable.pdf")
    if pdf_doc is not None:
        st.session_state.pdf_doc = pdf_doc
    return df, pdf_doc

# Add this after the helper functions
def check_required_files():
    """Check if required files exist"""
    return os.path.exists("./data/op.csv") and os.path.exists("./data/ocr_searchable.pdf")

# Add this helper function after the other helper functions
def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate if DataFrame has required columns for visualization"""
    required_columns = ["Test", "Test type", "Sentiment"]
    return all(col in df.columns for col in required_columns) and not df.empty

# Modify the app initialization
if 'needs_reload' not in st.session_state:
    st.session_state.needs_reload = True
if 'files_ready' not in st.session_state:
    st.session_state.files_ready = check_required_files()
if 'pdf_doc' not in st.session_state:
    st.session_state.pdf_doc = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

# --- Tabs ---
tabs = st.tabs(["Upload", "Report", "Search", "Q&A"])

# --- Upload ---
with tabs[0]:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    # Process the uploaded PDF
                    st.info("Step 1/3: Converting document...")
                    success = process_uploaded_pdf(uploaded_file)
                    
                    if success:
                        st.info("Step 2/3: Extracting data...")
                        crew_result = process_with_crew()
                        
                        st.info("Step 3/3: Initializing Q&A system...")
                        get_rag_chain.clear()
                        _ = get_rag_chain("./data/ocr_searchable.pdf")
                        
                        st.session_state.needs_reload = True
                        st.session_state.files_ready = True
                        st.success("Document processed successfully! You can now use the Report, Search, and Q&A tabs.")
                    else:
                        st.error("Failed to process document")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

# --- Report ---
with tabs[1]:
    if not st.session_state.files_ready:
        st.info("Please upload and process a document first using the Upload tab.")
    else:
        # Initialize df and pdf_doc
        df = st.session_state.df  # Use session state DataFrame
        pdf_doc = None
        
        if st.session_state.needs_reload:
            df, pdf_doc = reload_data()
            st.session_state.df = df  # Store DataFrame in session state
            st.session_state.needs_reload = False
        
        report_col, viz_col = st.columns([0.45, 0.55])
        
        with report_col:
            st.markdown("### 📊 Report Analysis")
            if df.empty or df is None:
                st.warning("No data available. The processed CSV file might be empty or failed to load.")
            else:
                st.dataframe(df, use_container_width=True)

        with viz_col:
            st.markdown("### 📈 Key Insights")
            
            if not validate_dataframe(df):
                st.warning("Cannot create visualizations. Data format is incorrect or missing required columns.")
            else:
                with st.spinner("Loading charts..."):
                    try:
                        # Bar Chart
                        fig_bar = px.bar(
                            df,
                            x="Sentiment",
                            y="Test",
                            color="Test type",
                            title="Sentiment Analysis Distribution",
                            template="plotly_white",
                            height=300
                        )
                        fig_bar.update_layout(margin=dict(t=30, b=40, l=20, r=20))
                        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
                        
                        # Histogram
                        fig_hist = px.histogram(
                            df,
                            x="Sentiment",
                            color="Test type",
                            title="Sentiment Distribution Analysis",
                            template="plotly_white",
                            height=300
                        )
                        fig_hist.update_layout(margin=dict(t=30, b=40, l=20, r=20))
                        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
                    except Exception as e:
                        st.error(f"Error creating visualizations: {str(e)}")

# --- Search ---
with tabs[2]:
    if not st.session_state.files_ready:
        st.info("Please upload and process a document first using the Upload tab.")
    else:
        if st.session_state.needs_reload:
            _, _ = reload_data()
            st.session_state.needs_reload = False
        
        st.header("Search")
        keyword = st.text_input("Enter keyword to search in PDF:")
        
        # Use session state for PDF document
        if st.session_state.pdf_doc is None:
            st.error("PDF document failed to load. Please try uploading the document again.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Search"):
                    if keyword:
                        try:
                            with st.spinner("Searching..."):
                                results = search_pdf(st.session_state.pdf_doc, keyword)
                                if results:
                                    st.session_state.search_results = results
                                    st.success(f"Found {len(results)} matches")
                                    st.write(f"Keyword found on pages: {results}")
                                    for page_num in results:
                                        st.write(f"Page {page_num}")
                                        try:
                                            page = st.session_state.pdf_doc.load_page(page_num - 1)
                                            pix = page.get_pixmap()
                                            st.image(pix.tobytes(), caption=f"Page {page_num}", width=700)
                                        except Exception as e:
                                            st.error(f"Error displaying page {page_num}: {str(e)}")
                                else:
                                    st.info(f"No matches found for '{keyword}'")
                        except Exception as e:
                            st.error(f"Error during search: {str(e)}")
                    else:
                        st.warning("Please enter a keyword to search")
                else:
                    if st.session_state.search_results:
                        st.write(f"Keyword found on pages: {st.session_state.search_results}")
                        for page_num in st.session_state.search_results:
                            st.write(f"Page {page_num}")
                            try:
                                page = st.session_state.pdf_doc.load_page(page_num - 1)
                                pix = page.get_pixmap()
                                st.image(pix.tobytes(), caption=f"Page {page_num}", width=700)
                            except Exception as e:
                                st.error(f"Error displaying page {page_num}: {str(e)}")
            
            with col2:
                if st.button("Clear Search Results"):
                    try:
                        for page_num in range(len(st.session_state.pdf_doc)):
                            page = st.session_state.pdf_doc.load_page(page_num)
                            for annot in page.annots():
                                annot.delete()
                        st.session_state.search_results = []
                        st.success("Search results cleared")
                    except Exception as e:
                        st.error(f"Error clearing search results: {str(e)}")

# --- Q&A ---
with tabs[3]:
    if not st.session_state.files_ready:
        st.info("Please upload and process a document first using the Upload tab.")
    else:
        st.header("Q&A")
        st.write("Ask questions about the document content using RAG")
        
        # Initialize RAG chain when Q&A tab is accessed
        with st.spinner("Processing document for Q&A..."):
            rag_chain = get_rag_chain("./data/ocr_searchable.pdf")
        
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    try:
                        answer = rag_chain.invoke(question)
                        st.write("Answer:", answer)
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
            else:
                st.warning("Please enter a question")