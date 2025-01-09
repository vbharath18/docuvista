import streamlit as st
import pandas as pd
import plotly.express as px
import logging
import fitz  # PyMuPDF
import os
import base64
from dotenv import load_dotenv
from rag_handler import process_pdf_for_embeddings, setup_rag
from azure_document_processor import process_uploaded_pdf
from crewai_processor import process_with_crew

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
        return pd.read_csv(file_path, on_bad_lines='skip')
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

# Add this cache decorator for RAG setup
@st.cache_resource
def get_rag_chain(file_path: str):
    """Cache the RAG chain setup to avoid reprocessing"""
    document_splits = process_pdf_for_embeddings(file_path)
    return setup_rag(document_splits)

# Add this after the existing helper functions
def reload_data():
    """Reload all data sources"""
    df = load_data("./data/final.csv")
    pdf_doc = load_pdf("./data/ocr_searchable.pdf")
    if pdf_doc is not None:
        st.session_state.pdf_doc = pdf_doc
    return df, pdf_doc

# Add this after the helper functions
def check_required_files():
    """Check if required files exist"""
    return os.path.exists("./data/final.csv") and os.path.exists("./data/ocr_searchable.pdf")

# Add this helper function after the other helper functions
def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate if DataFrame has required columns for visualization"""
    # Required columns for basic functionality
    required_columns = ["Test", "Test type", "Observation"]
    
    # Required columns for visualizations
    viz_columns = ["Test type", "Test", "Result", "Unit", "Interval", "Observation"]
    
    # Check basic required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        present_columns = df.columns.tolist()
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info(f"Current columns in your data: {', '.join(present_columns)}")
        return False
    
    # Check visualization columns
    missing_viz_columns = [col for col in viz_columns if col not in df.columns]
    
    if missing_viz_columns:
        st.warning(f"Some visualization columns are missing: {', '.join(missing_viz_columns)}")
        
    # Clean column names by removing trailing spaces
    df.columns = df.columns.str.strip()
    
    if df.empty:
        st.error("DataFrame is empty")
        return False
        
    return True

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
tabs = st.tabs(["Upload", "Report", "Triage"])

# --- Upload ---
with tabs[0]:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    # Process the uploaded PDF
                    st.info("Step 1/3: Converting document to OCR format...")
                    success = process_uploaded_pdf(uploaded_file)
                    
                    if success:
                        st.info("Step 2/3: Extracting relavant data & generating reports using AI agents...")
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
                            x="Observation",
                            y="Test",
                            color="Test type",
                            title="Test Analysis Distribution",
                            template="plotly_white",
                            height=300
                        )
                        fig_bar.update_layout(margin=dict(t=30, b=40, l=20, r=20))
                        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
                        
                        # Histogram
                        fig_hist = px.histogram(
                            df,
                            x="Observation",
                            color="Test type",
                            title="Observation Distribution Analysis",
                            template="plotly_white",
                            height=300
                        )
                        fig_hist.update_layout(margin=dict(t=30, b=40, l=20, r=20))
                        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
                    except Exception as e:
                        st.error(f"Error creating visualizations: {str(e)}")

# --- Triage ---
with tabs[2]:
    if not st.session_state.files_ready:
        st.info("Please upload and process a document first using the Upload tab.")
    else:
        if st.session_state.needs_reload:
            _, _ = reload_data()
            st.session_state.needs_reload = False
        
        st.header("Triage")
        
        search_col, qa_col = st.columns(2)
        
        with search_col:
            st.subheader("Search")
            keyword = st.text_input("Enter keyword to search in PDF:")
            
            # Use session state for PDF document
            if st.session_state.pdf_doc is None:
                st.error("PDF document failed to load. Please try uploading the document again.")
            else:
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
        
        with qa_col:
            st.subheader("Q&A")
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