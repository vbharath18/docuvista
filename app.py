import streamlit as st
import pandas as pd
import plotly.express as px
import logging
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from rag_handler import process_pdf_for_embeddings, setup_rag
from azure_document_processor import process_uploaded_pdf
from crewai_processor import process_with_crew
from autogen_processor import process_with_autogen
import asyncio

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

@st.cache_resource
def get_rag_chain(file_path: str):
    """Cache the RAG chain setup to avoid reprocessing"""
    document_splits = process_pdf_for_embeddings(file_path)
    return setup_rag(document_splits)

def reload_data():
    """Reload all data sources"""
    df = load_data("./data/final.csv")
    pdf_doc = load_pdf("./data/ocr_searchable.pdf")
    if pdf_doc is not None:
        st.session_state.pdf_doc = pdf_doc
    return df, pdf_doc

def check_required_files():
    """Check if required files exist"""
    return os.path.exists("./data/final.csv") and os.path.exists("./data/ocr_searchable.pdf")

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate if DataFrame has required columns for visualization"""
    required_columns = ["Test", "Test type", "Observation"]
    viz_columns = ["Test type", "Test", "Result", "Unit", "Interval", "Observation"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        present_columns = df.columns.tolist()
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info(f"Current columns in your data: {', '.join(present_columns)}")
        return False
    
    missing_viz_columns = [col for col in viz_columns if col not in df.columns]
    if missing_viz_columns:
        st.warning(f"Some visualization columns are missing: {', '.join(missing_viz_columns)}")
        
    df.columns = df.columns.str.strip()
    
    if df.empty:
        st.error("DataFrame is empty")
        return False
        
    return True

# --- Helper Classes ---
import io
import sys

class StreamToStreamlit(io.StringIO):
    """Redirects stdout to a Streamlit container with colorized logs."""
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.log = ""
    def write(self, s):
        self.log += s
        self._render_log()
        return len(s)
    def flush(self):
        pass
    def _render_log(self):
        html_log = ""
        for line in self.log.splitlines():
            if "ERROR" in line:
                html_log += f'<div style="color:#ff4b4b;">{line}</div>'
            elif "WARNING" in line:
                html_log += f'<div style="color:#ffa500;">{line}</div>'
            elif "INFO" in line:
                html_log += f'<div style="color:#1e90ff;">{line}</div>'
            else:
                html_log += f'<div style="color:#d3d3d3;">{line}</div>'
        self.container.markdown(
            f'''<div style="height:350px;overflow-y:auto;background:#181818;padding:8px;border-radius:6px;font-size:13px;">{html_log}</div>''',
            unsafe_allow_html=True
        )

def render_log_to_streamlit(log_container, log_text):
    """Render log text to a Streamlit container with colorization."""
    html_log = ""
    for line in log_text.splitlines():
        if "ERROR" in line:
            html_log += f'<div style="color:#ff4b4b;">{line}</div>'
        elif "WARNING" in line:
            html_log += f'<div style="color:#ffa500;">{line}</div>'
        elif "INFO" in line:
            html_log += f'<div style="color:#1e90ff;">{line}</div>'
        else:
            html_log += f'<div style="color:#d3d3d3;">{line}</div>'
    log_container.markdown(
        f'''<div style="height:350px;overflow-y:auto;background:#181818;padding:8px;border-radius:6px;font-size:13px;">{html_log}</div>''',
        unsafe_allow_html=True
    )

from contextlib import contextmanager

@contextmanager
def redirect_stdout_to_streamlit(container):
    """Context manager to redirect stdout to Streamlit log container."""
    stream = StreamToStreamlit(container)
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout

@contextmanager
def capture_stdout():
    """Context manager to capture stdout into a StringIO buffer."""
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        yield buffer
    finally:
        sys.stdout = old_stdout

# --- App Initialization ---
if 'needs_reload' not in st.session_state:
    st.session_state.needs_reload = True
if 'files_ready' not in st.session_state:
    st.session_state.files_ready = check_required_files()
if 'pdf_doc' not in st.session_state:
    st.session_state.pdf_doc = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'current_page_idx' not in st.session_state:
    st.session_state.current_page_idx = 0
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'answer' not in st.session_state:
    st.session_state.answer = ""

# --- Tabs ---
tabs = st.tabs(["Upload", "Report", "Triage"])

# --- Upload ---
with tabs[0]:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        processing_option = st.radio(
            "Choose which Agentic AI framework you want to use for the processing:",
            ("Use CrewAI", "Use AutoGen")
        )
        
        if st.button("Process Document"):
            progress_bar = st.progress(0, text="Starting document processing...")
            with st.spinner("Processing document..."):
                try:
                    st.info("Step 1/3: Converting scanned document to a machine readable format...")
                    progress_bar.progress(10, text="Converting scanned document...")
                    success = process_uploaded_pdf(uploaded_file)
                    progress_bar.progress(40, text="Document converted. Extracting data...")
                    if success:
                        log_container = st.empty()
                        if processing_option == "Use CrewAI":
                            st.info("Step 2/3: Extracting relevant data & generating reports using CrewAI...")
                            with redirect_stdout_to_streamlit(log_container):
                                crew_result = process_with_crew()
                        else:
                            st.info("Step 2/3: Extracting relevant data & generating reports using AutoGen...")
                            with capture_stdout() as log_buffer:
                                asyncio.run(process_with_autogen())
                            render_log_to_streamlit(log_container, log_buffer.getvalue())
                        progress_bar.progress(80, text="Initializing Q&A system...")
                        st.info("Step 3/3: Initializing Q&A system...")
                        get_rag_chain.clear()
                        _ = get_rag_chain("./data/ocr_searchable.pdf")
                        progress_bar.progress(100, text="Done!")
                        st.session_state.needs_reload = True
                        st.session_state.files_ready = True
                        st.success("Document processed successfully! You can now use the Report, Search, and Q&A tabs.")
                        progress_bar.empty()
                    else:
                        progress_bar.empty()
                        st.error("Failed to process document")
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Error processing document: {str(e)}")

# --- Report ---
with tabs[1]:
    if not st.session_state.files_ready:
        st.info("Please upload and process a document first using the Upload tab.")
    else:
        df = st.session_state.df
        pdf_doc = None
        
        if st.session_state.needs_reload:
            df, pdf_doc = reload_data()
            st.session_state.df = df
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
            df.columns = df.columns.str.strip()
            if not validate_dataframe(df):
                st.warning("Cannot create visualizations. Data format is incorrect or missing required columns.")
            else:
                with st.spinner("Loading charts..."):
                    try:
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
                        
                        fig_hist = px.histogram(
                            df,
                            x="Observation",
                            color="Test type",
                            title="Observation Analysis Distribution",
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
            
            if st.session_state.pdf_doc is None:
                st.error("PDF document failed to load. Please try uploading the document again.")
            else:
                search_col1, search_col2 = st.columns([1, 1], gap="small")
                with search_col1:
                    search_clicked = st.button("Search")
                with search_col2:
                    clear_clicked = st.button("Clear Search Results")
                
                if search_clicked:
                    if keyword:
                        try:
                            with st.spinner("Searching..."):
                                results = search_pdf(st.session_state.pdf_doc, keyword)
                                if results:
                                    st.session_state.search_results = results
                                    st.session_state.current_page_idx = 0
                                    st.success(f"Found {len(results)} matches")
                                    st.write(f"Keyword found on pages: {results}")
                                    
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    with col1:
                                        if st.button("Previous", disabled=st.session_state.current_page_idx == 0):
                                            st.session_state.current_page_idx -= 1
                                            st.rerun()
                                    with col2:
                                        st.write(f"Page {st.session_state.current_page_idx + 1} of {len(results)}")
                                    with col3:
                                        if st.button("Next", disabled=st.session_state.current_page_idx == len(results) - 1):
                                            st.session_state.current_page_idx += 1
                                            st.rerun()
                                    
                                    current_page_num = results[st.session_state.current_page_idx]
                                    st.write(f"Showing Page {current_page_num}")
                                    try:
                                        page = st.session_state.pdf_doc.load_page(current_page_num - 1)
                                        pix = page.get_pixmap()
                                        st.image(pix.tobytes(), caption=f"Page {current_page_num}", width=700)
                                    except Exception as e:
                                        st.error(f"Error displaying page {current_page_num}: {str(e)}")
                                else:
                                    st.info(f"No matches found for '{keyword}'")
                        except Exception as e:
                            st.error(f"Error during search: {str(e)}")
                    else:
                        st.warning("Please enter a keyword to search")
                else:
                    if st.session_state.search_results:
                        st.write(f"Keyword found on pages: {st.session_state.search_results}")
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            if st.button("Previous", disabled=st.session_state.current_page_idx == 0):
                                st.session_state.current_page_idx -= 1
                                st.rerun()
                        with col2:
                            st.write(f"Page {st.session_state.current_page_idx + 1} of {len(st.session_state.search_results)}")
                        with col3:
                            if st.button("Next", disabled=st.session_state.current_page_idx == len(st.session_state.search_results) - 1):
                                st.session_state.current_page_idx += 1
                                st.rerun()
                        
                        current_page_num = st.session_state.search_results[st.session_state.current_page_idx]
                        st.write(f"Showing Page {current_page_num}")
                        try:
                            page = st.session_state.pdf_doc.load_page(current_page_num - 1)
                            pix = page.get_pixmap()
                            st.image(pix.tobytes(), caption=f"Page {current_page_num}", width=700)
                        except Exception as e:
                            st.error(f"Error displaying page {current_page_num}: {str(e)}")
                
                if clear_clicked:
                    try:
                        temp_pdf_path = "./data/temp_cleared.pdf"
                        new_doc = fitz.open()
                        
                        for page_num in range(len(st.session_state.pdf_doc)):
                            new_doc.insert_pdf(st.session_state.pdf_doc, from_page=page_num, to_page=page_num)
                        
                        st.session_state.pdf_doc.close()
                        
                        new_doc.save(temp_pdf_path)
                        new_doc.close()
                        
                        if os.path.exists(temp_pdf_path):
                            os.replace(temp_pdf_path, "./data/ocr_searchable.pdf")
                        
                        st.session_state.search_results = []
                        st.session_state.pdf_doc = fitz.open("./data/ocr_searchable.pdf")
                        st.success("Search results cleared")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing search results: {str(e)}")
        
        with qa_col:
            st.subheader("Q&A")
            
            with st.spinner("Processing document for Q&A..."):
                rag_chain = get_rag_chain("./data/ocr_searchable.pdf")
            
            question = st.text_input("Enter your question:")
            if st.button("Get Answer"):
                if question:
                    with st.spinner("Generating answer..."):
                        try:
                            st.session_state.answer = rag_chain.invoke(question)
                            st.write("Answer:", st.session_state.answer)
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")
                else:
                    st.warning("Please enter a question")
            elif st.session_state.answer:
                st.write("Answer:", st.session_state.answer)