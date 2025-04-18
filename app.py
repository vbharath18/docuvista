# --- Imports ---
import os
import sys
import io
import logging
import asyncio
import streamlit as st
import pandas as pd
import plotly.express as px
import fitz  # PyMuPDF
from dotenv import load_dotenv

from rag_handler import process_pdf_for_embeddings, setup_rag
from processor.azure_document_processor import process_uploaded_pdf
from processor.crewai_processor import process_with_crew
from processor.autogen_processor import process_with_autogen
from streamlit_helpers import StreamToStreamlit, render_log_to_streamlit, redirect_stdout_to_streamlit, capture_stdout

# --- Environment & Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Streamlit Layout ---
st.set_page_config(layout="wide")

# --- Helper Functions ---
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        logging.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        st.error(f"Data file not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def load_pdf(file_path: str) -> fitz.Document | None:
    """Loads a PDF file."""
    try:
        doc = fitz.open(file_path)
        logging.info(f"Loaded PDF: {file_path}")
        return doc
    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        st.error(f"Error loading PDF: {e}")
        return None

def search_pdf(doc: fitz.Document, keyword: str) -> list[int]:
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
    logging.info(f"Keyword '{keyword}' found on pages: {results}")
    return results

@st.cache_resource
def get_rag_chain(file_path: str):
    """Cache the RAG chain setup to avoid reprocessing."""
    document_splits = process_pdf_for_embeddings(file_path)
    return setup_rag(document_splits)

def reload_data():
    """Reload all data sources."""
    df = load_data("./data/final.csv")
    pdf_doc = load_pdf("./data/ocr_searchable.pdf")
    if pdf_doc is not None:
        st.session_state.pdf_doc = pdf_doc
    return df, pdf_doc

def check_required_files() -> bool:
    """Check if required files exist."""
    files_exist = os.path.exists("./data/final.csv") and os.path.exists("./data/ocr_searchable.pdf")
    if not files_exist:
        logging.warning("Required files are missing.")
    return files_exist

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate if DataFrame has required columns for visualization."""
    required_columns = ["Test", "Test type", "Observation"]
    viz_columns = ["Test type", "Test", "Result", "Unit", "Interval", "Observation"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info(f"Current columns in your data: {', '.join(df.columns.tolist())}")
        return False
    missing_viz_columns = [col for col in viz_columns if col not in df.columns]
    if missing_viz_columns:
        st.warning(f"Some visualization columns are missing: {', '.join(missing_viz_columns)}")
    df.columns = df.columns.map(str).str.strip()
    if df.empty:
        st.error("DataFrame is empty")
        return False
    return True

# --- Session State Initialization ---
for key, default in {
    'needs_reload': True,
    'files_ready': check_required_files(),
    'pdf_doc': None,
    'search_results': [],
    'current_page_idx': 0,
    'df': pd.DataFrame(),
    'answer': ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Tabs ---
tabs = st.tabs(["Upload", "Report", "Triage"])

# --- Upload Tab ---
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
                    logging.error(f"Error processing document: {e}")
                    st.error(f"Error processing document: {str(e)}")

# --- Report Tab ---
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
            df.columns = df.columns.map(str).str.strip()
            if not validate_dataframe(df):
                st.warning("Cannot create visualizations. Data format is incorrect or missing required columns.")
            else:
                with st.spinner("Loading charts..."):
                    try:
                        # Try to convert 'Result' to numeric for meaningful plots
                        df_numeric = df.copy()
                        df_numeric["Result_numeric"] = pd.to_numeric(df_numeric["Result"], errors="coerce")
                        has_numeric = df_numeric["Result_numeric"].notnull().any()

                        # Pie chart: Distribution of Test type
                        fig_pie = px.pie(
                            df,
                            names="Test type",
                            title="Distribution of Test Types",
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig_pie.update_traces(textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

                        if has_numeric:
                            # Grouped bar: Average Result by Test and Test type
                            fig_grouped = px.bar(
                                df_numeric.dropna(subset=["Result_numeric"]),
                                x="Test",
                                y="Result_numeric",
                                color="Test type",
                                barmode="group",
                                title="Average Result by Test and Test Type",
                                labels={"Result_numeric": "Average Result"},
                                height=350
                            )
                            fig_grouped.update_layout(margin=dict(t=30, b=40, l=20, r=20), xaxis_tickangle=-45)
                            st.plotly_chart(fig_grouped, use_container_width=True, config={'displayModeBar': False})

                            # Box plot: Result distribution by Test type
                            fig_box = px.box(
                                df_numeric.dropna(subset=["Result_numeric"]),
                                x="Test type",
                                y="Result_numeric",
                                points="all",
                                color="Test type",
                                title="Result Distribution by Test Type",
                                labels={"Result_numeric": "Result"},
                                height=350
                            )
                            fig_box.update_layout(margin=dict(t=30, b=40, l=20, r=20))
                            st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})
                        else:
                            st.info("'Result' column is not numeric. Showing only categorical insights.")

                        # Bar chart: Count of Observations by Test type
                        fig_obs = px.bar(
                            df,
                            x="Test type",
                            color="Observation",
                            title="Observation Counts by Test Type",
                            labels={"count": "Count"},
                            height=300
                        )
                        fig_obs.update_layout(margin=dict(t=30, b=40, l=20, r=20))
                        st.plotly_chart(fig_obs, use_container_width=True, config={'displayModeBar': False})
                    except Exception as e:
                        logging.error(f"Error creating visualizations: {e}")
                        st.error(f"Error creating visualizations: {str(e)}")

# --- Triage Tab ---
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
                            logging.error(f"Error during search: {e}")
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
                        logging.error(f"Error clearing search results: {e}")
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
                            logging.error(f"Error generating answer: {e}")
                            st.error(f"Error generating answer: {str(e)}")
                else:
                    st.warning("Please enter a question")
            elif st.session_state.answer:
                st.write("Answer:", st.session_state.answer)