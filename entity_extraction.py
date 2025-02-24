import os
from typing_extensions import List, Optional, Annotated
from datetime import date
import gradio as gr
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
from vector_search import process_markdown_for_embeddings, setup_rag, semantic_search, is_vector_store_initialized

# Load environment variables
load_dotenv()

# Check required environment variables

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


if not azure_openai_api_key:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable not set")
if not azure_endpoint:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable not set")

# Initialize Azure OpenAI client
llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_openai_api_key,
    api_version="2024-02-15-preview",
    deployment_name=azure_deployment_name,  # Add your deployment name here
    logprobs=True,
    top_logprobs=1
)

# Ensure the vector store is initialized
document_splits = process_markdown_for_embeddings()
rag_chain = setup_rag(document_splits)



class Demographics(BaseModel):
    """Information about a person."""

    patient_first_name: Optional[str] = Field(
        default=None, description="First Name of the patient"
    )
   
    patient_last_name: Optional[str] = Field(
    default=None, description="Last Name of the patient"
    )

    @field_validator('patient_first_name', 'patient_last_name', mode='after')  
    @classmethod
    def validate_name(cls, value: str, info: ValidationInfo) -> str:
        if not value:
            return value
        try:
            if not is_vector_store_initialized():
                return value  # Skip validation if vector store isn't ready
            answer = semantic_search(value, k=1)
            
            print(f"Validation result for {value}: {answer}")

            if not any(value in result.page_content for result in answer):
                print(f"Warning: Could not verify {value} in the knowledge base")
            return value
        except Exception as e:
            print(f"Warning: Validation error for {value}: {str(e)}")
            return value

    patient_dob: Optional[date] = Field(
        default=None, description="Date of birth of the patient in YYYY-MM-DD format"
    )
    patient_phone: Optional[str] = Field(
        default=None, description="Phone number of the patient"
    )
    patient_address: Optional[str] = Field(
        default=None, description="Address of the patient"
    )
    patient_sex: Optional[str] = Field(
        default=None, description="Sex of the patient"
    )

class Data(BaseModel):
    """Extracted data about patient"""

    # Creates a model so that we can extract multiple entities.
    people: List[Demographics]


structured_llm = llm.with_structured_output(schema=Data)

def process_text(text_input):
    try:
        prompt_template = PromptTemplate(input_variables=["text"], template="{text}")
        prompt = prompt_template.invoke({"text": text_input})
        result = structured_llm.invoke(prompt)
        
        if not result.people:
            return "No data was extracted from the text"

        # Create lists with consistent lengths
        data_lists = []
        for person in result.people:
            person_data = {
                "First Name": person.patient_first_name or "",
                "Last Name": person.patient_last_name or "",
                "Date of Birth": person.patient_dob or None,
                "Phone": person.patient_phone or "",
                "Address": person.patient_address or "",
                "Sex": person.patient_sex or ""
            }
            data_lists.append(person_data)

        if not data_lists:
            return "No valid data extracted"

        df = pd.DataFrame(data_lists)
        return df

    except Exception as e:
        print(f"Error during processing: {str(e)}")  # For debugging
        return f"Error processing the text: {str(e)}"

# Update file reading with proper path handling
import os.path

default_text = "Please input text to extract demographics."
ocr_file_path = os.path.join(os.path.dirname(__file__), 'data', 'ocr.md')

try:
    if os.path.exists(ocr_file_path):
        with open(ocr_file_path, 'r', encoding='utf-8') as file:
            default_text = file.read()
except Exception as e:
    print(f"Warning: Could not read OCR file: {str(e)}")

# Create Gradio interface
demo = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(value=default_text, lines=10, label="Input Text"),
    outputs=gr.Dataframe(),
    title="Demographics Extractor",
    description="Extract patient demographics from medical documents",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)






