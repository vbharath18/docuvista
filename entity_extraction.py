import os
from typing_extensions import List, Optional, Annotated
from datetime import date
import gradio as gr
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
from vector_search import process_markdown_for_embeddings, setup_rag

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
)

def get_rag_chain(file_path: str):
    """Cache the RAG chain setup to avoid reprocessing"""
    document_splits = process_markdown_for_embeddings()
    return setup_rag(document_splits)


rag_chain = get_rag_chain("./data/ocr.md")  


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
    def citation_exists(cls, value: str) -> str:
        answer = rag_chain.invoke(value)
        print("BBBB"+ answer)
        if value not in answer:
            raise ValueError(f'{value} is not correct')
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

        # Convert the results to a pandas DataFrame
        data = {
            "First Name": [person.patient_first_name for person in result.people],
            "Last Name": [person.patient_last_name for person in result.people],
            "Date of Birth": [person.patient_dob for person in result.people],
            "Phone": [person.patient_phone for person in result.people],
            "Address": [person.patient_address for person in result.people],
            "Sex": [person.patient_sex for person in result.people],
        }
        return pd.DataFrame(data)
    except Exception as e:
        return f"Error processing the text: {str(e)}"

# Read default content from OCR markdown file
with open('./data/ocr.md', 'r', encoding='utf-8') as file:
    default_text = file.read()

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






