import os
from langfuse import Langfuse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Langfuse client
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-f0a7e8da-96aa-4880-b40b-c304ee9cf6bd"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-54a39df1-58bb-4fb8-8df6-352b28fa84a4"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)
