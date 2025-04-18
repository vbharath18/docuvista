# crewai_processor.py

import os
from pathlib import Path
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool, FileWriterTool

os.environ['OTEL_SDK_DISABLED'] = 'true'

# Validate required environment variables for Azure OpenAI
REQUIRED_ENV_VARS = [
    "AZURE_OPENAI_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY"
]
for var in REQUIRED_ENV_VARS:
    if not os.environ.get(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")


def process_with_crew():
    """Process the markdown file with CrewAI using best practices."""
    # Initialize the LLM with validated environment variables
    llm = LLM(
        api_version=os.environ["AZURE_OPENAI_VERSION"],
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"]
    )

    # Use pathlib for file paths
    data_dir = Path("./data")
    ocr_md_path = data_dir / "ocr.md"
    rp_csv_path = data_dir / "rp.csv"
    final_csv_path = data_dir / "final.csv"

    # Initialize tools
    file_read_tool = FileReadTool()
    file_writer_tool = FileWriterTool()

    # Agent for CSV extraction
    extraction_agent = Agent(
        role="Lab Test Data Extractor",
        goal="Extract tests and results as valid CSV.",
        backstory="You are a lab test results data extraction agent.",
        tools=[file_read_tool, file_writer_tool],
        llm=llm,
        name="ExtractionAgent"
    )

    # Agent for adding observations
    observation_agent = Agent(
        role="Lab Test Observation Annotator",
        goal="Add sentiment analysis observations to each test result in the CSV.",
        backstory="You are a medical data annotator specializing in sentiment analysis.",
        tools=[file_read_tool, file_writer_tool],
        llm=llm,
        name="ObservationAgent"
    )

    # Task: Extract CSV from markdown
    extract_csv_task = Task(
        description=(
            f"""
            Analyse '{ocr_md_path}' (Markdown format). Output CSV only (no Markdown code fences).
            - Enclose string data in quotes.
            - Columns: 'Test type', 'Test', 'Result', 'Unit', 'Interval'.
            - Use pydantic schema validation.
            - Leave non-applicable columns empty.
            """
        ),
        expected_output="A correctly formatted CSV data structure only.",
        agent=extraction_agent,
        output_file=str(rp_csv_path),
        tools=[file_read_tool, file_writer_tool],
        max_retries=1,
        name="ExtractCSVTask"
    )

    # Task: Add observation column
    add_observation_task = Task(
        description=(
            f"""
            Analyse CSV data and perform sentiment analysis for each test result, adding an 'Observation' column.
            Output CSV only (no Markdown code fences). Use pydantic schema validation.
            """
        ),
        expected_output="A correctly formatted CSV data file.",
        agent=observation_agent,
        output_file=str(final_csv_path),
        tools=[file_read_tool, file_writer_tool],
        max_retries=1,
        name="AddObservationTask"
    )

    # Create and run the crew
    crew = Crew(
        agents=[extraction_agent, observation_agent],
        tasks=[extract_csv_task, add_observation_task],
        verbose=True,
    )
    return crew.kickoff()
