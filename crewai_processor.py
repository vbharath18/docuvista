# crewai_processor.py

import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool, FileWriterTool

os.environ['OTEL_SDK_DISABLED'] = 'true'

def process_with_crew():
    """Process the markdown file with CrewAI"""
    
    # Initialize the LLM with environment variables for API configuration
    llm = LLM(
        api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        base_url=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY")
    )
    
    # Initialize file read and write tools
    file_read_tool = FileReadTool()
    file_writer_tool = FileWriterTool()
    
    # Define an agent for extracting and processing data
    csv_agent = Agent(
        role="Extract, process data and record data",
        goal="""Extract tests and results as instructed. The result MUST be valid CSV.""",
        backstory="""You are a lab test results data extraction agent""",
        tools=[file_read_tool],
        llm=llm,
    )
    
    # Define a task to create a CSV from the markdown file
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
        tools=[file_read_tool],
        max_retries=1
    )
    
    # Define a task to add an observation column to the CSV
    add_observation = Task(
        description=""" 
                Analyse CSV data and perform sentiment analysis of each test result and add it 
                to the 'Observation' column. Your output should be in CSV format. Respond without using Markdown code fences.  
                """,
        expected_output="A correctly formatted CSV data file",
        agent=csv_agent,
        output_file="./data/final.csv",
        tools=[file_read_tool],
        max_retries=1
    )
    
    # Create a crew with the defined agents and tasks
    crew = Crew(
        agents=[csv_agent, csv_agent],
        tasks=[create_CSV, add_observation],
        verbose=True,
    )

    # Start the processing with the crew
    return crew.kickoff()
