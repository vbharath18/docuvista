# crewai_processor.py

import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool, FileWriterTool

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
        goal="""Extract tests and results as instructed. The result MUST be valid CSV.""",
        backstory="""You are a lab test results data extraction agent""",
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
    
    add_observation = Task(
        description=""" 
                Analyse CSV data and calculate the observation of each test results in
                the 'Observation' column. Add a new column to the CSV that records that sentiment.
                Your output should be in CSV format. Respond without using Markdown code fences.  
                """,
        expected_output="A correctly formatted CSV data file",
        agent=csv_agent,
        output_file="./data/final.csv",
        tools=[file_read_tool]
    )
    
    crew = Crew(
        agents=[csv_agent, csv_agent],
        tasks=[create_CSV, add_observation],
        verbose=False,
    )
    
    return crew.kickoff()
