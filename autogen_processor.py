import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

load_dotenv()

# Initialize the model client using environment variables
model_client = AzureOpenAIChatCompletionClient(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_VERSION")
)

# Define termination conditions for the chat agents
text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_mention_termination | max_messages_termination

async def main():
    # Define file paths
    input_md = Path("./data/ocr.md")
    output_csv = Path("./data/rp.csv")
    final_csv = Path("./data/final.csv")
  
   
    # Task 1: Extract CSV from Markdown
    extraction_agent = AssistantAgent(
        "ExtractionAgent",
        description="Extract tests and results as instructed. The result MUST be valid CSV.",
        model_client=model_client,
        system_message="""
            Extract tests and results from the provided markdown file.
            Ensure that string data is enclosed in quotes.
            Columns: 
                "Test type": Name of the test type is found after Patient Information,                
                "Test": Name of the test, 
                "Result": Result of the test, 
                "Unit": Unit of the test, 
                "Interval": Biological reference interval.
            Do not add extra columns. Leave columns empty if not applicable.
            Output must be valid CSV without markdown formatting.
            Do not use function calling.
            Respond without using Markdown code fences.
            """
    )
    
    team1 = MagenticOneGroupChat(
        [extraction_agent],
        model_client=model_client,
        termination_condition=termination,
        max_turns=1
    )
    
    md_content = input_md.read_text()
    result1 = await Console(team1.run_stream(task=md_content))
    csv_output = result1.messages[3].content
    output_csv.write_text(csv_output)

    # Task 2: Add Observation Column to CSV
    observation_agent = AssistantAgent(
        "ObservationAgent",
        description="Add sentiment observation column to CSV",
        model_client=model_client,
        system_message="""
            Given CSV content, add a new column named "Observation" that provides a sentiment analysis of each test result.
            Ensure output is valid CSV without markdown formatting.
            Do not use function calling.
            Respond without using Markdown code fences.
            """
    )
    
    team2 = MagenticOneGroupChat(
        [observation_agent],
        model_client=model_client,
        termination_condition=termination,
        max_turns=1
    )
    
    csv_content = output_csv.read_text()
    result2 = await Console(team2.run_stream(task=csv_content))
    final_output = result2.messages[3].content
    final_csv.write_text(final_output)
    
    print("Processing complete. CSV files saved to './data/rp.csv' and './data/final.csv'.")


async def process_with_autogen():
    await main()


if __name__ == "__main__":
    asyncio.run(main())
