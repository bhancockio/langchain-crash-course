# Import necessary libraries
from dotenv import load_dotenv  # To load environment variables from a .env file
from langchain import hub  # To pull prompt templates from the hub
from langchain.agents import (  # To create and run agents
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool  # To define tools the agent can use
from langchain_openai import ChatOpenAI  # To use the OpenAI model

# Load environment variables from .env file
load_dotenv()


# Define a very simple tool function that returns the current time
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime  # Import datetime module to get current time

    now = datetime.datetime.now()  # Get current time
    return now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format


# List of tools available to the agent
tools = [
    Tool(
        name="Time",  # Name of the tool
        func=get_current_time,  # Function that the tool will execute
        description="Useful for when you need to know the current time",  # Description of the tool
    ),
]

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/react")

# Initialize a ChatOpenAI model
llm = ChatOpenAI(
    model="gpt-4o", temperature=0
)  # Using GPT-4 model with deterministic output

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=llm,  # Language model the agent will use to generate responses
    tools=tools,  # List of tools the agent can use
    prompt=prompt,  # Prompt template to guide the agent
    stop_sequence=True,  # Add a stop sequence to prevent hallucinations
)

# Create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,  # Agent to execute
    tools=tools,  # List of tools available to the agent
    verbose=True,  # Enable detailed logging for debugging purposes
)

# Run the agent with a test query
response = agent_executor.invoke({"input": "What time is it?"})

# Print the response from the agent
print("response:", response)
