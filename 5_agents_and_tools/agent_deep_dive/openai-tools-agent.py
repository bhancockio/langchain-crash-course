import os

import pandas as pd
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function

# Load environment variables from .env file
load_dotenv()


# Define Tools
def get_current_time():
    """Returns the current time in H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    from wikipedia import summary

    try:
        return summary(query, sentences=2)  # Limit to two sentences for brevity
    except:
        return "I couldn't find any information on that."


tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic.",
    ),
]

# Load the OpenAI Tools Agent Prompt
prompt = hub.pull("hwchase17/openai-tools-agent")

# Initialize a ChatOpenAI model
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")

# Create the OpenAI Tools Agent
agent = create_openai_tools_agent(
    llm=llm,
    tools=format_tool_to_openai_function(tools),
    prompt=prompt,
    verbose=True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

# Test Queries and Display Responses
queries = [
    "What time is it?",
    "Who is the president of the United States?",
    "What is the capital of France?",
]

for query in queries:
    response = agent_executor.invoke({"input": query})
    print(f"\nQuery: {query}")
    print("Response:", response["output"])
