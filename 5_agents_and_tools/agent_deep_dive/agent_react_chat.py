from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool

load_dotenv()


# Define Tools
def get_current_time(*args, **kwargs):
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

# Load the Chat ReAct Prompt
prompt = hub.pull("hwchase17/react-chat")

# Initialize a ChatOpenAI model
llm = ChatOpenAI(temperature=0)

# Create the ReAct Agent with Conversation Buffer Memory
memory = ConversationBufferMemory(memory_key="chat_history")  # Initialize memory
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

# Chat Loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])
