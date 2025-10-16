from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Setup messages
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

# ---- Local LLM Example using LM Studio with ChatOpenAI ----

# The base_url points to the LM Studio server.
# The 'model' parameter can be set to any string, as LM Studio ignores it.
model = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio") # type: ignore

# Invoke the model with the initial set of messages
result = model.invoke(messages)
print(f"Answer from Local LLM: {result.content}")

# Add the AI's response to the message history for a new query
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]

# Invoke the model again with the updated message history
result = model.invoke(messages)
print(f"Answer from Local LLM: {result.content}")