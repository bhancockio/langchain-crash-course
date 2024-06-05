from langchain.chains import create_retrieval_qa_with_history_chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Set up the chat model and message history
chat = ChatOpenAI(temperature=0.7)
history = ChatMessageHistory()

# Define the chat prompt templates
system_template = """You are a helpful AI assistant that answers questions based on the provided context. 
The user will provide you with a question, and you should respond with the most relevant information from the context.
If the question cannot be answered from the provided context, say that you do not have enough information to answer the question.
"""
human_template = "{question}"

chat_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate(system_template),
        HumanMessagePromptTemplate(human_template),
        MessagesPlaceholder(variable_name="history"),
    ]
)

# Create the retrieval QA chain with history
qa_chain = create_retrieval_qa_with_history_chain(
    llm=chat,
    retriever=retriever,
    prompt=chat_prompt,
    memory=history,
)

# Example usage
query = "What is the purpose of LangChain?"
result = qa_chain({"question": query})
print(result["result"])

# Add the query and response to the chat history
history.add_user_message(query)
history.add_ai_message(result["result"])

# Ask a follow-up question
follow_up = "Can you elaborate on the features of LangChain?"
result = qa_chain({"question": follow_up})
print(result["result"])

history.add_user_message(follow_up)
history.add_ai_message(result["result"])
