import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

load_dotenv()

# 1. Load and Prepare Your Document
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books/odyssey.txt")

loader = TextLoader(file_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 2. Create Chroma Vector Store (In-Memory for Demo)
db = Chroma.from_documents(docs, OpenAIEmbeddings())

# 3. Set Up ReAct Agent with Document Store Retriever
# Load the ReAct Docstore Prompt
react_docstore_prompt = hub.pull("hwchase17/react-docstore")
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")

# Chain for question answering
chain = load_qa_chain(llm, chain_type="stuff")

# Create the ReAct Agent with document store retriever
agent = create_react_agent(
    llm=llm,
    tools=[
        Tool(
            name="Answer Question",
            func=chain.run,
            description="useful for when you need to answer questions about the context",
        )
    ],
    prompt=react_docstore_prompt,
    verbose=True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

# 4. Have a Conversation!
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke({"input": query})
    print(f"AI: {response['output']}")
