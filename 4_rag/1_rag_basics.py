import os

import tiktoken
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Step 1: Load the Document
# Construct the file path dynamically to ensure it works correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "odyssey.txt")

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Read the text content from your file
loader = TextLoader(file_path)
documents = loader.load()

# Step 2: Split the Document into Chunks
# Make the text easier to manage by splitting it into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")


# Step 3: Create Embeddings
# Turn text into numbers so the computer can understand meaning
embeddings = OpenAIEmbeddings()  # Uses OpenAI's embedding model (requires your API key)

# Step 4: Create Vector Store
# A database to store the embeddings for efficient searching
db = Chroma.from_documents(
    docs, embeddings, persist_directory="chroma_db"
)  # Embed each chunk and store in the Chroma database

# Step 5: User's Question
# What the user wants to know
query = "Who is Odysseus' wife?"

# Step 6: Retrieve Relevant Documents
# Find the most relevant parts of the text based on the query
# TODO: In video do deep dive into VectorStoreRetriever in vectorestores.py
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.7},
)
relevant_docs = retriever.invoke(query)  # Get the top 3 most relevant documents

# Display the Relevant Results
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Step 7: Optional - Show similarity scores for each document
# Uncomment the following lines if you want to display similarity scores
# results = db.similarity_search_with_relevance_scores(query, k=3)
# if len(results) == 0 or results[0][1] < 0.7:
#     print(f"Unable to find matching results.")
# else:
#     for doc, score in results:
#         print(f"Score: {score:.4f}")
#         print(doc.page_content)
#         print("\n")
