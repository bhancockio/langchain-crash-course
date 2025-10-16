import os
import warnings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # List all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    print(f"Found {len(book_files)} book files: {book_files}")

    # Read the text content from each file and store it with metadata
    documents = []
    for i, book_file in enumerate(book_files, 1):
        print(f"Processing book {i}/{len(book_files)}: {book_file}")
        file_path = os.path.join(books_dir, book_file)
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            book_docs = loader.load()
        except UnicodeDecodeError:
            try:
                loader = TextLoader(file_path, encoding='latin-1')
                book_docs = loader.load()
            except UnicodeDecodeError:
                loader = TextLoader(file_path, encoding='utf-8', errors='ignore')
                book_docs = loader.load()
        
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)
        print(f"Loaded {len(book_docs)} documents from {book_file}")

    # Split the documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print(f"\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk preview: {docs[0].page_content[:100]}...")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    print("--- Finished creating embeddings ---")

    # Create the vector store and persist it (with progress)
    print("\n--- Creating and persisting vector store ---")
    print(f"Processing {len(docs)} document chunks...")
    
    # Process in smaller batches to avoid memory issues
    batch_size = 100
    if len(docs) > batch_size:
        print(f"Processing in batches of {batch_size}...")
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1}")
            
            if i == 0:
                # Create initial database
                db = Chroma.from_documents(
                    batch, embeddings, persist_directory=persistent_directory)
            else:
                # Add to existing database
                db.add_documents(batch)
    else:
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
    
    print("--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
