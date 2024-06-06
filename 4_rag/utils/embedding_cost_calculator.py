import os

import tiktoken

# Define the file path for the document
file_path = os.path.join(os.path.dirname(__file__), "..", "books", "odyssey.txt")

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Read the content of the file
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)  # Use the appropriate encoding for the model

# Tokenize the text and count the tokens
tokens = tokenizer.encode(text)
total_tokens = len(tokens)

# Calculate the cost based on OpenAI's pricing
cost_per_million_tokens = 0.02  # $0.02 per million tokens
cost = (total_tokens / 1_000_000) * cost_per_million_tokens

# Print the results
print(f"Total number of tokens: {total_tokens}")
print(f"Estimated cost for processing: ${cost:.6f}")
