# Example Source: https://python.langchain.com/v0.2/docs/integrations/memory/google_firestore/

from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI

"""
Steps to replicate this example:
1. Create a Firebase account
2. Create a new Firebase project
    - Copy the project ID
3. Create a Firestore database in the Firebase project
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
"""

load_dotenv()

# Setup Firebase Firestore
PROJECT_ID = "langchain-demo-abf48"
SESSION_ID = "user_session_new"  # This could be a username or a unique ID
COLLECTION_NAME = "chat_history"

# Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# Initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

# Initialize Chat Model
model = ChatOpenAI()

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")
