from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableableMap, RunnableBranch
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define prompt templates for different feedback types
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}.",
        ),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)


# Define the branch conditions based on feedback sentiment
def is_positive(feedback):
    return "good" in feedback.lower() or "excellent" in feedback.lower()


def is_negative(feedback):
    return "bad" in feedback.lower() or "poor" in feedback.lower()


def is_neutral(feedback):
    return "okay" in feedback.lower() or "neutral" in feedback.lower()


# Define the runnable branches for handling feedback
branches = RunnableBranch(
    (lambda x: is_positive(x), positive_feedback_template | model | StrOutputParser()),
    (lambda x: is_negative(x), negative_feedback_template | model | StrOutputParser()),
    (lambda x: is_neutral(x), neutral_feedback_template | model | StrOutputParser()),
    escalate_feedback_template | model | StrOutputParser(),
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = branches

# Run the chain with an example review
# Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

review = (
    "The product is terrible. It broke after just one use and the quality is very poor."
)
result = chain.invoke(review)

# Output the result
print(result)
