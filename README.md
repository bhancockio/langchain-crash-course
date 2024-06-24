# LangChain Crash Course

Welcome to the LangChain Crash Course repository! This repo contains all the code examples you'll need to follow along with the LangChain Master Class for Beginners video. By the end of this course, you'll know how to use LangChain to create your own AI agents, build RAG chatbots, and automate tasks with AI.

## Course Outline

1. **Setup Environment**
2. **Chat Models**
3. **Prompt Templates**
4. **Chains**
5. **RAG (Retrieval-Augmented Generation)**
6. **Agents & Tools**

## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- Poetry (Follow this [Poetry installation tutorial](https://python-poetry.org/docs/#installation) to install Poetry on your system)

### Installation

1. Clone the repository:

   ```bash
   <!-- TODO: UPDATE TO MY  -->
   git clone https://github.com/bhancockio/langchain-crash-course
   cd langchain-crash-course
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install --no-root
   ```

3. Set up your environment variables:

   - Rename the `.env.example` file to `.env` and update the variables inside with your own values. Example:

   ```bash
   mv .env.example .env
   ```

4. Activate the Poetry shell to run the examples:

   ```bash
   poetry shell
   ```

5. Run the code examples:

   ```bash
    python 1_chat_models/1_chat_model_basic.py
   ```

## Repository Structure

Here's a breakdown of the folders and what you'll find in each:

### 1. Chat Models

- `1_chat_model_basic.py`
- `2_chat_model_basic_conversation.py`
- `3_chat_model_alternatives.py`
- `4_chat_model_conversation_with_user.py`
- `5_chat_model_save_message_history_firestore.py`

Learn how to interact with models like ChatGPT, Claude, and Gemini.

### 2. Prompt Templates

- `1_prompt_template_basic.py`
- `2_prompt_template_with_chat_model.py`

Understand the basics of prompt templates and how to use them effectively.

### 3. Chains

- `1_chains_basics.py`
- `2_chains_under_the_hood.py`
- `3_chains_extended.py`
- `4_chains_parallel.py`
- `5_chains_branching.py`

Learn how to create chains using Chat Models and Prompts to automate tasks.

### 4. RAG (Retrieval-Augmented Generation)

- `1a_rag_basics.py`
- `1b_rag_basics.py`
- `2a_rag_basics_metadata.py`
- `2b_rag_basics_metadata.py`
- `3_rag_text_splitting_deep_dive.py`
- `4_rag_embedding_deep_dive.py`
- `5_rag_retriever_deep_dive.py`
- `6_rag_one_off_question.py`
- `7_rag_conversational.py`
- `8_rag_web_scrape_firecrawl.py`
- `8_rag_web_scrape.py`

Explore the technologies like documents, embeddings, and vector stores that enable RAG queries.

### 5. Agents & Tools

- `1_agent_and_tools_basics.py`
- `agent_deep_dive/`
  - `1_agent_react_chat.py`
  - `2_react_docstore.py`
- `tools_deep_dive/`
  - `1_tool_constructor.py`
  - `2_tool_decorator.py`
  - `3_tool_base_tool.py`

Learn about agents, how they work, and how to build custom tools to enhance their capabilities.

## How to Use This Repository

1. **Watch the Video:** Start by watching the LangChain Master Class for Beginners video on YouTube at 2X speed for a high-level overview.

2. **Run the Code Examples:** Follow along with the code examples provided in this repository. Each section in the video corresponds to a folder in this repo.

3. **Join the Community:** If you get stuck or want to connect with other AI developers, join the FREE Skool community [here](https://www.skool.com/ai-developer-accelerator/about).

## Comprehensive Documentation

Each script in this repository contains detailed comments explaining the purpose and functionality of the code. This will help you understand the flow and logic behind each example.

## FAQ

**Q: What is LangChain?**  
A: LangChain is a framework designed to simplify the process of building applications that utilize language models.

**Q: How do I set up my environment?**  
A: Follow the instructions in the "Getting Started" section above. Ensure you have Python 3.10 or 3.11 installed, install Poetry, clone the repository, install dependencies, rename the `.env.example` file to `.env`, and activate the Poetry shell.

**Q: I am getting an error when running the examples. What should I do?**  
A: Ensure all dependencies are installed correctly and your environment variables are set up properly. If the issue persists, seek help in the Skool community or open an issue on GitHub.

**Q: Can I contribute to this repository?**  
A: Yes! Contributions are welcome. Please open an issue or submit a pull request with your changes.

**Q: Where can I find more information about LangChain?**  
A: Check out the official LangChain documentation and join the Skool community for additional resources and support.

## Support

If you encounter any issues or have questions, feel free to open an issue on GitHub or ask for help in the Skool community.

## License

This project is licensed under the MIT License.
