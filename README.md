# Conversational Research Agent with Dynamic Knowledge Base

## Overview

This project implements a conversational AI research assistant that can interact with a local collection of research papers. Users can ask questions about the papers, and the agent uses a Retrieval Augmented Generation (RAG) approach with an Anthropic Claude model to provide answers grounded in the document content.

A key feature is the agent's ability to dynamically expand its knowledge base: users can instruct the agent to search for new papers on arXiv, download them, process them, and integrate them into its vector store for immediate querying. The agent also maintains conversational context, allowing for follow-up questions and a more natural interaction flow.

## Features

* **Conversational Interface:** Interacts with users in a chat-like manner, remembering the context of the current conversation.
* **Retrieval Augmented Generation (RAG):** Answers questions based on content retrieved from a local collection of PDF research papers.
* **Dynamic Knowledge Base Expansion:**
    * Searches for papers on arXiv based on user queries (titles, keywords, arXiv IDs).
    * Allows user selection if multiple papers are found.
    * Downloads selected PDFs into a local directory.
    * Processes new PDFs (text extraction, chunking, embedding).
    * Updates the vector store (FAISS index and text chunk metadata) in real-time.
* **Persistent Storage:** Downloaded papers and the vector store are saved locally for use across sessions.
* **Modular Design:** Code is organized into separate modules for paper management, vector store operations, and the main agent logic.

## Tech Stack & Key Libraries

* **Python 3.x**
* **Anthropic API:** For Large Language Model (Claude) capabilities.
* **Sentence-Transformers:** For generating text embeddings.
* **FAISS (cpu):** For efficient similarity search in the vector store.
* **PyPDF:** For extracting text from PDF documents.
* **arXiv:** Python library for searching and retrieving metadata from arXiv.
* **NLTK:** For sentence tokenization.
* **dotenv:** For managing API keys.
* **NumPy:** For numerical operations.

## Setup and Installation

1.  **Clone the Repository (or set up project directory):**
    If this were a Git repository:
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```
    Otherwise, ensure all the Python files (`research_agent.py`, `paper_manager.py`, etc.) and `initial_papers.json` are in your main project directory.

2.  **Set up Python Virtual Environment:**
    Make the setup script executable and run it:
    ```bash
    chmod +x setup_research_env.sh
    ./setup_research_env.sh
    ```
    This will create a virtual environment named `.research-agent-env` (or as defined in the script) and a `requirements.txt` file.

3.  **Activate the Virtual Environment:**
    ```bash
    source .research-agent-env/bin/activate
    ```
    (On Windows, it might be `.research-agent-env\Scripts\activate`)

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up Environment Variables:**
    Create a `.env` file in the project root (you can copy `.env.example` if provided, or create it manually):
    ```
    ANTHROPIC_API_KEY="your_anthropic_api_key_here"
    ```
    Replace `"your_anthropic_api_key_here"` with your actual Anthropic API key.

6.  **Download NLTK 'punkt' Tokenizer Data:**
    The scripts attempt to download this automatically if missing. If you encounter issues, you can do it manually after activating the environment:
    ```bash
    python -c "import nltk; nltk.download('punkt')"
    ```

## Running the Application

Follow these steps in order:

1.  **Step 1: Initial Paper Download (Optional, but Recommended for a starting knowledge base)**
    This script uses `initial_papers.json` to download a predefined set of papers.
    ```bash
    python bulk_download_papers.py
    ```
    This will populate the `research_papers/` directory.

2.  **Step 2: Initial Vector Store Build**
    This script processes all PDFs in the `research_papers/` directory and creates the FAISS index and text chunk metadata.
    ```bash
    python vector_store_manager.py
    ```
    This will create and populate the `vector_store/` directory.

3.  **Step 3: Run the Research Agent**
    Start the interactive agent:
    ```bash
    python research_agent.py
    ```

## Usage

Once the agent is running, you can interact with it via the command line:

* **Asking Questions:** Simply type your question about the content of the ingested research papers and press Enter.
    Example: `What is the main idea behind the Transformer architecture?`

* **Adding New Papers:** Use the `/add_paper` command followed by a query (paper title, keywords, or arXiv ID).
    Example: `/add_paper Retentive Network A Successor to Transformer`
    Example: `/add_paper 2305.10601`
    The agent will search arXiv, may prompt you to select from multiple results, then download, process, and add the paper to its knowledge base.

* **Clearing Conversation History:** To reset the current chat context (but not the knowledge base):
    ```
    /clear_history
    ```

* **Exiting the Agent:**
    ```
    /quit
    ```
    or
    ```
    /exit
    ```

## Configuration Notes

* **LLM Model:** The Anthropic model used can be changed in `research_agent.py` (e.g., `LLM_MODEL_NAME = "claude-3-sonnet-20240229"`). `claude-3-haiku` is faster for conversation.
* **Embedding Model:** The sentence transformer model is set in `vector_store_manager.py` and `research_agent.py` (`EMBEDDING_MODEL_NAME`). Ensure it's consistent.
* **Top-K Retrieval:** The number of chunks retrieved for RAG context can be adjusted via `TOP_K_RESULTS` in `research_agent.py`.
* **Chat History Length:** `MAX_CHAT_HISTORY_MESSAGES` in `research_agent.py` controls how many past messages are kept in the conversational context.

## Potential Future Enhancements

* **Advanced Chat Memory:** Implement summarization for longer chat histories.
* **Support for More Document Types:** Extend beyond PDFs (e.g., .txt, .md, .docx).
* **Graphical User Interface (GUI):** Using Streamlit or Gradio.
* **Improved RAG:** Techniques like re-ranking retrieved chunks, query expansion, or hybrid search.
* **Citation Analysis:** Functionality to analyze and answer questions about citation patterns within the local corpus (a more complex feature).
* **Error Handling:** More comprehensive error handling for file operations, API calls, and library interactions.