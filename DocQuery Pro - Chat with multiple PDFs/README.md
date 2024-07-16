# DocQuery Pro - Chat with multiple PDFs

DocQuery Pro is a Python application designed to facilitate querying and interaction with multiple PDF documents using advanced natural language processing techniques.

## Description

This project integrates document processing, embedding generation, and a querying interface to provide a seamless experience for extracting information from PDF files. It also supports maintaining long-term history of chats, allowing users to continue conversations from previous sessions using the same session ID.
## Files

### 1. `get_embedding_function.py`

This script defines a function to retrieve embeddings using the Ollama embeddings model (`nomic-embed-text`).

### 2. `populate_database.py`

This script manages the population and maintenance of a database (`Chroma`) with documents extracted from PDFs. It includes functionality to reset the database and add new documents.

**Execution:**
- To reset the database entirely:
  ```bash
  python populate_database.py --reset
- To add to the existing database:
  ```bash
  python populate_database.py

### 3. `QnA_app.py`

This script provides a Streamlit-based interface for querying the database populated by `populate_database.py` using a chat-like interaction approach.

**Execution:**
- Ensure dependencies are installed (`langchain`, `streamlit`, `langchain-community`, `langchain-text-splitters`, `langchain-groq`).
- Run the app:
  ```bash
  streamlit run QnA_app.py

## Models Used

- **Embeddings Model:** Ollama embeddings (`nomic-embed-text`).
- Requires Ollama server running locally for embedding generation.

- **ChatGroq Model:** LLaMA (Large Language Model Meta-Learning Architecture) `llama3-8b-8192`.
- Used for generating responses based on queried content.

## Setup Instructions

1. **Dependencies:**
 - Python 3.x
 - `langchain` library (install via `pip install langchain`)
 - `streamlit` library (install via `pip install streamlit`)
 - `langchain_community` (specifically for `OllamaEmbeddings`, install via `pip install langchain-community`)
 - `langchain_text_splitters` (install via `pip install langchain-text-splitters`)
 - `langchain_groq` (install via `pip install langchain-groq`)

2. Start the Ollama server locally to use the embeddings model.

3. Ensure the following directories exist:
 - `Database`: for storing the database files.
 - `Data`: for storing the input PDF documents.

## Usage

1. Populate the database using `populate_database.py`.
2. Run `QnA_app.py` to interact with the queried database via Streamlit interface.

---

Feel free to customize further based on additional features or requirements of your project.


