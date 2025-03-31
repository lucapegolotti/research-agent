# research_agent.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv
import nltk # For NLTK check
import time # For potential delays if needed

# Custom modules
import paper_manager
import vector_store_manager

# --- Configuration ---
# These should ideally match or be sourced consistently with vector_store_manager.py
VECTOR_STORE_DIR = "vector_store" # Or vector_store_manager.VECTOR_STORE_DIR
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "research_papers.index") # Or vector_store_manager.FAISS_INDEX_PATH
TEXT_CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "text_chunks.json") # Or vector_store_manager.TEXT_CHUNKS_PATH

# Embedding model (must be the same as used in vector_store_manager.py)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Or vector_store_manager.EMBEDDING_MODEL_NAME

# Anthropic API Configuration
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL_NAME = "claude-3-5-haiku-20241022"
MAX_TOKENS_TO_SAMPLE = 2000

# RAG Configuration
TOP_K_RESULTS = 5

# --- Global Variables (to be loaded once) ---
sentence_model: SentenceTransformer = None
faiss_index: faiss.Index = None
text_chunks_data: list = []
anthropic_client: Anthropic = None

# --- Initialization Functions ---
def load_resources():
    """Loads the sentence model, FAISS index, text chunks, and initializes Anthropic client."""
    global sentence_model, faiss_index, text_chunks_data, anthropic_client

    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not found. Please set it in your .env file.")
        exit(1)
    
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    print("Anthropic client initialized.")

    print("Loading RAG resources...")
    try:
        # Check NLTK 'punkt' for sentence tokenization (used in vector_store_manager)
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            print("NLTK 'punkt' tokenizer not found. Attempting to download...")
            nltk.download('punkt', quiet=True)
            print("NLTK 'punkt' downloaded.")
        except Exception as e:
            print(f"NLTK 'punkt' check/download issue: {e}. Ensure it's available.")

        # Load Sentence Transformer model
        print(f"  Loading sentence transformer model: {EMBEDDING_MODEL_NAME}...")
        sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("  Sentence model loaded.")

        # Load FAISS index
        if not os.path.exists(FAISS_INDEX_PATH):
            print(f"Error: FAISS index not found at {FAISS_INDEX_PATH}.")
            print("Please run `python vector_store_manager.py` to build the initial store.")
            exit(1)
        print(f"  Loading FAISS index from: {FAISS_INDEX_PATH}...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"  FAISS index loaded. Total vectors: {faiss_index.ntotal}")

        # Load text chunks
        if not os.path.exists(TEXT_CHUNKS_PATH):
            print(f"Error: Text chunks not found at {TEXT_CHUNKS_PATH}.")
            print("Please run `python vector_store_manager.py` to build the initial store.")
            exit(1)
        print(f"  Loading text chunks from: {TEXT_CHUNKS_PATH}...")
        with open(TEXT_CHUNKS_PATH, 'r', encoding='utf-8') as f:
            text_chunks_data = json.load(f)
        print(f"  Text chunks loaded. Total chunks: {len(text_chunks_data)}")

        if faiss_index.ntotal != len(text_chunks_data) and faiss_index.ntotal > 0 : # Allow 0 if store is empty initially
             print(f"Warning: Mismatch between FAISS index size ({faiss_index.ntotal}) and text_chunks count ({len(text_chunks_data)}).")
             print("This might indicate an issue with the vector store. Consider rebuilding.")


    except Exception as e:
        print(f"Error loading resources: {e}")
        exit(1)
    print("All resources loaded successfully.\n")

# --- RAG Core Functions ---
def retrieve_relevant_chunks(query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """Retrieves the top_k most relevant text chunks for a given query."""
    if not sentence_model or not faiss_index or not text_chunks_data:
        print("Error: RAG Resources not loaded properly.")
        return []

    # print(f"\nEmbedding query: \"{query[:50]}...\"") # Less verbose
    query_embedding = sentence_model.encode([query], convert_to_numpy=True)
    
    # print(f"Searching FAISS index for top {top_k} results...") # Less verbose
    distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k)
    
    retrieved_chunks_info = []
    if indices.size > 0: # Check if any results were found
        for i in range(len(indices[0])):
            chunk_index_in_faiss = indices[0][i]
            if 0 <= chunk_index_in_faiss < len(text_chunks_data):
                retrieved_chunk = text_chunks_data[chunk_index_in_faiss]
                retrieved_chunks_info.append({
                    'text': retrieved_chunk['text'],
                    'source_pdf': retrieved_chunk['source_pdf'],
                    'chunk_id_in_pdf': retrieved_chunk.get('chunk_id_in_pdf', -1), # Handle if key missing
                    'global_chunk_id': retrieved_chunk['global_chunk_id'],
                    'distance': float(distances[0][i])
                })
            else:
                print(f"Warning: Retrieved index {chunk_index_in_faiss} is out of bounds for text_chunks_data.")
            
    # print(f"Retrieved {len(retrieved_chunks_info)} chunks.") # Less verbose
    return retrieved_chunks_info

def ask_claude_with_context(query: str, context_chunks: list[dict]) -> str:
    """Constructs a prompt with context and asks Claude to answer the query."""
    global anthropic_client
    if not context_chunks:
        return "I couldn't find any relevant information in the provided documents to answer your question. Try rephrasing or adding more specific terms."

    context_str = "\n\n---\n\n".join([
        f"Source Document: {chunk['source_pdf']} (Internal Chunk ID: {chunk['global_chunk_id']})\nContent: {chunk['text']}"
        for chunk in context_chunks
    ])

    prompt = f"""You are a helpful research assistant. Your task is to answer the user's question based *only* on the following provided context from research papers.
If the context does not contain sufficient information to answer the question, clearly state that you cannot answer based on the provided information.
Do not use any external knowledge or make assumptions beyond what is in the text. Be concise and directly address the question.

Provided Context:
{context_str}

User's Question: {query}

Answer:"""

    # print("\nSending prompt to Claude...") # Less verbose
    try:
        response = anthropic_client.messages.create(
            model=LLM_MODEL_NAME,
            max_tokens=MAX_TOKENS_TO_SAMPLE,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.content[0].text
        # print("Claude's response received.") # Less verbose
        return answer
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        return "Sorry, I encountered an error while trying to communicate with the language model."

# --- Function to Add New Paper ---
def add_new_paper_to_store(paper_query: str):
    """Finds, downloads, processes, and adds a new paper to the vector store."""
    global faiss_index, text_chunks_data, sentence_model # We'll modify these globals

    print(f"\nAttempting to find and add paper related to: '{paper_query}'...")
    
    # 1. Download the paper using paper_manager
    # search_arxiv_and_prompt_download will handle user interaction for selection
    downloaded_pdf_path = paper_manager.search_arxiv_and_prompt_download(
        query=paper_query,
        download_dir=paper_manager.PAPERS_DIR # Use PAPERS_DIR from paper_manager
    )

    if not downloaded_pdf_path:
        print("Paper download cancelled or failed.")
        return

    filename = os.path.basename(downloaded_pdf_path)
    # Basic check if paper (by filename) might already be processed.
    # A more robust check could use arXiv IDs if stored in text_chunks_data.
    if any(chunk['source_pdf'] == filename for chunk in text_chunks_data):
        print(f"Paper '{filename}' appears to have already been processed and is in the vector store. Skipping re-addition.")
        # Optionally, you might want to remove the just-downloaded duplicate if you're sure.
        # os.remove(downloaded_pdf_path) # Be cautious with this
        return

    # 2. Process the new PDF using vector_store_manager
    print(f"Processing '{filename}' to add to vector store...")
    # The next global_chunk_id will be the current length of text_chunks_data
    start_global_id = len(text_chunks_data)
    
    new_embeddings_np, new_chunks_metadata = vector_store_manager.process_single_pdf(
        pdf_path=downloaded_pdf_path,
        sentence_model=sentence_model, # The globally loaded sentence model
        start_global_chunk_id=start_global_id
    )

    if new_embeddings_np is None or not new_chunks_metadata:
        print(f"Failed to process the new paper: {filename}. No new data to add.")
        return

    # 3. Update in-memory FAISS index and text_chunks_data
    print("Updating in-memory vector store with new paper...")
    faiss_index.add(new_embeddings_np) # FAISS expects float32, process_single_pdf should return this
    text_chunks_data.extend(new_chunks_metadata)
    print(f"  Added {len(new_chunks_metadata)} new chunks from '{filename}'.")
    print(f"  FAISS index size now: {faiss_index.ntotal} vectors.")
    print(f"  Total text_chunks entries: {len(text_chunks_data)}.")

    # 4. Save updated index and chunks to disk for persistence
    print("Saving updated vector store to disk...")
    try:
        faiss.write_index(faiss_index, FAISS_INDEX_PATH) # Use path constants
        with open(TEXT_CHUNKS_PATH, 'w', encoding='utf-8') as f:
            json.dump(text_chunks_data, f, indent=4)
        print("Vector store updated and saved successfully.")
    except Exception as e:
        print(f"CRITICAL Error saving updated vector store: {e}")
        print("The in-memory store is updated, but changes are NOT persistent on disk for this session's additions!")
        
    print(f"Paper '{filename}' successfully added to the knowledge base.")

# --- Main Interaction Loop ---
def main():
    load_resources() # Load models and data once
    
    print("\nResearch Agent Assistant Initialized.")
    print("===================================")
    print("Commands:")
    print("  /add_paper <title, keywords, or arXiv ID>  - Find and add a new paper.")
    print("  /quit or /exit                             - End the session.")
    print("Ask me anything about the content of your research papers!")
    print("===================================")


    while True:
        user_input = input("\nYour input: ").strip()
        
        if not user_input:
            continue

        if user_input.lower() in ['/quit', '/exit']:
            print("Exiting Research Agent Assistant. Goodbye!")
            break
        
        if user_input.lower().startswith("/add_paper "):
            paper_query = user_input[len("/add_paper "):].strip()
            if paper_query:
                add_new_paper_to_store(paper_query)
            else:
                print("Please provide a paper title, keywords, or arXiv ID after /add_paper.")
            continue

        # Regular query processing
        print(f"Processing your query: \"{user_input[:60]}...\"")
        retrieved_chunks = retrieve_relevant_chunks(user_input)

        if not retrieved_chunks:
            print("\n--- Answer ---")
            print("I could not find any directly relevant information in the current documents for your query.")
            print("--------------")
            continue
        
        # Optional: Display retrieved chunk sources for transparency
        # print("\n--- Relevant Context Sources ---")
        # sources_summary = set()
        # for chunk_info in retrieved_chunks:
        #     sources_summary.add(chunk_info['source_pdf'])
        # for i, src in enumerate(list(sources_summary)[:3]): # Show top 3 unique sources
        #     print(f"  - {src}")
        # if len(sources_summary) > 3:
        #     print(f"  ...and {len(sources_summary)-3} other source(s).")
        # print("-----------------------------")

        answer = ask_claude_with_context(user_input, retrieved_chunks)
        
        print("\n--- Answer ---")
        print(answer)
        print("--------------")

if __name__ == "__main__":
    main()