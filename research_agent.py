import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

# --- Configuration ---
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "research_papers.index")
TEXT_CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "text_chunks.json")

# Embedding model (must be the same as used in build_vector_store.py)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Anthropic API Configuration
# Load environment variables from .env file (for ANTHROPIC_API_KEY)
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL_NAME = "claude-3-5-haiku-20241022"
MAX_TOKENS_TO_SAMPLE = 2000 # Max tokens for Claude's response

# RAG Configuration
TOP_K_RESULTS = 5  # Number of relevant chunks to retrieve

# --- Global Variables (to be loaded once) ---
sentence_model = None
faiss_index = None
text_chunks_data = []

# --- Initialization Functions ---
def load_resources():
    """Loads the sentence model, FAISS index, and text chunks."""
    global sentence_model, faiss_index, text_chunks_data

    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not found. Please set it in your .env file.")
        exit()

    print("Loading resources...")
    try:
        # Load Sentence Transformer model
        print(f"  Loading sentence transformer model: {EMBEDDING_MODEL_NAME}...")
        sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("  Sentence model loaded.")

        # Load FAISS index
        if not os.path.exists(FAISS_INDEX_PATH):
            print(f"Error: FAISS index not found at {FAISS_INDEX_PATH}. Did you run build_vector_store.py?")
            exit()
        print(f"  Loading FAISS index from: {FAISS_INDEX_PATH}...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"  FAISS index loaded. Total vectors: {faiss_index.ntotal}")

        # Load text chunks
        if not os.path.exists(TEXT_CHUNKS_PATH):
            print(f"Error: Text chunks not found at {TEXT_CHUNKS_PATH}. Did you run build_vector_store.py?")
            exit()
        print(f"  Loading text chunks from: {TEXT_CHUNKS_PATH}...")
        with open(TEXT_CHUNKS_PATH, 'r', encoding='utf-8') as f:
            text_chunks_data = json.load(f)
        print(f"  Text chunks loaded. Total chunks: {len(text_chunks_data)}")

    except Exception as e:
        print(f"Error loading resources: {e}")
        exit()
    print("All resources loaded successfully.\n")

# --- RAG Core Functions ---
def retrieve_relevant_chunks(query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """Retrieves the top_k most relevant text chunks for a given query."""
    if not sentence_model or not faiss_index or not text_chunks_data:
        print("Error: Resources not loaded.")
        return []

    print(f"\nEmbedding query: \"{query[:50]}...\"")
    query_embedding = sentence_model.encode([query], convert_to_numpy=True)
    
    print(f"Searching FAISS index for top {top_k} results...")
    # D: distances, I: indices of the vectors in the FAISS index
    distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k)
    
    retrieved_chunks_info = []
    for i in range(len(indices[0])):
        chunk_index_in_faiss = indices[0][i]
        # The text_chunks_data list is ordered the same way as embeddings were added to FAISS
        if 0 <= chunk_index_in_faiss < len(text_chunks_data):
            retrieved_chunk = text_chunks_data[chunk_index_in_faiss]
            retrieved_chunks_info.append({
                'text': retrieved_chunk['text'],
                'source_pdf': retrieved_chunk['source_pdf'],
                'chunk_id_in_pdf': retrieved_chunk['chunk_id_in_pdf'],
                'global_chunk_id': retrieved_chunk['global_chunk_id'],
                'distance': float(distances[0][i]) # Optional: for debugging or ranking
            })
        else:
            print(f"Warning: Retrieved index {chunk_index_in_faiss} is out of bounds for text_chunks_data.")
            
    print(f"Retrieved {len(retrieved_chunks_info)} chunks.")
    return retrieved_chunks_info

def ask_claude_with_context(query: str, context_chunks: list[dict], client: Anthropic) -> str:
    """
    Constructs a prompt with context and asks Claude to answer the query.
    """
    if not context_chunks:
        return "I couldn't find any relevant information in the provided documents to answer your question."

    context_str = "\n\n---\n\n".join([
        f"Source: {chunk['source_pdf']} (Chunk ID: {chunk['global_chunk_id']})\nContent: {chunk['text']}" 
        for chunk in context_chunks
    ])

    prompt = f"""You are a helpful research assistant. Answer the user's question based *only* on the following provided context from research papers. If the context does not contain the answer, state that clearly. Do not use any external knowledge.

Provided Context:
{context_str}

User's Question: {query}

Answer:"""

    print("\nSending prompt to Claude...")


    try:
        response = client.messages.create(
            model=LLM_MODEL_NAME,
            max_tokens=MAX_TOKENS_TO_SAMPLE,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.content[0].text
        print("Claude's response received.")
        return answer
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        return "Sorry, I encountered an error while trying to generate an answer."

# --- Main Interaction Loop ---
def main():
    load_resources() # Load models and data once
    
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

    print("Research Agent Assistant Initialized. Type 'quit' or 'exit' to end.")
    print("Ask me anything about the content of your research papers!")

    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() in ['quit', 'exit']:
            print("Exiting Research Agent Assistant. Goodbye!")
            break
        if not user_query.strip():
            continue

        # 1. Retrieve relevant chunks
        retrieved_chunks = retrieve_relevant_chunks(user_query)

        if not retrieved_chunks:
            print("No relevant information found in the documents for your query.")
            continue
        
        # For debugging, show what was retrieved (optional)
        print("\n--- Top Retrieved Chunks ---")
        for i, chunk_info in enumerate(retrieved_chunks):
            print(f"Chunk {i+1} (Source: {chunk_info['source_pdf']}, Global ID: {chunk_info['global_chunk_id']}, Dist: {chunk_info['distance']:.4f}):")
            print(f"  \"{chunk_info['text'][:150]}...\"") # Print snippet
        print("--- End Retrieved Chunks ---")


        # 2. Ask Claude with context
        answer = ask_claude_with_context(user_query, retrieved_chunks, anthropic_client)
        
        print("\n--- Answer ---")
        print(answer)
        print("--------------")

if __name__ == "__main__":
    main()