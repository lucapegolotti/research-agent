# research_agent.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv
import nltk # For NLTK check
import time

# Custom modules
import paper_manager
import vector_store_manager

# --- Configuration ---
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "research_papers.index")
TEXT_CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "text_chunks.json")
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL_NAME = "claude-3-5-haiku-20241022" # Using Haiku for faster/cheaper conversation, or use Sonnet/Opus
MAX_TOKENS_TO_SAMPLE = 2000
TOP_K_RESULTS = 5
MAX_CHAT_HISTORY_MESSAGES = 10 # Number of messages (e.g., 5 user + 5 assistant turns) to keep

# --- Global Variables ---
sentence_model: SentenceTransformer = None
faiss_index: faiss.Index = None
text_chunks_data: list = []
anthropic_client: Anthropic = None

# --- Initialization Functions ---
def load_resources():
    """Loads resources and initializes Anthropic client."""
    global sentence_model, faiss_index, text_chunks_data, anthropic_client
    # ... (Keep the existing load_resources function exactly as it was in the previous version) ...
    # It should initialize anthropic_client, sentence_model, faiss_index, text_chunks_data
    # and handle NLTK 'punkt' download.
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not found. Please set it in your .env file.")
        exit(1)
    
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    print("Anthropic client initialized.")

    print("Loading RAG resources...")
    try:
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            print("NLTK 'punkt' tokenizer not found. Attempting to download...")
            nltk.download('punkt', quiet=True)
            print("NLTK 'punkt' downloaded.")
        except Exception as e:
            print(f"NLTK 'punkt' check/download issue: {e}. Ensure it's available.")

        print(f"  Loading sentence transformer model: {EMBEDDING_MODEL_NAME}...")
        sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("  Sentence model loaded.")

        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXT_CHUNKS_PATH):
            print(f"Error: FAISS index or text chunks not found in {VECTOR_STORE_DIR}.")
            print("Please run `python vector_store_manager.py` to build the initial store.")
            exit(1)
            
        print(f"  Loading FAISS index from: {FAISS_INDEX_PATH}...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"  FAISS index loaded. Total vectors: {faiss_index.ntotal}")

        print(f"  Loading text chunks from: {TEXT_CHUNKS_PATH}...")
        with open(TEXT_CHUNKS_PATH, 'r', encoding='utf-8') as f:
            text_chunks_data = json.load(f)
        print(f"  Text chunks loaded. Total chunks: {len(text_chunks_data)}")

        if faiss_index.ntotal > 0 and faiss_index.ntotal != len(text_chunks_data):
             print(f"Warning: Mismatch between FAISS index size ({faiss_index.ntotal}) and text_chunks count ({len(text_chunks_data)}).")

    except Exception as e:
        print(f"Error loading resources: {e}")
        exit(1)
    print("All resources loaded successfully.\n")


# --- RAG Core Functions ---
def retrieve_relevant_chunks(query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    # ... (Keep this function exactly as it was) ...
    if not sentence_model or not faiss_index or not text_chunks_data:
        print("Error: RAG Resources not loaded properly.")
        return []
    query_embedding = sentence_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k)
    retrieved_chunks_info = []
    if indices.size > 0:
        for i in range(len(indices[0])):
            chunk_index_in_faiss = indices[0][i]
            if 0 <= chunk_index_in_faiss < len(text_chunks_data):
                retrieved_chunk = text_chunks_data[chunk_index_in_faiss]
                retrieved_chunks_info.append({
                    'text': retrieved_chunk['text'],
                    'source_pdf': retrieved_chunk['source_pdf'],
                    'chunk_id_in_pdf': retrieved_chunk.get('chunk_id_in_pdf', -1),
                    'global_chunk_id': retrieved_chunk['global_chunk_id'],
                    'distance': float(distances[0][i])
                })
    return retrieved_chunks_info


def ask_claude_with_context(
    current_query: str,
    context_chunks: list[dict],
    chat_history: list[dict] # New parameter
) -> str:
    """
    Constructs a prompt with RAG context and conversational history, 
    then asks Claude to answer the current query.
    """
    global anthropic_client
    
    # Prepare the RAG context string for the current query
    if not context_chunks:
        # If no specific RAG context for THIS query, we can still proceed with chat history
        # Or decide to return a specific message. Let's allow Claude to respond based on history if no new context.
        rag_context_str = "No specific documents were retrieved for this particular question. Please answer based on the ongoing conversation if relevant, or state if you cannot answer."
    else:
        rag_context_str = "\n\n---\n\n".join([
            f"Source Document: {chunk['source_pdf']} (Internal Chunk ID: {chunk['global_chunk_id']})\nContent: {chunk['text']}"
            for chunk in context_chunks
        ])

    # Construct the messages list for the API
    # Start with a copy of the existing chat history
    messages_for_api = list(chat_history)

    # Add the current user query with its specific RAG context
    # This prompt structure helps Claude distinguish between general conversation and the specific RAG task for the current query
    current_user_prompt_content = f"""You are a helpful research assistant.
Here is some relevant context retrieved from research papers for the user's *current* question:
<retrieved_context>
{rag_context_str}
</retrieved_context>

Based *only* on the provided <retrieved_context> for the current question, and the preceding conversation history (if relevant for context like pronouns or follow-ups), please answer the user's current question.
If the <retrieved_context> is "No specific documents were retrieved...", then answer based on the conversation history or state if you cannot answer the current question.
Do not use any external knowledge. Be concise.

User's Current Question: {current_query}

Answer:"""

    messages_for_api.append({"role": "user", "content": current_user_prompt_content})
    
    # print("\nSending prompt to Claude (with history)...") # For debugging
    # print(f"Full messages structure for API: {json.dumps(messages_for_api, indent=2)}")


    try:
        response = anthropic_client.messages.create(
            model=LLM_MODEL_NAME,
            max_tokens=MAX_TOKENS_TO_SAMPLE,
            messages=messages_for_api,
            # system_prompt= "You are a helpful research assistant. Focus on answering based on provided context from papers." # Optional system prompt
        )
        answer = response.content[0].text
        return answer
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        return "Sorry, I encountered an error while trying to communicate with the language model."

# --- Function to Add New Paper ---
def add_new_paper_to_store(paper_query: str):
    # ... (Keep this function exactly as it was in the previous version) ...
    # It modifies globals: faiss_index, text_chunks_data, sentence_model
    global faiss_index, text_chunks_data, sentence_model
    print(f"\nAttempting to find and add paper related to: '{paper_query}'...")
    downloaded_pdf_path = paper_manager.search_arxiv_and_prompt_download(
        query=paper_query, download_dir=paper_manager.PAPERS_DIR
    )
    if not downloaded_pdf_path: print("Paper download cancelled or failed."); return
    filename = os.path.basename(downloaded_pdf_path)
    if any(chunk['source_pdf'] == filename for chunk in text_chunks_data):
        print(f"Paper '{filename}' appears to have already been processed. Skipping."); return
    print(f"Processing '{filename}' to add to vector store...")
    start_global_id = len(text_chunks_data)
    new_embeddings_np, new_chunks_metadata = vector_store_manager.process_single_pdf(
        pdf_path=downloaded_pdf_path, sentence_model=sentence_model, start_global_chunk_id=start_global_id
    )
    if new_embeddings_np is None or not new_chunks_metadata:
        print(f"Failed to process {filename}."); return
    print("Updating in-memory vector store..."); faiss_index.add(new_embeddings_np)
    text_chunks_data.extend(new_chunks_metadata)
    print(f"  Added {len(new_chunks_metadata)} new chunks. Index: {faiss_index.ntotal}, Chunks: {len(text_chunks_data)}")
    try:
        print("Saving updated vector store to disk..."); faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        with open(TEXT_CHUNKS_PATH, 'w', encoding='utf-8') as f: json.dump(text_chunks_data, f, indent=4)
        print("Vector store updated and saved.")
    except Exception as e: print(f"CRITICAL Error saving updated vector store: {e}")
    print(f"Paper '{filename}' successfully added.")


# --- Main Interaction Loop ---
def main():
    load_resources()
    
    chat_history = [] # Initialize chat history for the session

    print("\nResearch Agent Assistant Initialized (Conversational Mode).")
    print("==========================================================")
    print("Commands:")
    print("  /add_paper <title, keywords, or arXiv ID>  - Find and add a new paper.")
    print("  /clear_history                             - Clear conversation history.")
    print("  /quit or /exit                             - End the session.")
    print("Ask me anything about the content of your research papers!")
    print("==========================================================")

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
                # Optional: Decide if adding a paper should clear chat history or not.
                # For now, it doesn't.
            else:
                print("Please provide a paper title, keywords, or arXiv ID after /add_paper.")
            continue
        
        if user_input.lower() == "/clear_history":
            chat_history = []
            print("Conversation history cleared.")
            continue

        # Regular query processing
        current_raw_query = user_input # Store the raw query for history
        print(f"Processing your query: \"{current_raw_query[:60]}...\"")
        
        retrieved_chunks = retrieve_relevant_chunks(current_raw_query)
        
        # The 'answer' is generated based on current query, its RAG context, and past chat history
        answer = ask_claude_with_context(current_raw_query, retrieved_chunks, chat_history)
        
        print("\n--- Answer ---")
        print(answer)
        print("--------------")

        # Manage and append to chat history
        # Simple truncation: keep only the last MAX_CHAT_HISTORY_MESSAGES
        if len(chat_history) >= MAX_CHAT_HISTORY_MESSAGES:
            # Remove the oldest turn (1 user message + 1 assistant message = 2 items)
            # to make space for the new turn.
            num_messages_to_remove = (len(chat_history) - MAX_CHAT_HISTORY_MESSAGES) + 2
            chat_history = chat_history[num_messages_to_remove:]
            # print(f"(Trimmed chat history to last {len(chat_history)} messages)") # For debugging

        chat_history.append({"role": "user", "content": current_raw_query})
        chat_history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()