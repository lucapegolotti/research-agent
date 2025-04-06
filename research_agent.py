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
import re # For clean_text, if not in vector_store_manager explicitly called here
from typing import Optional

# Custom modules
import paper_manager
import vector_store_manager

# --- Configuration ---
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "research_papers.index")
TEXT_CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "text_chunks.json")
CHAT_HISTORY_FILE = "chat_history.json" # File to store persistent chat history

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# Updated model names as per your request
LLM_MODEL_NAME = "claude-3-5-haiku-20241022" # Main Q&A model
SUMMARIZER_MODEL_NAME = "claude-3-5-haiku-20241022" # Using the same Haiku model for summarization

MAX_TOKENS_TO_SAMPLE = 2000 # For main Q&A
TOP_K_RESULTS = 5

# Chat Summarization Configuration
SUMMARIZATION_TRIGGER_COUNT = 12 # Number of messages (user + assistant) to trigger summarization
MESSAGES_TO_KEEP_RAW_AFTER_SUMMARY = 4 # Number of recent raw messages to keep (e.g., last 2 turns)
MAX_SUMMARY_TOKENS = 350 # Max tokens for the generated summary

# --- Global Variables ---
sentence_model: SentenceTransformer = None
faiss_index: faiss.Index = None
text_chunks_data: list = []
anthropic_client: Anthropic = None

# --- Initialization Functions ---
def load_resources():
    """Loads RAG resources and initializes Anthropic client."""
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
        except Exception as e: # pylint: disable=broad-except
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

        if faiss_index.ntotal > 0 and faiss_index.ntotal != len(text_chunks_data):
             print(f"Warning: Mismatch between FAISS index size ({faiss_index.ntotal}) and text_chunks count ({len(text_chunks_data)}).")
             print("This might indicate an issue with the vector store. Consider rebuilding.")

    except Exception as e: # pylint: disable=broad-except
        print(f"Error loading resources: {e}")
        exit(1)
    print("All RAG resources loaded successfully.\n")

# --- RAG Core Functions ---
def retrieve_relevant_chunks(query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """Retrieves the top_k most relevant text chunks for a given query."""
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
            else:
                print(f"Warning: Retrieved index {chunk_index_in_faiss} is out of bounds for text_chunks_data.")
    return retrieved_chunks_info

# --- Chat Summarization ---
def summarize_chat_history(history_to_summarize: list[dict]) -> Optional[str]:
    """Summarizes a portion of the chat history using an LLM."""
    global anthropic_client # Ensure client is accessible
    if not history_to_summarize:
        return None

    print("\nSummarizing older parts of the conversation...")
    conversation_text = ""
    for turn in history_to_summarize:
        role_label = "User" if turn["role"] == "user" else "Assistant"
        conversation_text += f"{role_label}: {turn['content']}\n\n"

    summarizer_prompt = f"""Please provide a concise summary of the following conversation excerpt.
Focus on key topics, questions, and information exchanged. This summary will serve as a memory for an ongoing chat.
Output only the summary itself.

Conversation Excerpt:
<conversation>
{conversation_text}
</conversation>

Concise Summary:"""

    try:
        response = anthropic_client.messages.create(
            model=SUMMARIZER_MODEL_NAME, # Uses the specified Haiku model
            max_tokens=MAX_SUMMARY_TOKENS,
            messages=[{"role": "user", "content": summarizer_prompt}]
        )
        summary = response.content[0].text.strip()
        print("Summarization complete.")
        return summary
    except Exception as e: # pylint: disable=broad-except
        print(f"Error during chat summarization: {e}")
        return None

# --- LLM Interaction ---
def ask_claude_with_context(
    current_query: str,
    context_chunks: list[dict],
    chat_history: list[dict]
) -> str:
    """Constructs a prompt with RAG context and conversational history, then asks Claude."""
    global anthropic_client 
    
    rag_context_str = "No specific documents were retrieved for this particular question." # Default if no chunks
    if context_chunks:
        rag_context_str = "\n\n---\n\n".join([
            f"Source Document: {chunk['source_pdf']} (Internal Chunk ID: {chunk['global_chunk_id']})\nContent: {chunk['text']}"
            for chunk in context_chunks
        ])

    messages_for_api = list(chat_history) 

    # MODIFIED PROMPT LOGIC STARTS HERE
    current_user_prompt_content = f"""You are a helpful research assistant.
Your main task is to answer questions based on the <retrieved_context> from research papers.
However, you should also pay close attention to the full conversation history provided.

**Instructions for answering:**
1.  If the "User's Current Question" is clearly about the ongoing conversation itself (e.g., "What was my last question?", "What did you just say?", "Can you repeat your previous answer?", "Summarize our discussion so far"), please prioritize using the **conversation history** to answer it directly. In these cases, the <retrieved_context> might be irrelevant and can be ignored if it doesn't pertain to the meta-question.
2.  For all other questions that are about research topics, scientific concepts, or the content of papers, you **MUST prioritize the <retrieved_context>**. If the <retrieved_context> is empty or does not contain the answer for these topical questions, then state that you cannot answer from the provided documents for the current question.
3.  If the <retrieved_context> is "No specific documents were retrieved...", this applies to topical questions. For conversational questions (Rule 1), this lack of retrieved documents is expected and does not prevent you from answering using the conversation history.

Retrieved Context (for topical questions):
<retrieved_context>
{rag_context_str}
</retrieved_context>

User's Current Question: {current_query}

Answer:"""
    # MODIFIED PROMPT LOGIC ENDS HERE

    messages_for_api.append({"role": "user", "content": current_user_prompt_content})
    
    try:
        response = anthropic_client.messages.create(
            model=LLM_MODEL_NAME,
            max_tokens=MAX_TOKENS_TO_SAMPLE,
            messages=messages_for_api
        )
        answer = response.content[0].text
        return answer
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        return "Sorry, I encountered an error while trying to communicate with the language model."

# --- Function to Add New Paper ---
def add_new_paper_to_store(paper_query: str):
    """Finds, downloads, processes, and adds a new paper to the vector store."""
    global faiss_index, text_chunks_data, sentence_model # These are modified

    print(f"\nAttempting to find and add paper related to: '{paper_query}'...")
    downloaded_pdf_path = paper_manager.search_arxiv_and_prompt_download(
        query=paper_query, download_dir=paper_manager.PAPERS_DIR # Use PAPERS_DIR from paper_manager
    )

    if not downloaded_pdf_path:
        print("Paper download cancelled or failed.")
        return

    filename = os.path.basename(downloaded_pdf_path)
    if any(chunk['source_pdf'] == filename for chunk in text_chunks_data):
        print(f"Paper '{filename}' appears to have already been processed. Skipping re-addition.")
        return

    print(f"Processing '{filename}' to add to vector store...")
    start_global_id = len(text_chunks_data) # Next available global ID
    
    new_embeddings_np, new_chunks_metadata = vector_store_manager.process_single_pdf(
        pdf_path=downloaded_pdf_path,
        sentence_model=sentence_model, 
        start_global_chunk_id=start_global_id
    )

    if new_embeddings_np is None or not new_chunks_metadata:
        print(f"Failed to process the new paper: {filename}. No new data to add.")
        return

    print("Updating in-memory vector store with new paper...")
    faiss_index.add(new_embeddings_np) # process_single_pdf should ensure float32
    text_chunks_data.extend(new_chunks_metadata)
    print(f"  Added {len(new_chunks_metadata)} new chunks from '{filename}'.")
    print(f"  FAISS index size now: {faiss_index.ntotal} vectors.")
    print(f"  Total text_chunks entries: {len(text_chunks_data)}.")

    print("Saving updated vector store to disk...")
    try:
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        with open(TEXT_CHUNKS_PATH, 'w', encoding='utf-8') as f:
            json.dump(text_chunks_data, f, indent=2)
        print("Vector store updated and saved successfully.")
    except Exception as e: # pylint: disable=broad-except
        print(f"CRITICAL Error saving updated vector store: {e}")
        print("In-memory store is updated, but changes are NOT persistent on disk for this addition!")
        
    print(f"Paper '{filename}' successfully added to the knowledge base.")

# --- Main Interaction Loop ---
def main():
    load_resources()
    
    chat_history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
            print(f"Loaded {len(chat_history)} messages from previous session: {CHAT_HISTORY_FILE}")
        except Exception as e: # pylint: disable=broad-except
            print(f"Warning: Could not load chat history from {CHAT_HISTORY_FILE}: {e}. Starting fresh.")
            chat_history = []
    else:
        print("No existing chat history found. Starting a new session.")

    print("\nResearch Agent Assistant Initialized (Conversational & Persistent Mode).")
    print(f"Using LLM: {LLM_MODEL_NAME}, Summarizer: {SUMMARIZER_MODEL_NAME}")
    print("=====================================================================")
    print("Commands:")
    print("  /add_paper <title, keywords, or arXiv ID>  - Find and add a new paper.")
    print("  /clear_history                             - Clear conversation history (current & persistent).")
    print("  /quit or /exit                             - End the session (history will be saved).")
    print("Ask me anything about the content of your research papers!")
    print("=====================================================================")

    while True:
        user_input = input("\nYour input: ").strip()
        history_modified_this_turn = False

        if not user_input:
            continue

        if user_input.lower() in ['/quit', '/exit']:
            print("Saving final chat history...")
            try:
                with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
                    json.dump(chat_history, f, indent=2)
                print(f"Chat history saved to {CHAT_HISTORY_FILE}.")
            except Exception as e: # pylint: disable=broad-except
                print(f"Error saving chat history on exit: {e}")
            print("Exiting Research Agent Assistant. Goodbye!")
            break
        
        if user_input.lower().startswith("/add_paper "):
            paper_query = user_input[len("/add_paper "):].strip()
            if paper_query:
                add_new_paper_to_store(paper_query)
            else:
                print("Please provide a paper query after /add_paper.")
            continue # No direct chat history modification by this command's output
        
        if user_input.lower() == "/clear_history":
            chat_history = []
            try:
                if os.path.exists(CHAT_HISTORY_FILE):
                    os.remove(CHAT_HISTORY_FILE)
                print("Conversation history cleared (in memory and on disk).")
            except Exception as e: # pylint: disable=broad-except
                print(f"Error clearing chat history file: {e}. History cleared in memory only.")
            history_modified_this_turn = True
        
        if not history_modified_this_turn and user_input.lower() != "/clear_history":
            current_raw_query = user_input
            print(f"Processing your query: \"{current_raw_query[:60]}...\"")
            
            retrieved_chunks = retrieve_relevant_chunks(current_raw_query)
            answer = ask_claude_with_context(current_raw_query, retrieved_chunks, chat_history)
            
            print("\n--- Answer ---")
            print(answer)
            print("--------------")

            current_chat_turn_user = {"role": "user", "content": current_raw_query}
            current_chat_turn_assistant = {"role": "assistant", "content": answer}
            
            potential_new_history = list(chat_history)
            potential_new_history.append(current_chat_turn_user)
            potential_new_history.append(current_chat_turn_assistant)

            if len(potential_new_history) >= SUMMARIZATION_TRIGGER_COUNT:
                print(f"Chat history length ({len(potential_new_history)}) triggers summarization ({SUMMARIZATION_TRIGGER_COUNT} target).")
                num_messages_to_summarize = len(potential_new_history) - MESSAGES_TO_KEEP_RAW_AFTER_SUMMARY
                
                if num_messages_to_summarize > 0:
                    history_portion_to_summarize = potential_new_history[:num_messages_to_summarize]
                    recent_raw_history_to_keep = potential_new_history[num_messages_to_summarize:]
                    summary_text = summarize_chat_history(history_portion_to_summarize)

                    if summary_text:
                        chat_history = [{"role": "assistant", "content": f"[Summary of earlier conversation: {summary_text}]"}]
                        chat_history.extend(recent_raw_history_to_keep)
                        print(f"Chat history summarized. New length: {len(chat_history)} messages.")
                    else:
                        print("Summarization failed. Appending current turn without full summarization.")
                        chat_history = potential_new_history
                        if len(chat_history) > SUMMARIZATION_TRIGGER_COUNT + 4: 
                             chat_history = chat_history[-(SUMMARIZATION_TRIGGER_COUNT + 2):] # Safety truncation
                else: 
                    chat_history = potential_new_history
            else: 
                chat_history = potential_new_history
            history_modified_this_turn = True

        if history_modified_this_turn:
            try:
                with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
                    json.dump(chat_history, f, indent=2)
            except Exception as e: # pylint: disable=broad-except
                print(f"Error saving chat history to {CHAT_HISTORY_FILE}: {e}")

if __name__ == "__main__":
    main()