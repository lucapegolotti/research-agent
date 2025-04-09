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
from datetime import datetime, timedelta, timezone # For /stay_updated
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

# For /stay_updated feature
NUM_RECENT_PAPERS_TO_DISPLAY = 7
DAYS_RECENT_THRESHOLD = 90 # e.g., last 3 months

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
                print(f"Warning: Retrieved index {chunk_index_in_faiss} is out of bounds for text_chunks_data (size: {len(text_chunks_data)}).")
    return retrieved_chunks_info

# --- Chat Summarization ---
def summarize_chat_history(history_to_summarize: list[dict]) -> Optional[str]:
    """Summarizes a portion of the chat history using an LLM."""
    global anthropic_client
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
            model=SUMMARIZER_MODEL_NAME,
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
    
    rag_context_str = "No specific documents were retrieved for this particular question."
    if context_chunks:
        rag_context_str = "\n\n---\n\n".join([
            f"Source Document: {chunk['source_pdf']} (Internal Chunk ID: {chunk['global_chunk_id']})\nContent: {chunk['text']}"
            for chunk in context_chunks
        ])

    messages_for_api = list(chat_history) 

    current_user_prompt_content = f"""You are a helpful research assistant.
Your main task is to answer questions based on the <retrieved_context> from research papers.
However, you should also pay close attention to the full conversation history provided.

**Instructions for answering:**
1.  If the "User's Current Question" is clearly about the ongoing conversation itself (e.g., "What was my last question?", "What did you just say?", "Can you repeat your previous answer?", "Summarize our discussion so far"), please prioritize using the **conversation history** to answer it directly. In these cases, the <retrieved_context> might be irrelevant and can be ignored if it doesn't pertain to the meta-question.
2.  For all other questions that are about research topics, scientific concepts, or the content of papers, you **MUST prioritize the <retrieved_context>**. If the <retrieved_context> is empty ("No specific documents were retrieved...") or does not contain the answer for these topical questions, then state that you cannot answer from the provided documents for the current question.
3.  For Rule 1, if the <retrieved_context> is "No specific documents were retrieved...", this is expected for meta-questions and does not prevent you from answering using the conversation history.

Retrieved Context (for topical questions):
<retrieved_context>
{rag_context_str}
</retrieved_context>

User's Current Question: {current_query}

Answer:"""
    messages_for_api.append({"role": "user", "content": current_user_prompt_content})
    
    try:
        response = anthropic_client.messages.create(
            model=LLM_MODEL_NAME,
            max_tokens=MAX_TOKENS_TO_SAMPLE,
            messages=messages_for_api
        )
        answer = response.content[0].text
        return answer
    except Exception as e: # pylint: disable=broad-except
        print(f"Error calling Anthropic API: {e}")
        return "Sorry, I encountered an error while trying to communicate with the language model."

# --- Helper for Processing and Integrating a Downloaded Paper ---
def _process_and_integrate_paper(pdf_filepath: str, title: str, arxiv_id: Optional[str]):
    """Helper to process a single downloaded PDF and add to store. Returns True if successful."""
    global faiss_index, text_chunks_data, sentence_model

    filename = os.path.basename(pdf_filepath)
    if any(chunk['source_pdf'] == filename for chunk in text_chunks_data):
        print(f"Paper '{filename}' (from {pdf_filepath}) seems already processed. Skipping.")
        return False 

    print(f"Processing '{filename}' to add to vector store...")
    start_global_id = len(text_chunks_data)
    
    new_embeddings_np, new_chunks_metadata = vector_store_manager.process_single_pdf(
        pdf_path=pdf_filepath,
        sentence_model=sentence_model,
        start_global_chunk_id=start_global_id
    )

    if new_embeddings_np is None or not new_chunks_metadata:
        print(f"Failed to process the new paper: {filename}. No new data to add.")
        return False

    print("Updating in-memory vector store with new paper...")
    faiss_index.add(new_embeddings_np)
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
        print(f"Paper '{title}' (ID: {arxiv_id if arxiv_id else 'N/A'}) successfully added.")
        return True
    except Exception as e: # pylint: disable=broad-except
        print(f"CRITICAL Error saving updated vector store: {e}")
        return False

# --- Command Functions ---
def handle_add_paper_command(paper_query: str):
    """Handles the /add_paper command."""
    print(f"\nAttempting to find and add specific paper: '{paper_query}'...")
    paper_info = paper_manager.search_arxiv_and_prompt_download(
        query=paper_query,
        download_dir=paper_manager.PAPERS_DIR
    )
    if paper_info and paper_info.get("filepath"):
        _process_and_integrate_paper(paper_info["filepath"], paper_info["title"], paper_info.get("arxiv_id"))
    else:
        print("Could not obtain paper information or download failed for /add_paper.")

def handle_stay_updated_command(area_query: str):
    """Handles the /stay_updated command."""
    fetched_papers = paper_manager.fetch_latest_papers_by_query(
        area_query=area_query,
        num_to_present=NUM_RECENT_PAPERS_TO_DISPLAY,
        days_recent=DAYS_RECENT_THRESHOLD
    )

    if not fetched_papers:
        print(f"No recent papers found for '{area_query}' in the last {DAYS_RECENT_THRESHOLD} days.")
        return

    print(f"\nFound {len(fetched_papers)} recent papers for '{area_query}':")
    for i, paper in enumerate(fetched_papers):
        try:
            # Ensure published_date_iso is a string before calling fromisoformat
            pub_date_obj = datetime.fromisoformat(str(paper['published_date_iso']))
            pub_date_str = pub_date_obj.strftime('%Y-%m-%d')
        except (ValueError, TypeError): 
            pub_date_str = paper.get('published_date_iso', 'N/A') # Fallback
            
        authors_str = ', '.join(paper.get('authors', [])[:2]) + (' et al.' if len(paper.get('authors', [])) > 2 else '')
        print(f"\n  {i+1}. Title: {paper.get('title', 'N/A')}")
        print(f"     Authors: {authors_str}")
        print(f"     ArXiv ID: {paper.get('arxiv_id', 'N/A')} (Published: {pub_date_str})")
        print(f"     Abstract: {paper.get('summary', 'N/A')[:250]}...")

    while True:
        selections_input = input("\nEnter numbers of papers to add (e.g., '1 3 4'), 'all', or 'none': ").strip().lower()
        if selections_input == 'none':
            print("No papers selected for addition.")
            return
        if selections_input == 'all':
            selected_indices = list(range(len(fetched_papers)))
            break
        try:
            selected_indices = [int(s.strip()) - 1 for s in re.split(r'[,\s]+', selections_input) if s.strip()]
            if all(0 <= idx < len(fetched_papers) for idx in selected_indices):
                break
            else:
                print("Invalid selection. Please enter numbers from the list, 'all', or 'none'.")
        except ValueError:
            print("Invalid input format. Please use numbers separated by spaces or commas, 'all', or 'none'.")

    added_count = 0
    for idx in selected_indices:
        if not (0 <= idx < len(fetched_papers)): continue
        selected_paper_info = fetched_papers[idx]
        print(f"\nAttempting to add selected paper: '{selected_paper_info.get('title', 'N/A')}'")
        downloaded_filepath = paper_manager.download_pdf(
            pdf_url=selected_paper_info.get('pdf_url',''), # Ensure these keys exist
            title=selected_paper_info.get('title','Untitled Paper'),
            arxiv_id=selected_paper_info.get('arxiv_id'),
            download_dir=paper_manager.PAPERS_DIR
        )
        if downloaded_filepath:
            if _process_and_integrate_paper(downloaded_filepath, selected_paper_info.get('title','Untitled Paper'), selected_paper_info.get('arxiv_id')):
                added_count += 1
        time.sleep(0.5) 
    print(f"\nFinished processing selections. Added {added_count} new paper(s).")

def display_help():
    """Displays the help message with available commands."""
    help_text = """
Available Commands:
-------------------
  /help                                        - Display this help message.
  /add_paper <title/keywords/arXiv ID>         - Search for a specific paper on arXiv, then optionally download and add it.
  /stay_updated <area/keywords>                - Fetch a list of recent papers (last ~90 days) from arXiv for a research area. 
                                                 You can then select papers to download and add.
  /list_papers                                 - List all papers currently loaded in the knowledge base.
  /clear_history                               - Clear the chat conversation history (memory & persistent file).
  /quit or /exit                               - Exit the Research Agent Assistant (saves chat history).

You can also type any question to query the papers already in your knowledge base.
"""
    print(help_text)

def display_indexed_papers():
    """Displays a list of unique paper filenames currently in the vector store."""
    global text_chunks_data

    if not text_chunks_data:
        print("\nThe knowledge base is currently empty. No papers have been processed yet.")
        return

    indexed_paper_filenames = set()
    for chunk in text_chunks_data:
        if 'source_pdf' in chunk:
            indexed_paper_filenames.add(chunk['source_pdf'])

    if not indexed_paper_filenames:
        print("\nNo paper source information found in the current knowledge base.")
        return

    print("\nPapers currently in the knowledge base:")
    print("---------------------------------------")
    sorted_paper_filenames = sorted(list(indexed_paper_filenames))
    for i, paper_filename in enumerate(sorted_paper_filenames):
        print(f"  {i+1}. {paper_filename}")
    print("---------------------------------------")
    print(f"Total unique papers: {len(sorted_paper_filenames)}")

# --- Main Interaction Loop ---
def main():
    load_resources()
    
    chat_history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f: chat_history = json.load(f)
            print(f"Loaded {len(chat_history)} messages from: {CHAT_HISTORY_FILE}")
        except Exception as e: # pylint: disable=broad-except
            print(f"Warn: Could not load chat history from {CHAT_HISTORY_FILE}: {e}. Starting fresh."); chat_history = []
    else: print("No existing chat history. Starting new session.")

    print("\nResearch Agent Assistant Initialized.")
    print(f"LLM: {LLM_MODEL_NAME}, Summarizer: {SUMMARIZER_MODEL_NAME}")
    print("=====================================================================")
    print("Type /help for a list of commands.")
    print("Or, ask questions about your research papers!")
    print("=====================================================================")

    while True:
        user_input = input("\nYour input: ").strip()
        history_modified_this_turn = False

        if not user_input: continue

        if user_input.lower() == "/help":
            display_help()
            continue

        elif user_input.lower() == "/list_papers":
            display_indexed_papers()
            continue

        elif user_input.lower() in ['/quit', '/exit']:
            print("Saving final chat history...");
            try:
                with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f: json.dump(chat_history, f, indent=2)
                print(f"Chat history saved to {CHAT_HISTORY_FILE}.")
            except Exception as e: print(f"Error saving chat history on exit: {e}") # pylint: disable=broad-except
            print("Exiting. Goodbye!"); break
        
        elif user_input.lower().startswith("/add_paper "):
            paper_query = user_input[len("/add_paper "):].strip()
            if paper_query: handle_add_paper_command(paper_query)
            else: print("Provide paper query after /add_paper.")
            continue 
        
        elif user_input.lower().startswith("/stay_updated "):
            area_query = user_input[len("/stay_updated "):].strip()
            if area_query: handle_stay_updated_command(area_query)
            else: print("Provide area/keywords after /stay_updated.")
            continue

        elif user_input.lower() == "/clear_history":
            chat_history = [];
            try:
                if os.path.exists(CHAT_HISTORY_FILE): os.remove(CHAT_HISTORY_FILE)
                print("Conversation history cleared (memory & disk).")
            except Exception as e: print(f"Error clearing history file: {e}. Cleared in memory.") # pylint: disable=broad-except
            history_modified_this_turn = True
        
        else: # Regular query processing
            current_raw_query = user_input
            print(f"Processing query: \"{current_raw_query[:60]}...\"")
            retrieved_chunks = retrieve_relevant_chunks(current_raw_query)
            answer = ask_claude_with_context(current_raw_query, retrieved_chunks, chat_history)
            print("\n--- Answer ---"); print(answer); print("--------------")

            current_turn_user = {"role": "user", "content": current_raw_query}
            current_turn_assistant = {"role": "assistant", "content": answer}
            
            potential_new_history = list(chat_history)
            potential_new_history.extend([current_turn_user, current_turn_assistant])

            if len(potential_new_history) >= SUMMARIZATION_TRIGGER_COUNT:
                print(f"History length ({len(potential_new_history)}) triggers summarization.");
                to_summarize_count = len(potential_new_history) - MESSAGES_TO_KEEP_RAW_AFTER_SUMMARY
                if to_summarize_count > 0:
                    summary = summarize_chat_history(potential_new_history[:to_summarize_count])
                    if summary:
                        chat_history = [{"role": "assistant", "content": f"[Summary of earlier conversation: {summary}]"}]
                        chat_history.extend(potential_new_history[to_summarize_count:])
                        print(f"History summarized. New length: {len(chat_history)}.")
                    else: 
                        print("Summarization failed. Using unsummarized history for this turn."); 
                        chat_history = potential_new_history
                        if len(chat_history) > SUMMARIZATION_TRIGGER_COUNT + MESSAGES_TO_KEEP_RAW_AFTER_SUMMARY + 2: 
                            chat_history = chat_history[-(SUMMARIZATION_TRIGGER_COUNT + MESSAGES_TO_KEEP_RAW_AFTER_SUMMARY):]
                            print(f"Applied safety truncation. History length now: {len(chat_history)}")
                else: 
                    chat_history = potential_new_history 
            else: 
                chat_history = potential_new_history
            history_modified_this_turn = True

        if history_modified_this_turn:
            try:
                with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f: json.dump(chat_history, f, indent=2)
            except Exception as e: print(f"Error saving chat history: {e}") # pylint: disable=broad-except

if __name__ == "__main__":
    main()