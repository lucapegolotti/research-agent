# research_agent.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv
import nltk
import time
import re
from datetime import (
    datetime,
    timedelta,
    timezone,
)  # Added for date handling in new feature
from typing import Optional

# Custom modules
import paper_manager
import vector_store_manager

# --- Configuration ---
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "research_papers.index")
TEXT_CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "text_chunks.json")
CHAT_HISTORY_FILE = "chat_history.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL_NAME = "claude-3-5-haiku-20241022"
SUMMARIZER_MODEL_NAME = "claude-3-5-haiku-20241022"

MAX_TOKENS_TO_SAMPLE = 2000
TOP_K_RESULTS = 5

SUMMARIZATION_TRIGGER_COUNT = 12
MESSAGES_TO_KEEP_RAW_AFTER_SUMMARY = 4
MAX_SUMMARY_TOKENS = 350

# For /stay_updated feature
NUM_RECENT_PAPERS_TO_DISPLAY = 7
DAYS_RECENT_THRESHOLD = 90  # e.g., last 3 months

# --- Global Variables ---
sentence_model: SentenceTransformer = None
faiss_index: faiss.Index = None
text_chunks_data: list = []
anthropic_client: Anthropic = None


# --- Initialization Functions ---
def load_resources():
    """Loads RAG resources and initializes Anthropic client."""
    # ... (This function remains the same as the last complete version) ...
    global sentence_model, faiss_index, text_chunks_data, anthropic_client
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not found.")
        exit(1)
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    print("Anthropic client initialized.")
    print("Loading RAG resources...")
    try:
        try:
            nltk.data.find("tokenizers/punkt")
        except nltk.downloader.DownloadError:
            print("NLTK 'punkt' tokenizer not found. Downloading...")
            nltk.download("punkt", quiet=True)
            print("NLTK 'punkt' downloaded.")
        except Exception as e:
            print(f"NLTK 'punkt' check issue: {e}.")
        print(f"  Loading sentence transformer model: {EMBEDDING_MODEL_NAME}...")
        sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("  Sentence model loaded.")
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXT_CHUNKS_PATH):
            print(
                f"Error: FAISS/text_chunks not found in {VECTOR_STORE_DIR}. Run vector_store_manager.py."
            )
            exit(1)
        print(f"  Loading FAISS index from: {FAISS_INDEX_PATH}...")
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"  FAISS index loaded. Vectors: {faiss_index.ntotal}")
        print(f"  Loading text chunks from: {TEXT_CHUNKS_PATH}...")
        with open(TEXT_CHUNKS_PATH, "r", encoding="utf-8") as f:
            text_chunks_data = json.load(f)
            print(f"  Text chunks loaded: {len(text_chunks_data)}")
        if faiss_index.ntotal > 0 and faiss_index.ntotal != len(text_chunks_data):
            print(
                f"Warning: FAISS ({faiss_index.ntotal}) vs chunks ({len(text_chunks_data)}) mismatch."
            )
    except Exception as e:
        print(f"Error loading resources: {e}")
        exit(1)
    print("All RAG resources loaded successfully.\n")


# --- RAG Core & LLM Interaction (retrieve_relevant_chunks, summarize_chat_history, ask_claude_with_context) ---
# ... (These functions remain the same as the last complete version) ...
def retrieve_relevant_chunks(query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    if not sentence_model or not faiss_index:
        return []
    query_embedding = sentence_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k)
    results = []
    if indices.size > 0:
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if 0 <= idx < len(text_chunks_data):
                results.append(
                    {**text_chunks_data[idx], "distance": float(distances[0][i])}
                )
    return results


def summarize_chat_history(history_to_summarize: list[dict]) -> Optional[str]:
    global anthropic_client
    if not history_to_summarize:
        return None
    print("\nSummarizing older conversation...")
    convo_text = "".join(
        f"{('User' if t['role']=='user' else 'Assistant')}: {t['content']}\n\n"
        for t in history_to_summarize
    )
    prompt = f"Summarize this conversation concisely for memory: <convo>{convo_text}</convo> Summary:"
    try:
        resp = anthropic_client.messages.create(
            model=SUMMARIZER_MODEL_NAME,
            max_tokens=MAX_SUMMARY_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = resp.content[0].text.strip()
        print("Summarization complete.")
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None


def ask_claude_with_context(
    current_query: str, context_chunks: list[dict], chat_history: list[dict]
) -> str:
    global anthropic_client
    rag_context_str = "No specific documents retrieved for this question."
    if context_chunks:
        rag_context_str = "\n\n---\n\n".join(
            [
                f"Source: {c['source_pdf']} (ID: {c['global_chunk_id']})\nContent: {c['text']}"
                for c in context_chunks
            ]
        )

    messages = list(chat_history)
    prompt_content = f"""You are a research assistant.
1. If "User's Current Question" is about the conversation (e.g., "my last question"), use conversation history. Ignore <retrieved_context> if irrelevant for such meta-questions.
2. For topical questions, MUST prioritize <retrieved_context>. If empty/irrelevant for topical questions, state that.
3. For Rule 1, "No specific documents retrieved" in <retrieved_context> is expected if it's a meta-question.

Retrieved Context (for topical questions):
<retrieved_context>
{rag_context_str}
</retrieved_context>

User's Current Question: {current_query}
Answer:"""
    messages.append({"role": "user", "content": prompt_content})
    try:
        resp = anthropic_client.messages.create(
            model=LLM_MODEL_NAME, max_tokens=MAX_TOKENS_TO_SAMPLE, messages=messages
        )
        return resp.content[0].text
    except Exception as e:
        print(f"API Error: {e}")
        return "Error communicating with LLM."


# --- Helper for Processing and Integrating a Downloaded Paper ---
def _process_and_integrate_paper(
    pdf_filepath: str, title: str, arxiv_id: Optional[str]
):
    """Helper to process a single downloaded PDF and add to store."""
    global faiss_index, text_chunks_data, sentence_model  # These are modified

    filename = os.path.basename(pdf_filepath)
    # Check if paper (by filename) is already processed. More robust check could use arXiv ID if available.
    if any(chunk["source_pdf"] == filename for chunk in text_chunks_data):
        print(
            f"Paper '{filename}' (from {pdf_filepath}) seems to have already been processed and is in the vector store. Skipping re-addition."
        )
        return False  # Indicate not added or already present

    print(f"Processing '{filename}' to add to vector store...")
    start_global_id = len(text_chunks_data)

    new_embeddings_np, new_chunks_metadata = vector_store_manager.process_single_pdf(
        pdf_path=pdf_filepath,
        sentence_model=sentence_model,
        start_global_chunk_id=start_global_id,
    )

    if new_embeddings_np is None or not new_chunks_metadata:
        print(f"Failed to process the new paper: {filename}. No new data to add.")
        # Consider deleting the downloaded PDF if processing fails consistently
        # os.remove(pdf_filepath)
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
        with open(TEXT_CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(text_chunks_data, f, indent=2)
        print("Vector store updated and saved successfully.")
        print(
            f"Paper '{title}' (ID: {arxiv_id if arxiv_id else 'N/A'}) successfully added to the knowledge base."
        )
        return True
    except Exception as e:
        print(f"CRITICAL Error saving updated vector store: {e}")
        return False


# --- Command Functions ---
def handle_add_paper_command(paper_query: str):
    """Handles the /add_paper command."""
    print(f"\nAttempting to find and add specific paper: '{paper_query}'...")
    # search_arxiv_and_prompt_download returns a dict with 'filepath', 'title', 'arxiv_id' or None
    paper_info = paper_manager.search_arxiv_and_prompt_download(
        query=paper_query, download_dir=paper_manager.PAPERS_DIR
    )
    if paper_info and paper_info.get("filepath"):
        _process_and_integrate_paper(
            paper_info["filepath"], paper_info["title"], paper_info.get("arxiv_id")
        )
    else:
        print("Could not obtain paper information or download failed.")


def handle_stay_updated_command(area_query: str):
    """Handles the /stay_updated command."""
    fetched_papers = paper_manager.fetch_latest_papers_by_query(
        area_query=area_query,
        num_to_present=NUM_RECENT_PAPERS_TO_DISPLAY,
        days_recent=DAYS_RECENT_THRESHOLD,
    )

    if not fetched_papers:
        print(
            f"No recent papers found for '{area_query}' in the last {DAYS_RECENT_THRESHOLD} days."
        )
        return

    print(f"\nFound {len(fetched_papers)} recent papers for '{area_query}':")
    for i, paper in enumerate(fetched_papers):
        pub_date_str = datetime.fromisoformat(paper["published_date_iso"]).strftime(
            "%Y-%m-%d"
        )
        authors_str = ", ".join(paper["authors"][:2]) + (
            " et al." if len(paper["authors"]) > 2 else ""
        )
        print(f"\n  {i+1}. Title: {paper['title']}")
        print(f"     Authors: {authors_str}")
        print(f"     ArXiv ID: {paper['arxiv_id']} (Published: {pub_date_str})")
        print(f"     Abstract: {paper['summary'][:250]}...")  # Show a snippet

    while True:
        selections_input = (
            input(
                "\nEnter numbers of papers to add (e.g., '1 3 4'), 'all', or 'none': "
            )
            .strip()
            .lower()
        )
        if selections_input == "none":
            print("No papers selected for addition.")
            return
        if selections_input == "all":
            selected_indices = list(range(len(fetched_papers)))
            break
        try:
            selected_indices = [
                int(s.strip()) - 1
                for s in re.split(r"[,\s]+", selections_input)
                if s.strip()
            ]
            if all(0 <= idx < len(fetched_papers) for idx in selected_indices):
                break
            else:
                print(
                    "Invalid selection. Please enter numbers from the list, 'all', or 'none'."
                )
        except ValueError:
            print(
                "Invalid input format. Please use numbers separated by spaces or commas, 'all', or 'none'."
            )

    added_count = 0
    for idx in selected_indices:
        selected_paper_info = fetched_papers[idx]
        print(f"\nAttempting to add selected paper: '{selected_paper_info['title']}'")
        # We already have pdf_url, title, arxiv_id from fetch_latest_papers_by_query
        # First, download it using paper_manager.download_pdf
        downloaded_filepath = paper_manager.download_pdf(
            pdf_url=selected_paper_info["pdf_url"],
            title=selected_paper_info["title"],
            arxiv_id=selected_paper_info["arxiv_id"],
            download_dir=paper_manager.PAPERS_DIR,
        )
        if downloaded_filepath:
            if _process_and_integrate_paper(
                downloaded_filepath,
                selected_paper_info["title"],
                selected_paper_info["arxiv_id"],
            ):
                added_count += 1
        time.sleep(0.5)  # Small delay between processing papers

    print(
        f"\nFinished processing selections. Added {added_count} new paper(s) to the knowledge base."
    )


# --- Main Interaction Loop ---
def main():
    load_resources()

    chat_history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                chat_history = json.load(f)
            print(f"Loaded {len(chat_history)} messages from: {CHAT_HISTORY_FILE}")
        except Exception as e:
            print(f"Warn: Could not load chat history: {e}. Starting fresh.")
            chat_history = []
    else:
        print("No existing chat history found. Starting new session.")

    print("\nResearch Agent Assistant Initialized.")
    print(f"LLM: {LLM_MODEL_NAME}, Summarizer: {SUMMARIZER_MODEL_NAME}")
    print("=====================================================================")
    print("Commands:")
    print("  /add_paper <title/keywords/arXiv ID>  - Add a specific paper.")
    print(
        "  /stay_updated <area/keywords>         - Fetch & add recent papers for an area."
    )
    print("  /clear_history                        - Clear conversation history.")
    print("  /quit or /exit                        - End session (history saved).")
    print("Or, ask questions about your research papers!")
    print("=====================================================================")

    while True:
        user_input = input("\nYour input: ").strip()
        history_modified_this_turn = False

        if not user_input:
            continue

        if user_input.lower() in ["/quit", "/exit"]:
            # ... (save history and exit logic - remains same) ...
            print("Saving final chat history...")
            try:
                with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(chat_history, f, indent=2)
                print(f"Chat history saved to {CHAT_HISTORY_FILE}.")
            except Exception as e:
                print(f"Error saving chat history on exit: {e}")
            print("Exiting. Goodbye!")
            break

        elif user_input.lower().startswith("/add_paper "):
            paper_query = user_input[len("/add_paper ") :].strip()
            if paper_query:
                handle_add_paper_command(paper_query)
            else:
                print("Provide paper query after /add_paper.")
            continue

        elif user_input.lower().startswith("/stay_updated "):
            area_query = user_input[len("/stay_updated ") :].strip()
            if area_query:
                handle_stay_updated_command(area_query)
            else:
                print("Provide area/keywords after /stay_updated.")
            continue

        elif user_input.lower() == "/clear_history":
            # ... (clear history logic - remains same) ...
            chat_history = []
            try:
                if os.path.exists(CHAT_HISTORY_FILE):
                    os.remove(CHAT_HISTORY_FILE)
                print("Conversation history cleared (memory & disk).")
            except Exception as e:
                print(f"Error clearing history file: {e}. Cleared in memory.")
            history_modified_this_turn = True

        else:  # Regular query processing
            current_raw_query = user_input
            # ... (RAG, ask_claude_with_context, history update & summarization logic - remains same as previous full script) ...
            print(f'Processing query: "{current_raw_query[:60]}..."')
            retrieved_chunks = retrieve_relevant_chunks(current_raw_query)
            answer = ask_claude_with_context(
                current_raw_query, retrieved_chunks, chat_history
            )
            print("\n--- Answer ---")
            print(answer)
            print("--------------")

            current_turn_user = {"role": "user", "content": current_raw_query}
            current_turn_assistant = {"role": "assistant", "content": answer}

            potential_new_history = list(chat_history)
            potential_new_history.extend([current_turn_user, current_turn_assistant])

            if len(potential_new_history) >= SUMMARIZATION_TRIGGER_COUNT:
                print(
                    f"History length ({len(potential_new_history)}) triggers summarization."
                )
                to_summarize_count = (
                    len(potential_new_history) - MESSAGES_TO_KEEP_RAW_AFTER_SUMMARY
                )
                if to_summarize_count > 0:
                    summary = summarize_chat_history(
                        potential_new_history[:to_summarize_count]
                    )
                    if summary:
                        chat_history = [
                            {"role": "assistant", "content": f"[Sum: {summary}]"}
                        ]
                        chat_history.extend(potential_new_history[to_summarize_count:])
                        print(f"History summarized. New length: {len(chat_history)}.")
                    else:
                        print("Summarization failed. Using unsummarized history.")
                        chat_history = potential_new_history
                else:
                    chat_history = (
                        potential_new_history  # Not enough to split after all
                    )
            else:
                chat_history = potential_new_history
            history_modified_this_turn = True

        if history_modified_this_turn:
            try:
                with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(chat_history, f, indent=2)
            except Exception as e:
                print(f"Error saving chat history: {e}")


if __name__ == "__main__":
    main()
