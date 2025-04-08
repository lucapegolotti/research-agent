# vector_store_manager.py
import os
import re  # Make sure re is imported if used in clean_text
import json
import time
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Tuple, Optional

# Configuration (consistent with other scripts)
PAPERS_DIR = "research_papers"  # Source for initial bulk load
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "research_papers.index")
TEXT_CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "text_chunks.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MIN_SENTENCES_PER_CHUNK = 3
MAX_SENTENCES_PER_CHUNK = 7


# --- Helper Functions (clean_text, extract_text_from_pdf, chunk_text_by_sentences) ---
# (Ensure these are present and correct as previously defined)
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")  # Replace newlines with spaces
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with single space
    text = re.sub(r"-\s+", "", text)  # De-hyphenate words split across lines
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a single PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text_parts = [
            page.extract_text() for page in reader.pages if page.extract_text()
        ]
        full_text = " ".join(text_parts)
        return clean_text(full_text)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""


def chunk_text_by_sentences(
    text: str, min_sentences: int, max_sentences: int
) -> list[str]:
    """Chunks text into segments of min_sentences to max_sentences."""
    if not text:
        return []
    # Ensure NLTK 'punkt' is available
    try:
        nltk.data.find("tokenizers/punkt")
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download(
            "punkt", quiet=True
        )  # quiet=True to avoid verbose output during chunking
    except Exception as e:
        print(f"NLTK 'punkt' issue: {e}. Manual download might be needed.")

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk_sentences = []

    for sentence in sentences:
        current_chunk_sentences.append(sentence)
        if len(current_chunk_sentences) >= max_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = []

    # Add any remaining sentences as the last chunk, if it meets min criteria
    if current_chunk_sentences and len(current_chunk_sentences) >= min_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    elif chunks and current_chunk_sentences:  # Add to previous chunk if small
        # This logic can sometimes make the last chunk too long if the previous one was already maxed out.
        # A safer bet might be to just append if it meets a minimum, or discard if too small.
        # For simplicity here, we'll keep it but be mindful.
        chunks[-1] += " " + " ".join(current_chunk_sentences)

    return [chunk for chunk in chunks if chunk.strip()]


# --- End Helper Functions ---


def process_single_pdf(
    pdf_path: str, sentence_model: SentenceTransformer, start_global_chunk_id: int
) -> Tuple[Optional[np.ndarray], List[Dict]]:
    """Processes a single PDF, generates chunks and embeddings. (Used for dynamic updates)"""
    filename = os.path.basename(pdf_path)
    print(f"Processing PDF: {filename} (for dynamic add)...")  # Clarify context
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        print(
            f"  Skipping {filename} (dynamic add): extraction error or empty content."
        )
        return None, []
    # print(f"  Extracted ~{len(raw_text):,} characters (dynamic add).") # Less verbose for dynamic

    chunks = chunk_text_by_sentences(
        raw_text, MIN_SENTENCES_PER_CHUNK, MAX_SENTENCES_PER_CHUNK
    )
    if not chunks:
        print(f"  No valid chunks generated for {filename} (dynamic add).")
        return None, []
    # print(f"  Split into {len(chunks)} chunks (dynamic add).")

    # print(f"  Generating embeddings for {len(chunks)} chunks (dynamic add)...") # Less verbose
    chunk_embeddings_np = sentence_model.encode(
        chunks, show_progress_bar=False, convert_to_numpy=True
    )

    new_chunks_metadata = []
    for chunk_idx, chunk_text in enumerate(chunks):
        new_chunks_metadata.append(
            {
                "text": chunk_text,
                "source_pdf": filename,
                "chunk_id_in_pdf": chunk_idx,
                "global_chunk_id": start_global_chunk_id + chunk_idx,
            }
        )

    return chunk_embeddings_np.astype(np.float32), new_chunks_metadata


def build_initial_vector_store():
    """
    Processes ALL PDFs in the PAPERS_DIR, creates embeddings,
    and saves the FAISS index and text chunks. This is for the first-time build.
    """
    print("--- Starting Initial Vector Store Build ---")
    if not os.path.exists(PAPERS_DIR) or not os.listdir(PAPERS_DIR):
        print(
            f"Error: Papers directory '{PAPERS_DIR}' is empty or not found. "
            f"Please run `python bulk_download_papers.py` first."
        )
        return

    if not os.path.exists(VECTOR_STORE_DIR):
        print(f"Creating directory: {VECTOR_STORE_DIR}")
        os.makedirs(VECTOR_STORE_DIR)

    # 1. Load Sentence Transformer model
    print(f"Loading sentence transformer model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dimension: {embedding_dim}")

    all_text_chunks_with_metadata = []
    all_embeddings_list = []  # Use a list to append numpy arrays before concatenating

    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files in '{PAPERS_DIR}' for initial build.")

    global_chunk_id_counter = 0
    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(PAPERS_DIR, filename)
        print(f"\n[{i+1}/{len(pdf_files)}] Processing for initial build: {filename}...")

        # Use the core logic of process_single_pdf but adapt for bulk collection
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            print(f"  Skipping {filename} due to extraction error or empty content.")
            continue
        print(f"  Extracted ~{len(raw_text):,} characters.")

        chunks = chunk_text_by_sentences(
            raw_text, MIN_SENTENCES_PER_CHUNK, MAX_SENTENCES_PER_CHUNK
        )
        if not chunks:
            print(f"  No valid chunks generated for {filename}.")
            continue
        print(f"  Split into {len(chunks)} chunks.")

        print(f"  Generating embeddings for {len(chunks)} chunks...")
        chunk_embeddings_np = model.encode(
            chunks, show_progress_bar=False, convert_to_numpy=True
        )

        for chunk_idx, chunk_text in enumerate(chunks):
            all_text_chunks_with_metadata.append(
                {
                    "text": chunk_text,
                    "source_pdf": filename,
                    "chunk_id_in_pdf": chunk_idx,
                    "global_chunk_id": global_chunk_id_counter,
                }
            )
            global_chunk_id_counter += 1
        all_embeddings_list.append(chunk_embeddings_np)

    if not all_text_chunks_with_metadata:
        print(
            "No text chunks were generated from any PDF during initial build. Exiting."
        )
        return

    # Concatenate all embeddings into a single numpy array
    embeddings_np = np.concatenate(all_embeddings_list, axis=0).astype(np.float32)
    print(
        f"\nTotal text chunks for initial build: {len(all_text_chunks_with_metadata)}"
    )
    print(f"Shape of final embeddings matrix: {embeddings_np.shape}")

    # 2. Build FAISS index
    print("Building FAISS index for initial store...")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)
    print(f"FAISS index built. Total vectors in index: {index.ntotal}")

    # 3. Save FAISS index and text chunks
    print(f"Saving FAISS index to: {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"Saving text chunks metadata to: {TEXT_CHUNKS_PATH}")
    with open(TEXT_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_text_chunks_with_metadata, f, indent=4)

    print("\n--- Initial Vector Store Build Complete ---")
    print(f"FAISS index and text chunks saved successfully to '{VECTOR_STORE_DIR}'.")


if __name__ == "__main__":
    start_time = time.time()
    # This makes running "python vector_store_manager.py" perform the initial build.
    # You could add command-line arguments here if you want more control,
    # e.g., to specify a different papers directory or to only process a single file.
    build_initial_vector_store()
    end_time = time.time()
    print(
        f"Total processing time for initial build: {end_time - start_time:.2f} seconds."
    )
