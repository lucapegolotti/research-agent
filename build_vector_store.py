import os
import re
import time
import json # For saving text chunks metadata
import numpy as np
import faiss
from pypdf import PdfReader # Using pypdf
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

# --- Configuration ---
PAPERS_DIR = "research_papers"
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "research_papers.index")
TEXT_CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "text_chunks.json")

# Embedding model - 'all-MiniLM-L6-v2' is a good balance of speed and performance
# Other options: 'msmarco-distilbert-base-v4', 'all-mpnet-base-v2' (slower but potentially better)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Chunking strategy
# MAX_CHUNK_LENGTH_CHARS = 1500 # Approximate max characters per chunk
MIN_SENTENCES_PER_CHUNK = 3   # Minimum number of sentences to form a chunk
MAX_SENTENCES_PER_CHUNK = 7   # Maximum number of sentences per chunk

# --- Helper Functions ---

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'-\s+', '', text) # De-hyphenate words split across lines like "exam- ple" -> "example"
    # Add any other specific cleaning rules if needed
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a single PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
        full_text = " ".join(text_parts)
        return clean_text(full_text)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def chunk_text_by_sentences(text: str, min_sentences: int, max_sentences: int) -> list[str]:
    """Chunks text into segments of min_sentences to max_sentences."""
    if not text:
        return []
    
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
    elif chunks and current_chunk_sentences: # Add to previous chunk if small
        chunks[-1] += " " + " ".join(current_chunk_sentences)
        
    return [chunk for chunk in chunks if chunk.strip()]


# --- Main Processing ---
def process_papers_and_build_index():
    """
    Processes all PDFs in the PAPERS_DIR, creates embeddings,
    and saves the FAISS index and text chunks.
    """
    if not os.path.exists(PAPERS_DIR):
        print(f"Error: Papers directory '{PAPERS_DIR}' not found. Please run download_papers.py first.")
        return

    if not os.path.exists(VECTOR_STORE_DIR):
        print(f"Creating directory: {VECTOR_STORE_DIR}")
        os.makedirs(VECTOR_STORE_DIR)

    # 1. Download NLTK sentence tokenizer if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt')
    except Exception as e: # Handle other potential NLTK issues
        print(f"Could not verify NLTK 'punkt' tokenizer: {e}. Ensure it's downloaded manually if issues persist.")


    # 2. Load Sentence Transformer model
    print(f"Loading sentence transformer model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dimension: {embedding_dim}")

    all_text_chunks_with_metadata = [] # Stores {'text': chunk, 'source_pdf': filename, 'chunk_id': global_id}
    all_embeddings = []
    
    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files in '{PAPERS_DIR}'.")

    global_chunk_id_counter = 0
    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(PAPERS_DIR, filename)
        print(f"\n[{i+1}/{len(pdf_files)}] Processing: {filename}...")

        # 3. Extract text
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            print(f"  Skipping {filename} due to extraction error or empty content.")
            continue
        print(f"  Extracted ~{len(raw_text):,} characters.")

        # 4. Chunk text
        chunks = chunk_text_by_sentences(raw_text, MIN_SENTENCES_PER_CHUNK, MAX_SENTENCES_PER_CHUNK)
        if not chunks:
            print(f"  No valid chunks generated for {filename}.")
            continue
        print(f"  Split into {len(chunks)} chunks.")

        # 5. Generate embeddings for chunks
        print(f"  Generating embeddings for {len(chunks)} chunks...")
        chunk_embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
        
        for chunk_idx, chunk_text in enumerate(chunks):
            all_text_chunks_with_metadata.append({
                'text': chunk_text,
                'source_pdf': filename,
                'chunk_id_in_pdf': chunk_idx,
                'global_chunk_id': global_chunk_id_counter
            })
            global_chunk_id_counter += 1
        all_embeddings.append(chunk_embeddings)

    if not all_text_chunks_with_metadata:
        print("No text chunks were generated from any PDF. Exiting.")
        return

    # Concatenate all embeddings into a single numpy array
    embeddings_np = np.concatenate(all_embeddings, axis=0)
    print(f"\nTotal text chunks processed: {len(all_text_chunks_with_metadata)}")
    print(f"Shape of final embeddings matrix: {embeddings_np.shape}")

    # 6. Build FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embedding_dim)   # Using L2 distance
    # For associating vectors with their original IDs (global_chunk_id)
    # We can use IndexIDMap if we want to store the global_chunk_id directly in FAISS
    # index = faiss.IndexIDMap(index_flat)
    # index.add_with_ids(embeddings_np, np.array(range(len(all_text_chunks_with_metadata))))
    # Simpler: just add vectors. The order in FAISS will match order in all_text_chunks_with_metadata
    index.add(embeddings_np.astype(np.float32)) # FAISS expects float32
    
    print(f"FAISS index built. Total vectors in index: {index.ntotal}")

    # 7. Save FAISS index and text chunks
    print(f"Saving FAISS index to: {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"Saving text chunks metadata to: {TEXT_CHUNKS_PATH}")
    with open(TEXT_CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_text_chunks_with_metadata, f, indent=4)

    print("\n--- Processing Complete ---")
    print(f"FAISS index and text chunks saved successfully to '{VECTOR_STORE_DIR}'.")

if __name__ == "__main__":
    start_time = time.time()
    process_papers_and_build_index()
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")