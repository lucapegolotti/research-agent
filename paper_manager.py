# paper_manager.py
import os
import re # For sanitizing and word extraction
import requests
import time
import arxiv
from typing import Optional, List, Dict
from datetime import datetime, timedelta, timezone

PAPERS_DIR = "research_papers"

def generate_filename_from_title(title: str, download_dir: str, ensure_unique: bool = True) -> str:
    """
    Generates a sanitized filename from the first 3-5 words of a title.
    Handles potential collisions by appending a number if ensure_unique is True.
    Returns only the filename string (e.g., "attention-is-all.pdf").
    """
    if not title:
        base_name_str = "untitled-paper"
    else:
        # Extract words, convert to lowercase
        words = re.findall(r'\b\w+\b', title.lower())
        # Take first 3 to 5 words, or fewer if title is shorter
        num_words_for_filename = min(max(3, len(words)), 5) # Use 3-5 words
        base_name_parts = words[:num_words_for_filename]
        
        if not base_name_parts: # If title had no usable words
            base_name_str = "untitled-paper"
        else:
            base_name_str = "-".join(base_name_parts)

    # Further sanitize: remove non-alphanumeric/hyphen, ensure single hyphens
    base_name_str = re.sub(r'[^\w\s-]', '', base_name_str).strip().lower()
    base_name_str = re.sub(r'\s+', '-', base_name_str) 
    base_name_str = re.sub(r'-+', '-', base_name_str) 

    if not base_name_str: # If sanitization resulted in empty string
        base_name_str = "untitled-paper"
        
    filename_candidate = f"{base_name_str}.pdf"

    if ensure_unique:
        counter = 0
        final_filename = filename_candidate
        # Check for collision in the target directory
        while os.path.exists(os.path.join(download_dir, final_filename)):
            counter += 1
            final_filename = f"{base_name_str}-{counter}.pdf"
            if counter > 100: # Safety break for extreme cases
                import uuid
                unique_id = uuid.uuid4().hex[:6]
                final_filename = f"{base_name_str}-{unique_id}.pdf"
                break 
        return final_filename
    else:
        return filename_candidate


def download_pdf(pdf_url: str, title: str, arxiv_id: Optional[str] = None, download_dir: str = PAPERS_DIR) -> Optional[str]:
    """
    Downloads a single PDF from a given URL, using a title-based filename.
    Returns the full filepath if successful.
    The arxiv_id is now primarily for metadata/logging if needed, not for the filename itself unless title is absent.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Generate the filename using the new title-based logic, ensuring uniqueness for new downloads.
    # If a file with the exact same title (and thus base filename) is requested again,
    # the uniqueness handler will try to create name-1.pdf, name-2.pdf etc.
    # The check for *already processed* paper should happen in research_agent.py before calling this,
    # based on whether the paper (identified by title/arxiv_id) is already in its metadata.
    # This function focuses on downloading to a unique, title-based filename.

    filename_to_save = generate_filename_from_title(title if title else arxiv_id, download_dir, ensure_unique=True)
    filepath = os.path.join(download_dir, filename_to_save)

    # If, after generating a unique name, that specific file somehow exists (e.g., from a partial previous download)
    # but isn't in our vector store, we might overwrite or skip.
    # For simplicity here, we'll attempt to download if the agent decided this paper needs to be added.
    # The agent should ideally check if this *content* (via arxiv_id or title) is already in its store
    # before initiating a download. Our `_process_and_integrate_paper` has a check.

    print(f"Attempting to download: '{title}' as '{filename_to_save}' from {pdf_url}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, stream=True, headers=headers, timeout=60)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded to: {filepath}")
        return filepath # Return the actual path it was saved to
    except requests.exceptions.RequestException as e:
        print(f"Error downloading '{title}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while downloading '{title}': {e}")
        return None

# --- search_arxiv_and_prompt_download and fetch_latest_papers_by_query remain the same ---
# They will now call the modified download_pdf, which uses the new naming convention.

def search_arxiv_and_prompt_download(query: str, download_dir: str = PAPERS_DIR, max_results: int = 5) -> Optional[Dict[str, str]]:
    """
    Searches arXiv for a query, prompts user to select, and downloads the paper.
    Returns a dictionary with 'filepath', 'title', 'arxiv_id' if successful.
    (This function's internal logic remains the same, but it calls the updated download_pdf)
    """
    print(f"Searching arXiv for: \"{query}\"")
    try:
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        results = list(search.results())
    except Exception as e: print(f"Error searching arXiv: {e}"); return None
    if not results: print("No papers found on arXiv for your query."); return None

    print("\nFound the following papers on arXiv:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.title} (ID: {result.get_short_id()})")
        print(f"     Authors: {', '.join(str(a) for a in result.authors[:3])}{' et al.' if len(result.authors) > 3 else ''}")
        print(f"     Published: {result.published.strftime('%Y-%m-%d')}")

    while True:
        try:
            choice_input = input(f"Enter the number of the paper to download (1-{len(results)}), or 'c' to cancel: ")
            if choice_input.lower() == 'c': return None
            choice_idx = int(choice_input) - 1
            if 0 <= choice_idx < len(results):
                selected_paper = results[choice_idx]
                print(f"Selected: {selected_paper.title}")
                # download_pdf now handles title-based naming
                filepath = download_pdf(selected_paper.pdf_url, selected_paper.title, selected_paper.get_short_id(), download_dir)
                if filepath:
                    return {"filepath": filepath, "title": selected_paper.title, "arxiv_id": selected_paper.get_short_id()}
                return None 
            else: print("Invalid choice.")
        except ValueError: print("Invalid input. Please enter a number or 'c'.")
        except Exception as e: print(f"An error occurred: {e}"); return None


def fetch_latest_papers_by_query(
    area_query: str, num_to_present: int = 7, days_recent: int = 90, initial_fetch_count: int = 25
) -> List[Dict]:
    """
    Fetches the latest papers from arXiv based on a query, filtered by recency.
    (This function's internal logic remains the same)
    """
    print(f"\nFetching latest papers for area: '{area_query}' (last {days_recent} days)...")
    try:
        search = arxiv.Search(query=area_query, max_results=initial_fetch_count, sort_by=arxiv.SortCriterion.SubmittedDate)
        all_results = list(search.results())
        if not all_results: print("No results on arXiv for this query."); return []
    except Exception as e: print(f"Error searching arXiv: {e}"); return []

    recent_papers_metadata = []
    date_threshold = datetime.now(timezone.utc) - timedelta(days=days_recent)
    for result in all_results:
        published_date_aware = result.published.replace(tzinfo=result.published.tzinfo or timezone.utc) # Ensure timezone
        if published_date_aware >= date_threshold:
            paper_info = {
                "title": result.title, "arxiv_id": result.get_short_id(), "summary": result.summary,
                "authors": [str(a) for a in result.authors], "pdf_url": result.pdf_url,
                "published_date_iso": result.published.isoformat(), "published_date_obj": result.published
            }
            recent_papers_metadata.append(paper_info)
            if len(recent_papers_metadata) >= num_to_present: break
    if not recent_papers_metadata: print(f"No papers in last {days_recent} days for your query.")
    return recent_papers_metadata


if __name__ == '__main__':
    # Example test for generate_filename_from_title
    # test_dir = "test_paper_downloads"
    # if not os.path.exists(test_dir): os.makedirs(test_dir)
    # title1 = "Attention Is All You Need"
    # title2 = "Attention Is All You Really Need" # Potential collision base
    # title3 = "Attention Is All: A New Perspective" # Potential collision base
    # title4 = "A Very Long Title With More Than Five Words For Testing Length"
    # title5 = ""
    # title6 = "!@#$%^&*()"

    # print(f"'{title1}' -> '{generate_filename_from_title(title1, test_dir)}'")
    # # Simulate downloading to make files for collision test
    # open(os.path.join(test_dir, generate_filename_from_title(title1, test_dir, ensure_unique=False)), 'w').close()
    # print(f"'{title2}' -> '{generate_filename_from_title(title2, test_dir)}'")
    # open(os.path.join(test_dir, generate_filename_from_title(title2, test_dir, ensure_unique=False).replace(".pdf","-1.pdf")),'w').close() # Simulate collision
    # print(f"'{title3}' -> '{generate_filename_from_title(title3, test_dir)}'") # Should become name-2.pdf if name.pdf and name-1.pdf exist
    # print(f"'{title4}' -> '{generate_filename_from_title(title4, test_dir)}'")
    # print(f"'{title5}' (empty) -> '{generate_filename_from_title(title5, test_dir)}'")
    # print(f"'{title6}' (symbols) -> '{generate_filename_from_title(title6, test_dir)}'")
    pass