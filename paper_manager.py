# paper_manager.py
import os
import requests
import time
import arxiv # New import
from typing import Optional, List, Dict

# Directory to save downloaded papers (consistent with other scripts)
PAPERS_DIR = "research_papers"

def sanitize_filename(name: str) -> str:
    """Basic filename sanitization."""
    name = name.replace(' ', '_').replace('/', '_').replace(':', '_')
    return "".join(c for c in name if c.isalnum() or c in ('_', '-')).strip()

def download_pdf(pdf_url: str, title: str, arxiv_id: Optional[str] = None, download_dir: str = PAPERS_DIR) -> Optional[str]:
    """Downloads a single PDF from a given URL."""
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    filename_prefix = sanitize_filename(arxiv_id) if arxiv_id else sanitize_filename(title[:50])
    filepath = os.path.join(download_dir, f"{filename_prefix}.pdf")

    if os.path.exists(filepath):
        print(f"Paper already exists: {filepath}")
        return filepath

    print(f"Downloading: {title} from {pdf_url}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded to: {filepath}")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {title}: {e}")
        return None

def search_arxiv_and_prompt_download(query: str, download_dir: str = PAPERS_DIR, max_results: int = 5) -> Optional[str]:
    """Searches arXiv for a query, prompts user to select, and downloads the paper."""
    print(f"Searching arXiv for: \"{query}\"")
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(search.results())
    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return None

    if not results:
        print("No papers found on arXiv for your query.")
        return None

    print("\nFound the following papers on arXiv:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.title} (ID: {result.get_short_id()})")
        print(f"     Authors: {', '.join(str(a) for a in result.authors[:3])}{' et al.' if len(result.authors) > 3 else ''}")

    while True:
        try:
            choice = input(f"Enter the number of the paper to download (1-{len(results)}), or 'c' to cancel: ")
            if choice.lower() == 'c':
                return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(results):
                selected_paper = results[choice_idx]
                pdf_url = selected_paper.pdf_url
                title = selected_paper.title
                arxiv_id = selected_paper.get_short_id()
                print(f"Selected: {title}")
                return download_pdf(pdf_url, title, arxiv_id, download_dir)
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number or 'c'.")

if __name__ == '__main__':
    # Example usage:
    # query = input("Enter paper title or keywords to search on arXiv: ")
    # downloaded_path = search_arxiv_and_prompt_download(query)
    # if downloaded_path:
    # print(f"Paper ready at: {downloaded_path}")
    
    # Or to download directly if you know the arXiv ID and PDF URL:
    # test_arxiv_id = "1706.03762" # Attention is All You Need
    # paper = next(arxiv.Search(id_list=[test_arxiv_id]).results())
    # download_pdf(paper.pdf_url, paper.title, paper.get_short_id())
    pass