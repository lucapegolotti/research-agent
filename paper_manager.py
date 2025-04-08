# paper_manager.py
import os
import requests
import time
import arxiv  # Existing import
from typing import Optional, List, Dict
from datetime import datetime, timedelta, timezone  # New imports

# Directory to save downloaded papers (consistent with other scripts)
PAPERS_DIR = "research_papers"


def sanitize_filename(name: str) -> str:
    """Basic filename sanitization."""
    name = (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace('"', "")
        .replace("'", "")
    )
    # Keep more characters for readability, but still avoid problematic ones
    name = "".join(c for c in name if c.isalnum() or c in ("_", "-", ".")).strip()
    # Truncate if too long, ensuring it doesn't just become ".pdf"
    return name[:100] if len(name) > 100 else name


def download_pdf(
    pdf_url: str,
    title: str,
    arxiv_id: Optional[str] = None,
    download_dir: str = PAPERS_DIR,
) -> Optional[str]:
    """Downloads a single PDF from a given URL. Returns the filepath if successful."""
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Use arxiv_id for a clean filename if available, otherwise sanitize title
    if arxiv_id:
        base_filename = arxiv_id.replace("/", "_")  # Handles IDs like "cs/0603109"
    else:
        base_filename = sanitize_filename(title)

    filepath = os.path.join(download_dir, f"{base_filename}.pdf")

    if os.path.exists(filepath):
        print(f"Paper already exists: {filepath}")
        return filepath

    print(f"Downloading: '{title}' from {pdf_url}...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}  # Be a good citizen
        response = requests.get(
            pdf_url, stream=True, headers=headers, timeout=60
        )  # Increased timeout
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded to: {filepath}")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"Error downloading '{title}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while downloading '{title}': {e}")
        return None


def search_arxiv_and_prompt_download(
    query: str, download_dir: str = PAPERS_DIR, max_results: int = 5
) -> Optional[Dict[str, str]]:
    """
    Searches arXiv for a query, prompts user to select, and downloads the paper.
    Returns a dictionary with 'filepath', 'title', 'arxiv_id' if successful.
    """
    print(f'Searching arXiv for: "{query}"')
    try:
        search = arxiv.Search(
            query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(search.results())  # Use list() to consume the generator
    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return None

    if not results:
        print("No papers found on arXiv for your query.")
        return None

    print("\nFound the following papers on arXiv:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.title} (ID: {result.get_short_id()})")
        print(
            f"     Authors: {', '.join(str(a) for a in result.authors[:3])}{' et al.' if len(result.authors) > 3 else ''}"
        )
        print(f"     Published: {result.published.strftime('%Y-%m-%d')}")

    while True:
        try:
            choice_input = input(
                f"Enter the number of the paper to download (1-{len(results)}), or 'c' to cancel: "
            )
            if choice_input.lower() == "c":
                return None
            choice_idx = int(choice_input) - 1
            if 0 <= choice_idx < len(results):
                selected_paper = results[choice_idx]
                pdf_url = selected_paper.pdf_url
                title = selected_paper.title
                arxiv_id = selected_paper.get_short_id()
                print(f"Selected: {title}")
                filepath = download_pdf(pdf_url, title, arxiv_id, download_dir)
                if filepath:
                    return {"filepath": filepath, "title": title, "arxiv_id": arxiv_id}
                return None  # Download failed
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number or 'c'.")
        except Exception as e:
            print(f"An error occurred during selection: {e}")
            return None


def fetch_latest_papers_by_query(
    area_query: str,
    num_to_present: int = 7,
    days_recent: int = 90,  # Papers from the last 3 months approx.
    initial_fetch_count: int = 25,  # Fetch more initially to ensure enough after date filtering
) -> List[Dict]:
    """
    Fetches the latest papers from arXiv based on a query, filtered by recency.
    Returns a list of paper metadata dictionaries.
    """
    print(
        f"\nFetching latest papers for area: '{area_query}' (last {days_recent} days)..."
    )
    try:
        search = arxiv.Search(
            query=area_query,
            max_results=initial_fetch_count,  # Fetch more to filter by date
            sort_by=arxiv.SortCriterion.SubmittedDate,  # Get newest first
        )

        all_results = list(search.results())  # Consume generator
        if not all_results:
            print("No results found on arXiv for this query.")
            return []

    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return []

    recent_papers_metadata = []
    # Current date is Friday, May 23, 2025.
    # We need to use the actual current date when the script runs.
    date_threshold = datetime.now(timezone.utc) - timedelta(days=days_recent)

    for result in all_results:
        # Ensure published date is timezone-aware for correct comparison
        published_date_aware = result.published
        if published_date_aware.tzinfo is None:
            published_date_aware = published_date_aware.replace(tzinfo=timezone.utc)

        if published_date_aware >= date_threshold:
            paper_info = {
                "title": result.title,
                "arxiv_id": result.get_short_id(),
                "summary": result.summary,
                "authors": [str(a) for a in result.authors],
                "pdf_url": result.pdf_url,
                "published_date_iso": result.published.isoformat(),  # Store as ISO string
                "published_date_obj": result.published,  # Keep datetime object for potential use
            }
            recent_papers_metadata.append(paper_info)
            if len(recent_papers_metadata) >= num_to_present:
                break  # Stop once we have enough recent papers to present
        # else:
        # Since results are sorted by newest, we could potentially break early
        # if result.published is significantly older than threshold.
        # However, submitted vs published date can vary, so scanning a bit more is safer.
        # For simplicity, we scan through `initial_fetch_count` results.

    if not recent_papers_metadata:
        print(
            f"No papers found published/updated in the last {days_recent} days matching your query."
        )

    return recent_papers_metadata
