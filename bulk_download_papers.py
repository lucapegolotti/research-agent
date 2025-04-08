# bulk_download_papers.py
import json
import time
import arxiv  # Make sure this is installed
import paper_manager  # Your refactored paper downloading module

PAPERS_JSON_PATH = "initial_papers.json"


def download_papers_from_json(json_path: str):
    """
    Reads a JSON file containing a list of papers and downloads them
    using functions from paper_manager.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            papers_to_download = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}.")
        return

    print(f"Found {len(papers_to_download)} papers to process from {json_path}.")
    downloaded_count = 0
    failed_count = 0

    for i, paper_info in enumerate(papers_to_download):
        print(
            f"\nProcessing paper {i+1}/{len(papers_to_download)}: {paper_info.get('title', 'N/A')}"
        )
        title = paper_info.get("title", "Untitled Paper")
        arxiv_id = paper_info.get("arxiv_id")
        pdf_url = paper_info.get("pdf_url")
        file_prefix = paper_info.get("file_prefix")  # Used if no arXiv ID for filename

        downloaded_path = None

        if arxiv_id:
            try:
                # Fetch paper details from arXiv using the ID
                search = arxiv.Search(id_list=[arxiv_id])
                paper_result = next(search.results(), None)
                if paper_result:
                    print(
                        f"  Found on arXiv: {paper_result.title} (ID: {paper_result.get_short_id()})"
                    )
                    downloaded_path = paper_manager.download_pdf(
                        pdf_url=paper_result.pdf_url,
                        title=paper_result.title,  # Use title from arXiv for consistency
                        arxiv_id=paper_result.get_short_id(),  # Use ID from arXiv
                    )
                else:
                    print(f"  Could not find paper with arXiv ID: {arxiv_id} on arXiv.")
                    failed_count += 1
            except Exception as e:
                print(f"  Error fetching or downloading arXiv paper {arxiv_id}: {e}")
                failed_count += 1
        elif pdf_url:
            # Download directly if a PDF URL is provided
            print(f"  Attempting direct download from URL: {pdf_url}")
            downloaded_path = paper_manager.download_pdf(
                pdf_url=pdf_url,
                title=title,  # Use title from JSON
                arxiv_id=file_prefix,  # Use file_prefix as a stand-in for arxiv_id for filename consistency if needed
            )
            if not downloaded_path:
                failed_count += 1
        else:
            print(
                f"  Skipping paper '{title}': No arXiv ID or direct PDF URL provided."
            )
            failed_count += 1
            continue  # Skip to the next paper

        if downloaded_path:
            downloaded_count += 1

        time.sleep(1)  # Be polite to servers

    print(f"\n--- Bulk Download Summary ---")
    print(f"Successfully downloaded: {downloaded_count} paper(s).")
    print(f"Failed or skipped: {failed_count} paper(s).")
    print(f"Papers should be in the '{paper_manager.PAPERS_DIR}' directory.")


if __name__ == "__main__":
    print("Starting bulk download of initial papers...")
    download_papers_from_json(PAPERS_JSON_PATH)
