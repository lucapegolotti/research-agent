import os
import requests
import time

# List of selected LLM research papers (Title, arXiv ID, PDF URL)
# (Manually curated list focusing on influential papers)
PAPERS_TO_DOWNLOAD = [
    {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf"
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "arxiv_id": "1810.04805",
        "pdf_url": "https://arxiv.org/pdf/1810.04805.pdf"
    },
    {
        "title": "Language Models are Few-Shot Learners (GPT-3)",
        "arxiv_id": "2005.14165",
        "pdf_url": "https://arxiv.org/pdf/2005.14165.pdf"
    },
    {
        "title": "Improving Language Understanding by Generative Pre-Training (GPT-1)",
        "arxiv_id": "cs/0603109",
        "pdf_url": "https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf" 
    },
    {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "arxiv_id": "2201.11903",
        "pdf_url": "https://arxiv.org/pdf/2201.11903.pdf"
    },
    {
        "title": "Training language models to follow instructions with human feedback (InstructGPT)",
        "arxiv_id": "2203.02155",
        "pdf_url": "https://arxiv.org/pdf/2203.02155.pdf"
    },
    {
        "title": "Constitutional AI: Harmlessness from AI Feedback",
        "arxiv_id": "2212.08073",
        "pdf_url": "https://arxiv.org/pdf/2212.08073.pdf"
    },
    {
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "arxiv_id": "2302.13971",
        "pdf_url": "https://arxiv.org/pdf/2302.13971.pdf"
    },
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)",
        "arxiv_id": "2005.11401",
        "pdf_url": "https://arxiv.org/pdf/2005.11401.pdf"
    },
    {
        "title": "Scaling Language Models: Methods, Analysis & Insights from Training Gopher",
        "arxiv_id": "2112.11446",
        "pdf_url": "https://arxiv.org/pdf/2112.11446.pdf"
    },
    {
        "title": "Palm: Scaling language modeling with pathways",
        "arxiv_id": "2204.02311",
        "pdf_url": "https://arxiv.org/pdf/2204.02311.pdf"
    },
    {
        "title": "OPT: Open Pre-trained Transformer Language Models",
        "arxiv_id": "2205.01068",
        "pdf_url": "https://arxiv.org/pdf/2205.01068.pdf"
    },
    {
        "title": "Self-Consistency Improves Chain of Thought Reasoning in Language Models",
        "arxiv_id": "2203.11171",
        "pdf_url": "https://arxiv.org/pdf/2203.11171.pdf"
    },
    {
        "title": "A Survey of Large Language Models",
        "arxiv_id": "2303.18223",
        "pdf_url": "https://arxiv.org/pdf/2303.18223.pdf"
    },
    {
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "arxiv_id": "2302.04761",
        "pdf_url": "https://arxiv.org/pdf/2302.04761.pdf"
    },
    {
        "title": "Sparks of Artificial General Intelligence: Early experiments with GPT-4",
        "arxiv_id": "2303.12712",
        "pdf_url": "https://arxiv.org/pdf/2303.12712.pdf"
    },
    {
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "arxiv_id": "2305.14314",
        "pdf_url": "https://arxiv.org/pdf/2305.14314.pdf"
    },
    {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)",
        "arxiv_id": "2305.18290",
        "pdf_url": "https://arxiv.org/pdf/2305.18290.pdf"
    },
    {
        "title": "The Power of Scale for Parameter-Efficient Prompt Tuning",
        "arxiv_id": "2104.08691",
        "pdf_url": "https://arxiv.org/pdf/2104.08691.pdf"
    },
    {
        "title": "LLaMA 2: Open Foundation and Fine-Tuned Chat Models",
        "arxiv_id": "2307.09288",
        "pdf_url": "https://arxiv.org/pdf/2307.09288.pdf"
    }
]

# Directory to save downloaded papers
DOWNLOAD_DIR = "research_papers"

def download_paper(title: str, arxiv_id: str, pdf_url: str, download_dir: str):
    """Downloads a single paper and saves it."""
    if not os.path.exists(download_dir):
        print(f"Creating directory: {download_dir}")
        os.makedirs(download_dir)

    # Sanitize title for filename or just use arxiv_id
    filename = f"{arxiv_id.replace('/', '_')}_{title.replace(' ', '_').replace(':', '').replace('?', '').replace('/', '_')[:50]}.pdf"
    filepath = os.path.join(download_dir, filename)

    if os.path.exists(filepath):
        print(f"Already downloaded: {title} (ID: {arxiv_id})")
        return

    print(f"Downloading: {title} (ID: {arxiv_id})...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(pdf_url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded: {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {title} (ID: {arxiv_id}): {e}")
    except Exception as e:
        print(f"An unexpected error occurred while downloading {title} (ID: {arxiv_id}): {e}")

if __name__ == "__main__":
    print(f"Starting download of {len(PAPERS_TO_DOWNLOAD)} research papers...")
    start_time = time.time()

    for paper_info in PAPERS_TO_DOWNLOAD:
        download_paper(
            paper_info["title"],
            paper_info["arxiv_id"],
            paper_info["pdf_url"],
            DOWNLOAD_DIR
        )
        time.sleep(1)

    end_time = time.time()
    print(f"\nFinished downloading papers in {end_time - start_time:.2f} seconds.")
    print(f"Papers are saved in the '{DOWNLOAD_DIR}' directory.")