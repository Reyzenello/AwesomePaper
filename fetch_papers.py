import requests
import os
import time
import feedparser

# Configuration
ARXIV_API_URL = "http://export.arxiv.org/api/query?search_query="
QUERY = "machine learning"
RESULTS_PER_CALL = 1
OUTPUT_DIR = "papers/"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_papers(query, start=0, max_results=RESULTS_PER_CALL):
    url = f"{ARXIV_API_URL}{query}&start={start}&max_results={max_results}"
    response = requests.get(url)
    response.raise_for_status()
    return feedparser.parse(response.text)

def save_paper(paper_data, paper_id, title, authors, summary):
    paper_dir = os.path.join(OUTPUT_DIR, paper_id)
    os.makedirs(paper_dir, exist_ok=True)

    # Save paper details
    with open(os.path.join(paper_dir, "details.txt"), "w") as file:
        file.write(f"Title: {title}\n")
        file.write(f"Authors: {', '.join(authors)}\n")
        file.write(f"Summary: {summary}\n")

    # Create a review template
    with open(os.path.join(paper_dir, "review_template.md"), "w") as file:
        file.write(f"# Paper Review\n\n")
        file.write(f"**Title:** {title}\n\n")
        file.write(f"**Authors:** {', '.join(authors)}\n\n")
        file.write(f"**Abstract:**\n{summary}\n\n")
        file.write("## Review\n\n")
        file.write("**Summary:**\n[Write a summary of the paper]\n\n")
        file.write("**Strengths:**\n- Strength 1\n- Strength 2\n\n")
        file.write("**Weaknesses:**\n- Weakness 1\n- Weakness 2\n\n")
        file.write("**Questions:**\n- Question 1\n- Question 2\n\n")
        file.write("**Rating:**\n[Rate the paper out of 5]\n\n")
        file.write("**Comments:**\n[Any additional comments]\n\n")
        file.write("## References\n")
        file.write(f"- [arXiv Link](https://arxiv.org/abs/{paper_id})\n")

def main():
    start = 0
    print(f"Fetching papers {start} to {start + RESULTS_PER_CALL}...")
    feed = fetch_papers(QUERY, start=start)
    for entry in feed.entries:
        paper_id = entry.id.split('/abs/')[-1]
        title = entry.title
        authors = [author.name for author in entry.authors]
        summary = entry.summary
        save_paper(entry, paper_id, title, authors, summary)
    print("Fetched 1 paper")
    # No need to increment start or sleep, as we are fetching only once.

if __name__ == "__main__":
    main()
