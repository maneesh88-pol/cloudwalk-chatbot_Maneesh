# Developed by Maneesh Polavarapu for Nimbus Challenge
# Contact: polavarapumaneesh.info@gmail.com

import os
import yaml
import requests
import pathlib
from bs4 import BeautifulSoup
from rag_pipeline import load_env, text_splitter, get_or_create_db

load_env()

ROOT = pathlib.Path(__file__).parent
DATA_DIR = ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines() if ln.strip()]
    return "\n".join(lines)

def main():
    src_path = ROOT / "data" / "sources.yml"
    with open(src_path, "r", encoding="utf-8") as f:
        sources = yaml.safe_load(f)["sources"]

    db = get_or_create_db()

    all_texts, metadatas = [], []
    for s in sources:
        url = s["url"]
        label = s.get("label", url)
        print(f"Fetching: {label} ...")
        try:
            html = requests.get(url, timeout=20).text
            text = clean_html(html)
            fname = (label.replace(' ', '_') + '.txt')[:80]
            (DATA_DIR / fname).write_text(text, encoding="utf-8")

            chunks = text_splitter.split_text(text)
            all_texts.extend(chunks)
            metadatas.extend([{"source": url, "label": label}] * len(chunks))
        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")

    db.add_texts(all_texts, metadatas)
    print("✅ Ingestion complete! Vector DB saved at ./vectorstore/chroma")

if __name__ == "__main__":
    main()
