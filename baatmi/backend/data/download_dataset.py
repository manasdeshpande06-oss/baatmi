import os
import csv
import zipfile
import urllib.request
import io
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(DATA_DIR, "bbc_news.csv")

DATASET_URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"


def download_dataset():
    """Download the BBC News dataset zip from official source and extract to CSV."""
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 100000:
        print(f"Dataset already exists at {OUTPUT_CSV}")
        return OUTPUT_CSV

    print("Downloading BBC News dataset from official source (http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip)...")
    req = urllib.request.Request(DATASET_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        zip_bytes = resp.read()

    print(f"Downloaded {len(zip_bytes)} bytes. Extracting articles...")
    records = []
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for file_info in z.infolist():
            # Expected pattern: bbc/category/001.txt
            parts = file_info.filename.replace('\\', '/').split('/')
            if len(parts) >= 3 and parts[0] == 'bbc' and parts[2].endswith('.txt'):
                category = parts[1]
                try:
                    content = z.read(file_info).decode('latin-1').strip()
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    if lines:
                        title = lines[0]
                        body = "\n\n".join(lines[1:]) if len(lines) > 1 else lines[0]
                        records.append({
                            "category": category,
                            "title": title,
                            "text": content,
                            "body": body
                        })
                except Exception as e:
                    print(f"Error reading {file_info.filename}: {e}")

    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} articles across categories: {df['category'].value_counts().to_dict()}")
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"Dataset saved to {OUTPUT_CSV}")
    return OUTPUT_CSV


if __name__ == "__main__":
    download_dataset()

