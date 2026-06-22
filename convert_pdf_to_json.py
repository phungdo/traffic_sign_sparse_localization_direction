import os
import sys
import glob
import json
import pymupdf4llm

def main():
    pdf_files = glob.glob("COMP5340*.pdf")
    if not pdf_files:
        print("No COMP5340 PDF files found.")
        return

    for pdf_path in pdf_files:
        try:
            print(f"Processing {pdf_path}...")
            # Extract as page chunks (list of dicts with 'metadata' and 'text')
            md_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
            
            # Save as JSON
            json_path = pdf_path.replace('.pdf', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(md_chunks, f, ensure_ascii=False, indent=2)
                
            print(f"Saved {json_path}")
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    main()
