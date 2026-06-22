import sys
import pymupdf4llm

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_md.py <pdf_file>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        out_path = pdf_path.replace('.pdf', '.md')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(md_text)
        print(f"Successfully extracted {pdf_path} to {out_path}")
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")

if __name__ == "__main__":
    main()
