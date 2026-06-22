"""
Combined Nougat + Docling PDF extraction workflow.
Based on: https://ericmjl.github.io/blog/2024/12/20/accurately-extract-text-from-research-literature-pdfs-with-nougat-ocr-and-docling/

Workflow:
  1. Nougat-OCR  -> extracts text, equations (LaTeX), and tables accurately
  2. Docling      -> extracts figures/images as base64-encoded assets
  3. Combine      -> merge into a single JSON per PDF with all components
"""

import os
import sys
import glob
import json
import subprocess
import base64
from pathlib import Path


def run_nougat(pdf_path: str, output_dir: str) -> str | None:
    """Run Nougat-OCR on a PDF. Returns the path to the .mmd output file."""
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(pdf_path).stem

    cmd = ["nougat", pdf_path, "--out", output_dir, "--markdown"]
    print(f"  [Nougat] Processing {pdf_path}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  [Nougat] ERROR: {e.stderr[:500]}")
        return None

    mmd_path = os.path.join(output_dir, f"{stem}.mmd")
    if os.path.exists(mmd_path):
        print(f"  [Nougat] Output -> {mmd_path}")
        return mmd_path
    return None


def run_docling(pdf_path: str, output_dir: str) -> str | None:
    """Run Docling on a PDF. Returns the path to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "docling", pdf_path,
        "--output", output_dir,
        "--image-export-mode", "embedded",
        "--device", "auto",
    ]
    print(f"  [Docling] Processing {pdf_path}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  [Docling] Done.")
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"  [Docling] ERROR: {e.stderr[:500]}")
        return None


def collect_docling_figures(docling_output_dir: str, pdf_stem: str) -> list[dict]:
    """Scan the Docling output directory for extracted figure images."""
    figures = []
    # Docling typically saves figures under a subdirectory
    search_dirs = [
        docling_output_dir,
        os.path.join(docling_output_dir, pdf_stem),
        os.path.join(docling_output_dir, "figures"),
    ]

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            for fname in sorted(files):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    fpath = os.path.join(root, fname)
                    with open(fpath, "rb") as img_file:
                        b64 = base64.b64encode(img_file.read()).decode("utf-8")
                    figures.append({
                        "filename": fname,
                        "path": fpath,
                        "base64": b64,
                        "size_bytes": os.path.getsize(fpath),
                    })
    return figures


def combine_outputs(pdf_path: str, nougat_mmd: str | None, docling_figures: list[dict]) -> dict:
    """Combine Nougat text and Docling figures into a single JSON structure."""
    result = {
        "source_pdf": os.path.basename(pdf_path),
        "nougat_text": None,
        "nougat_pages": [],
        "docling_figures": docling_figures,
        "figure_count": len(docling_figures),
    }

    if nougat_mmd and os.path.exists(nougat_mmd):
        with open(nougat_mmd, "r", encoding="utf-8") as f:
            full_text = f.read()
        result["nougat_text"] = full_text

        # Split by page markers if Nougat inserts them (form-feed or \n\n---\n\n)
        pages = full_text.split("\n\n---\n\n")
        if len(pages) == 1:
            pages = full_text.split("\f")
        result["nougat_pages"] = [p.strip() for p in pages if p.strip()]

    return result


def main():
    pdf_pattern = "COMP5340*.pdf"
    pdf_files = sorted(glob.glob(pdf_pattern))

    if not pdf_files:
        print(f"No files matching '{pdf_pattern}' found in the current directory.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} lecture note PDFs.\n")

    nougat_dir = "nougat_output"
    docling_dir = "docling_output"
    combined_dir = "combined_output"
    os.makedirs(combined_dir, exist_ok=True)

    for pdf_path in pdf_files:
        stem = Path(pdf_path).stem
        print(f"{'='*60}")
        print(f"Processing: {pdf_path}")
        print(f"{'='*60}")

        # Step 1: Nougat for text + equations + tables
        mmd_path = run_nougat(pdf_path, nougat_dir)

        # Step 2: Docling for figures
        docling_out = run_docling(pdf_path, docling_dir)
        figures = []
        if docling_out:
            figures = collect_docling_figures(docling_out, stem)
            print(f"  [Docling] Extracted {len(figures)} figure(s).")

        # Step 3: Combine
        combined = combine_outputs(pdf_path, mmd_path, figures)
        out_json = os.path.join(combined_dir, f"{stem}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"  [Combined] -> {out_json}")
        print()

    print(f"All done. Combined outputs are in '{combined_dir}/'.")


if __name__ == "__main__":
    main()
