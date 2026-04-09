"""
build_rag.py – ETL pipeline: PDF → clean → split → embed → ChromaDB

Usage
-----
# Default (EMBEDDING_MODEL=minilm, CHUNK_SIZE=1000)
python build_rag.py

# Benchmark alternate config via env vars
EMBEDDING_MODEL=mpnet CHUNK_SIZE=2000 python build_rag.py

# Rebuild even if DB already exists
python build_rag.py --force
"""

import os
import re
import argparse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from termcolor import colored

from config import (
    get_embeddings, get_db_folder,
    DATA_FOLDER, FILES,
    ACTIVE_EMBEDDING, CHUNK_SIZE, CHUNK_OVERLAP,
)


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_vector_dbs(
    model_key: str = ACTIVE_EMBEDDING,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    force: bool = False,
) -> None:
    """Build ChromaDB indexes for all documents under data/.

    Indexes are stored at chroma_db/<model_key>/<company>/ so that
    multiple embedding / chunk-size experiments can coexist on disk.
    """
    embeddings = get_embeddings(model_key)
    db_root    = get_db_folder(model_key)

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(colored(f"⚠️  Created empty {DATA_FOLDER}/ directory.", "yellow"))

    # Merge static FILES map with any extra PDFs found in data/
    # Deduplicate by filename so the same PDF isn't indexed twice under different keys.
    all_files = FILES.copy()
    known_filenames = set(FILES.values())
    for fname in os.listdir(DATA_FOLDER):
        if fname.endswith(".pdf") and fname not in known_filenames:
            key = fname.split(".")[0].lower()
            if key not in all_files:
                all_files[key] = fname
                known_filenames.add(fname)
                print(colored(f"✨ Discovered new document: {fname} → key '{key}'", "green"))

    print(colored(
        f"\n📦 Build config — model: {model_key} | "
        f"chunk_size: {chunk_size} | overlap: {chunk_overlap}\n",
        "magenta",
    ))

    for key, filename in all_files.items():
        persist_dir = os.path.join(db_root, key)
        file_path   = os.path.join(DATA_FOLDER, filename)

        if os.path.exists(persist_dir) and not force:
            print(colored(
                f"✅ DB for '{key}' already exists at {persist_dir}. "
                f"Skipping (use --force to rebuild).",
                "yellow",
            ))
            continue

        if not os.path.exists(file_path):
            print(colored(f"❌ Missing source file: {file_path}", "red"))
            continue

        print(colored(f"🔨 Building index for '{key}' …", "cyan"))

        # 1. Load PDF
        loader = PyMuPDFLoader(file_path)
        docs   = loader.load()
        print(f"   Loaded {len(docs)} pages.")

        # 2. Clean text
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        # 3. Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        splits = splitter.split_documents(docs)
        print(f"   Split into {len(splits)} chunks.")

        # 4. Embed & persist
        print("   Embedding and storing … (may take a moment)")
        Chroma.from_documents(splits, embeddings, persist_directory=persist_dir)
        print(colored(f"🎉 Built DB for '{key}' → {persist_dir}", "green"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG vector databases.")
    parser.add_argument(
        "--model", default=ACTIVE_EMBEDDING,
        help="Embedding model key (e.g. minilm, mpnet)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help="Text chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=CHUNK_OVERLAP,
        help="Overlap between consecutive chunks",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if the DB already exists",
    )
    args = parser.parse_args()

    build_vector_dbs(
        model_key=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force=args.force,
    )
