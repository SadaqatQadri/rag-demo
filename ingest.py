"""
ingest.py — run this whenever the client's source documents change.

    python ingest.py

Reads every .md, .txt, .pdf and .docx file in docs/, chunks it, embeds it, and
rebuilds the vector store in chroma_store/.

It also prints a scope report: words per file and the page equivalent. Run this
BEFORE quoting a client so you're agreeing on a number you've actually measured,
not a number they guessed.
"""

from rag_core import build_index
from loaders import WORDS_PER_PAGE

if __name__ == "__main__":
    print("Reading docs/ ...\n")
    result = build_index(reset=True, verbose=True)

    print()
    print(f"Indexed {result['chunks']} chunks from {result['files']} file(s).")
    print(f"Scope: {result['words']:,} words = ~{result['pages']} pages "
          f"(at {WORDS_PER_PAGE} words per page).")

    if result["problems"]:
        print()
        print(f"{len(result['problems'])} file(s) could not be read:")
        for p in result["problems"]:
            print(f"  - {p['source']}: {p['error']}")
        print()
        print("Sort these out with the client before you start the delivery clock.")
