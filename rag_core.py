"""
rag_core.py — the actual engine: chunking, embedding/retrieval, and grounded generation.

This is deliberately framework-free (no LangChain/LlamaIndex) so every step is visible
and explainable in a client call. Swap the embedding function or LLM call out for a
managed/enterprise stack later without restructuring anything.

Per-client settings live in .env, so pointing this at a new client is a config change,
not a code change:
    COMPANY_NAME=Acme Ltd
    FALLBACK_MESSAGE=I don't have that in Acme's documents — please email help@acme.com
"""

import os
import re
import glob
from dataclasses import dataclass

import chromadb
from dotenv import load_dotenv

from loaders import load_directory, word_count, pages_equivalent, scope_report

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
STORE_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION_NAME = "client_docs"

# Groq is optional at import time so the retrieval half of this demo still runs
# (and can be screen-recorded) even with no API key set.
try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
COMPANY_NAME = os.environ.get("COMPANY_NAME", "Northwind Gear Co.")
FALLBACK_MESSAGE = os.environ.get(
    "FALLBACK_MESSAGE",
    "I don't have that in the documents I've been given — please contact the support team.",
)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    distance: float


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_long_block(block: str, max_chars: int) -> list[str]:
    """Break an oversized block on sentence boundaries, hard-cutting only as a last resort.

    PDFs and Word exports frequently produce one enormous block with no headings
    and no blank lines. Splitting those mid-sentence is what makes a bot quote
    half a policy, so sentences are tried first.
    """
    pieces, current = [], ""
    for sentence in _SENTENCE_END.split(block):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            pieces.append(current)
        if len(sentence) <= max_chars:
            current = sentence
        else:
            # A single "sentence" longer than the limit — usually a table row or a
            # list that lost its line breaks. Cut on whitespace, never mid-word.
            words, line = sentence.split(), ""
            for w in words:
                if len(line) + len(w) + 1 <= max_chars:
                    line = f"{line} {w}".strip()
                else:
                    pieces.append(line)
                    line = w
            current = line
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, source: str, max_chars: int = 800, overlap: int = 120) -> list[dict]:
    """Split a document into overlapping chunks on the best boundary available.

    Order of preference: markdown headings > blank lines > sentences > whitespace.
    Naive fixed-size chunking (cutting mid-sentence) is the #1 thing that makes a
    RAG bot give garbled answers.
    """
    blocks = re.split(r"\n(?=#{1,6} )|\n\n+", text.strip())

    chunks, current = [], ""
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(current) + len(block) + 2 <= max_chars:
            current = f"{current}\n\n{block}".strip()
        else:
            if current:
                chunks.append(current)
                current = ""
            if len(block) <= max_chars:
                current = block
            else:
                pieces = _split_long_block(block, max_chars)
                chunks.extend(pieces[:-1])
                current = pieces[-1] if pieces else ""
    if current:
        chunks.append(current)

    # A little overlap between consecutive chunks so context isn't lost at boundaries.
    overlapped = []
    for i, c in enumerate(chunks):
        if i > 0:
            tail = chunks[i - 1][-overlap:]
            c = f"{tail}\n{c}"
        overlapped.append({"text": c, "source": source})
    return overlapped


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------

def build_index(docs_dir: str = DOCS_DIR, store_dir: str = STORE_DIR,
                reset: bool = True, verbose: bool = False) -> dict:
    """Load every supported file in docs_dir, chunk it, and embed it into Chroma.

    Returns a summary dict: chunks, files, words, pages, problems.
    Handles .md, .txt, .pdf and .docx. Files that can't be read are reported
    rather than silently skipped.
    """
    documents, problems = load_directory(docs_dir)

    if verbose:
        print(scope_report(documents, problems))

    client = chromadb.PersistentClient(path=store_dir)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(COLLECTION_NAME)

    all_chunks, ids = [], []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["text"], doc["source"])):
            all_chunks.append(chunk)
            ids.append(f"{doc['source']}::{i}")

    if all_chunks:
        # Chroma has a per-call batch ceiling; a 300-page client doc will exceed it.
        BATCH = 500
        for start in range(0, len(all_chunks), BATCH):
            batch = all_chunks[start:start + BATCH]
            collection.add(
                documents=[c["text"] for c in batch],
                metadatas=[{"source": c["source"]} for c in batch],
                ids=ids[start:start + BATCH],
            )

    total_words = sum(d["words"] for d in documents)
    return {
        "chunks": len(all_chunks),
        "files": len(documents),
        "words": total_words,
        "pages": pages_equivalent(total_words),
        "problems": problems,
    }


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def retrieve(query: str, k: int = 4, store_dir: str = STORE_DIR) -> list[RetrievedChunk]:
    client = chromadb.PersistentClient(path=store_dir)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=min(k, collection.count()))
    out = []
    for text, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        out.append(RetrievedChunk(text=text, source=meta["source"], distance=dist))
    return out


# --------------------------------------------------------------------------
# Grounded generation
# --------------------------------------------------------------------------

def build_system_prompt(company: str = COMPANY_NAME, fallback: str = FALLBACK_MESSAGE) -> str:
    return f"""You are a support assistant for {company}. Answer ONLY using the CONTEXT
provided below. If the context does not contain the answer, reply with exactly this and
nothing else: "{fallback}" — do not guess, and do not use outside knowledge.
Always cite which source file(s) you used at the end of your answer, like:
(Source: product_faq.md)."""


def answer(query: str, k: int = 4, model: str = DEFAULT_MODEL) -> dict:
    """Retrieve relevant chunks and generate a grounded answer. Falls back to a
    retrieval-only response if no GROQ_API_KEY is configured, so the demo is
    still runnable/recordable without a live LLM key."""
    chunks = retrieve(query, k=k)
    if not chunks:
        return {
            "answer": "No documents are indexed yet. Run `python ingest.py` first.",
            "chunks": [],
        }

    context = "\n\n---\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key or Groq is None:
        preview = "\n\n".join(
            f"Match {i+1} — from {c.source} (distance {c.distance:.2f}, lower = closer):\n{c.text}"
            for i, c in enumerate(chunks)
        )
        return {
            "answer": (
                "[DEMO MODE — no GROQ_API_KEY set, showing raw retrieval instead of a generated answer]\n\n"
                f"{preview}"
            ),
            "chunks": chunks,
        }

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return {"answer": completion.choices[0].message.content, "chunks": chunks}
