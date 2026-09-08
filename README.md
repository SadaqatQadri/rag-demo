# Northwind Gear Co. — Support Assistant (RAG Demo)

A working example of a **document-grounded Q&A agent**: point it at a company's own
handbook, FAQ, or support docs, and it answers customer/employee questions using
*only* that content — with source citations, and an honest "I don't know" when the
answer isn't in the docs.

This is a portfolio piece, not a real client project — "Northwind Gear Co." and its
docs in `docs/` are fictional, written to look like a realistic small e-commerce
company's internal docs (HR policy, product FAQ, support escalation rules).

## Why this, and not a generic chatbot demo

Anyone can wire a chatbot to an LLM. The part that actually matters to a paying
client — and the part most "I made a chatbot" portfolio pieces skip — is making the
bot answer *only* from their content, refuse to hallucinate outside it, and show
where each answer came from. That's what this demonstrates:

1. **Ingestion** (`ingest.py`) — chunks source docs on paragraph/heading boundaries
   (not blind fixed-size cuts, which is what breaks most weekend RAG demos) and
   embeds them into a persistent vector store.
2. **Retrieval** (`rag_core.retrieve`) — semantic search over those chunks using
   Chroma's built-in embedding model, returning the most relevant passages plus
   their source file.
3. **Grounded generation** (`rag_core.answer`) — the retrieved passages are injected
   into the LLM prompt with an explicit instruction to answer only from context and
   cite sources, so it can't confidently make things up.
4. **UI** (`app.py`) — a Gradio chat interface, good enough to demo live on a call or
   share a public link of.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env        # then add your own Groq API key (console.groq.com/keys)
python ingest.py            # builds the vector index from docs/
python app.py                # launches the chat UI
```

Without a `GROQ_API_KEY` set, `app.py` still runs — it falls back to showing the raw
retrieved passages instead of a generated answer, so the retrieval half of the demo
is recordable/screenshot-able even with zero API cost.

## What this is *not*

Being straight about scope, because a client will ask and it's better they hear it
from the pitch than discover it in production:

- **No auth, rate limiting, or multi-tenant isolation.** Fine for a demo; not fine
  for a live client deployment without adding it.
- **No conversation memory** across turns — each question is answered independently.
  A real support-bot engagement usually needs follow-up-question handling.
- **No eval harness.** There's no automated way here to catch a regression when the
  prompt or chunking changes. For a paid engagement, a client should be shown a
  small test set of Q&A pairs with pass/fail, not just vibes.
- **No hosting.** This runs locally. A deliverable to a real client needs a decision
  on where it's deployed (their infra vs. a simple hosted endpoint) and who pays for
  LLM API usage going forward.
- **Chroma's local, unauthenticated MiniLM embeddings** — fine at this scale;
  a client with tens of thousands of documents needs a different retrieval setup
  (managed vector DB, hybrid keyword+semantic search, re-ranking).

## Related work

Built on the same LLM tool-calling / API-integration pattern used in
[JARVIS](https://github.com/SadaqatQadri/J.A.R.V.I.S), a personal voice assistant
with Groq-based intent routing, function dispatch to real APIs (Spotify, weather,
news), and persistent memory.
