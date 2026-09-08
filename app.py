"""
app.py — Gradio chat UI for the demo. Run:

    python app.py

Then open the local URL it prints.

Rate limiting is on by default. The bot runs on the CLIENT'S API key, so a public
link with no cap is somebody else's bill waiting to happen. Limits are per browser
session and per day, and both are configurable in .env:

    MAX_QUESTIONS_PER_SESSION=30
    MAX_QUESTIONS_PER_DAY=500
"""

import os
import time
from collections import defaultdict

import gradio as gr
from dotenv import load_dotenv

from rag_core import answer, build_index, STORE_DIR, COMPANY_NAME

load_dotenv()

MAX_PER_SESSION = int(os.environ.get("MAX_QUESTIONS_PER_SESSION", "30"))
MAX_PER_DAY = int(os.environ.get("MAX_QUESTIONS_PER_DAY", "500"))

# session id -> list of timestamps; plus a rolling 24h global count
_session_hits = defaultdict(list)
_global_hits = []

SESSION_WINDOW = 60 * 60        # per-session limit resets hourly
DAY_WINDOW = 60 * 60 * 24


def _prune(stamps, window):
    cutoff = time.time() - window
    return [t for t in stamps if t > cutoff]


def _rate_limited(session_id: str):
    """Returns None if allowed, or a message explaining the block."""
    global _global_hits
    now = time.time()

    _global_hits = _prune(_global_hits, DAY_WINDOW)
    if len(_global_hits) >= MAX_PER_DAY:
        return ("This assistant has reached its daily question limit. "
                "It'll reset within 24 hours — please contact the team directly in the meantime.")

    _session_hits[session_id] = _prune(_session_hits[session_id], SESSION_WINDOW)
    if len(_session_hits[session_id]) >= MAX_PER_SESSION:
        return (f"You've asked {MAX_PER_SESSION} questions in the last hour, which is the limit "
                "for one visitor. Please try again a little later.")

    _session_hits[session_id].append(now)
    _global_hits.append(now)
    return None


# Auto-build the index on first launch if it doesn't exist yet.
if not os.path.exists(STORE_DIR) or not os.listdir(STORE_DIR):
    build_index(reset=True)


def respond(message, history, request: gr.Request = None):
    session_id = getattr(request, "session_hash", None) or "anonymous"

    blocked = _rate_limited(session_id)
    if blocked:
        return blocked

    result = answer(message)
    sources = ", ".join(sorted({c.source for c in result["chunks"]})) or "none"
    footer = f"\n\n---\n*Retrieved from: {sources}*" if result["chunks"] else ""
    return result["answer"] + footer


demo = gr.ChatInterface(
    fn=respond,
    title=f"{COMPANY_NAME} — Support Assistant",
    description=(
        f"Answers questions using only {COMPANY_NAME}'s own documents, and cites the file "
        "each answer came from. If something isn't in those documents, it says so rather "
        "than guessing."
    ),
    examples=[
        "How many days of PTO do I get per year?",
        "Can I return a jacket I already wore once?",
        "What's the warranty on the hardshell backpacks?",
        "When does a support ticket get escalated to Tier 3?",
        "Do you ship to Europe?",
    ],
)

if __name__ == "__main__":
    demo.launch()
