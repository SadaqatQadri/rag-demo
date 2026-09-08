"""
streamlit_app.py — the deployable version of the demo.

app.py (Gradio) still works locally and is the one to run while developing.
This file exists because free hosting for Gradio dried up: Hugging Face moved
Gradio Spaces behind a paid plan, and Streamlit Community Cloud is the remaining
no-card option with enough memory to hold ChromaDB and the embedding model.

Deploy: push this repo to GitHub, then connect it at share.streamlit.io and point
it at streamlit_app.py. Put GROQ_API_KEY in the app's Secrets, not in the repo.
"""

import os
import time

import streamlit as st

# Streamlit Cloud passes secrets via st.secrets, but rag_core reads os.environ.
# Copy them across before importing anything that reads config.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass  # no secrets file locally, which is fine — .env covers it

from rag_core import answer, build_index, STORE_DIR, COMPANY_NAME  # noqa: E402

MAX_PER_SESSION = int(os.environ.get("MAX_QUESTIONS_PER_SESSION", "30"))
MAX_PER_DAY = int(os.environ.get("MAX_QUESTIONS_PER_DAY", "500"))

EXAMPLES = [
    "What's the warranty on the hardshell backpacks?",
    "Can I return a jacket I already wore once?",
    "How many days of PTO do I get per year?",
    "Do you ship to Europe?",
]

st.set_page_config(page_title=f"{COMPANY_NAME} — Support Assistant",
                   page_icon="💬", layout="centered")


@st.cache_resource(show_spinner=False)
def ensure_index():
    """Build the vector store once per app boot, then reuse it across sessions."""
    if not os.path.exists(STORE_DIR) or not os.listdir(STORE_DIR):
        build_index(reset=True)
    return True


@st.cache_resource
def _global_counter():
    """Shared across all visitors in this process — the daily spend ceiling."""
    return {"hits": []}


def rate_limited() -> str | None:
    now = time.time()
    day = _global_counter()
    day["hits"] = [t for t in day["hits"] if t > now - 86400]
    if len(day["hits"]) >= MAX_PER_DAY:
        return ("This demo has hit its daily question limit. It resets within 24 hours — "
                "message me on Fiverr in the meantime and I'll show you it running live.")

    st.session_state.setdefault("hits", [])
    st.session_state.hits = [t for t in st.session_state.hits if t > now - 3600]
    if len(st.session_state.hits) >= MAX_PER_SESSION:
        return (f"You've asked {MAX_PER_SESSION} questions in the last hour, which is the "
                "limit for one visitor. Try again a little later.")

    st.session_state.hits.append(now)
    day["hits"].append(now)
    return None


st.title(f"{COMPANY_NAME} — Support Assistant")
st.caption(
    f"Answers using only {COMPANY_NAME}'s own documents, and cites the file each answer "
    "came from. If something isn't in those documents, it says so instead of guessing. "
    "Northwind is a sample company used to demonstrate the system."
)

with st.spinner("Loading the documents… the first question after a quiet spell takes a minute."):
    ensure_index()

st.session_state.setdefault("messages", [])

if not st.session_state.messages:
    st.write("Try one of these:")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input(f"Ask about {COMPANY_NAME}…") or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        blocked = rate_limited()
        if blocked:
            st.markdown(blocked)
            st.session_state.messages.append({"role": "assistant", "content": blocked})
        else:
            with st.spinner("Searching the documents…"):
                result = answer(prompt)
            sources = ", ".join(sorted({c.source for c in result["chunks"]}))
            body = result["answer"]
            if sources:
                body += f"\n\n---\n*Retrieved from: {sources}*"
            st.markdown(body)
            st.session_state.messages.append({"role": "assistant", "content": body})
