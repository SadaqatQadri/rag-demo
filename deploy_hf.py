"""
deploy_hf.py — push this bot to a public Hugging Face Space.

    pip install huggingface_hub
    python deploy_hf.py

Reads HF_TOKEN and GROQ_API_KEY from your local .env. Neither value is printed.
Safe to run again — it updates the existing Space rather than making a new one.

The Groq key is uploaded as a Space *secret*, not as a file, so it isn't visible
in the Space's public file listing.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

SPACE_NAME = os.environ.get("HF_SPACE_NAME", "northwind-support-assistant")

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["app.py", "rag_core.py", "loaders.py", "requirements.txt"]

README_TEMPLATE = """---
title: {title}
emoji: 💬
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: {sdk_version}
app_file: app.py
pinned: false
---

# {title}

A document-grounded support assistant. It answers questions using only the files in
`docs/`, cites the document each answer came from, and says it doesn't know rather
than inventing an answer.

Built with Python, ChromaDB for retrieval and Groq for generation. No chatbot builder,
no LangChain — the retrieval pipeline is about 200 lines and every step is inspectable.

The company and documents here are a sample used to demonstrate the system.
"""


def fail(msg):
    print(f"\n{msg}")
    sys.exit(1)


def main():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        fail("huggingface_hub isn't installed. Run:  pip install huggingface_hub")

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        fail("No HF_TOKEN in .env. Add a line: HF_TOKEN=hf_...")
    if not token.startswith("hf_"):
        fail("HF_TOKEN doesn't look right — it should start with 'hf_'.")

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()

    api = HfApi(token=token)

    try:
        user = api.whoami()["name"]
    except Exception as e:
        fail(f"Hugging Face rejected the token ({e}).\n"
             "Make sure it's a WRITE token, not a read-only one.")

    repo_id = f"{user}/{SPACE_NAME}"
    print(f"Deploying to: https://huggingface.co/spaces/{repo_id}")

    try:
        import gradio
        sdk_version = gradio.__version__
    except Exception:
        sdk_version = "5.49.1"

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )
    print("  space ready")

    # Space README carries the config header HF needs. Written next to the project
    # as README_hf.md so your portfolio README.md stays as it is.
    title = os.environ.get("COMPANY_NAME", "Support Assistant") + " — Support Assistant"
    readme_path = os.path.join(HERE, "README_hf.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(README_TEMPLATE.format(title=title, sdk_version=sdk_version))

    api.upload_file(path_or_fileobj=readme_path, path_in_repo="README.md",
                    repo_id=repo_id, repo_type="space")

    for name in FILES:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            fail(f"Missing {name} — run this from inside the project folder.")
        api.upload_file(path_or_fileobj=path, path_in_repo=name,
                        repo_id=repo_id, repo_type="space")
        print(f"  uploaded {name}")

    docs_dir = os.path.join(HERE, "docs")
    if not os.path.isdir(docs_dir):
        fail("No docs/ folder found — there'd be nothing for the bot to answer from.")
    api.upload_folder(folder_path=docs_dir, path_in_repo="docs",
                      repo_id=repo_id, repo_type="space")
    print("  uploaded docs/")

    if groq_key:
        api.add_space_secret(repo_id=repo_id, key="GROQ_API_KEY", value=groq_key)
        print("  set GROQ_API_KEY as a Space secret")
    else:
        print("  NOTE: no GROQ_API_KEY found — the Space will run in retrieval-only mode.")

    for key in ("COMPANY_NAME", "FALLBACK_MESSAGE",
                "MAX_QUESTIONS_PER_SESSION", "MAX_QUESTIONS_PER_DAY", "GROQ_MODEL"):
        val = os.environ.get(key)
        if val:
            api.add_space_variable(repo_id=repo_id, key=key, value=val)
    print("  set config variables")

    print()
    print("Done. It takes 2-4 minutes to build, and the first question is slow")
    print("because it downloads the embedding model once.")
    print()
    print(f"  https://huggingface.co/spaces/{repo_id}")
    print()
    print("If the build fails, open the Space and read the 'Logs' tab — the error")
    print("will be at the bottom.")


if __name__ == "__main__":
    main()
