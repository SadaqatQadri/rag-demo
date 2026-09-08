"""
evaluate.py — measure whether the bot is actually right, instead of eyeballing it.

    python evaluate.py

Runs every question in eval_questions.json through the real pipeline and scores it:

  - questions with "expect": the answer must contain those substrings
  - questions with "should_refuse": true: the answer must decline rather than invent

Run this after any change to chunking, the prompt, k, or the model. A change that
"feels better" and quietly drops you from 92% to 76% is exactly what this catches.

Sell the output of this, too: a client who sees a measured score trusts the build
more than one who is told "it works well".
"""

import json
import os
import re
import sys
import time

from rag_core import answer, FALLBACK_MESSAGE

# Phrases that indicate the bot correctly declined instead of inventing something.
REFUSAL_MARKERS = [
    FALLBACK_MESSAGE.lower()[:40],
    "i don't have", "i do not have", "not in the document", "isn't in the document",
    "no information", "not covered", "don't have that", "unable to find",
    "not specified", "doesn't mention", "does not mention", "contact",
]


def looks_like_refusal(text: str) -> bool:
    t = text.lower()
    return any(m and m in t for m in REFUSAL_MARKERS)


def check(item, response: str):
    text = response.lower()
    if item.get("should_refuse"):
        return (looks_like_refusal(text), "declined as expected" if looks_like_refusal(text)
                else "INVENTED an answer for something not in the docs")
    missing = [e for e in item.get("expect", []) if e.lower() not in text]
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, "correct"


def main(path="eval_questions.json"):
    if not os.path.exists(path):
        print(f"No {path} found.")
        sys.exit(1)

    data = json.load(open(path, encoding="utf-8"))
    items = data["questions"]

    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: no GROQ_API_KEY set — the bot is in retrieval-only demo mode,")
        print("so these scores measure retrieval, not the final answers.\n")

    passed, failures = 0, []
    for i, item in enumerate(items, 1):
        response = answer(item["q"])["answer"]
        ok, note = check(item, response)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {i:>2}. {item['q']}")
        if not ok:
            print(f"        -> {note}")
            failures.append((item["q"], note, response))
        time.sleep(0.3)          # be polite to the API

    total = len(items)
    pct = 100 * passed / total if total else 0
    print()
    print(f"Score: {passed}/{total}  ({pct:.0f}%)")
    refused = sum(1 for i in items if i.get("should_refuse"))
    print(f"({refused} of {total} questions were 'should refuse' checks)")

    if pct < 85:
        print()
        print("Below 85%. Don't ship this to a client yet — look at the failures above.")
        print("Usual culprits: chunks too small to hold a whole policy, k too low,")
        print("or the source document genuinely not containing the answer.")

    return failures


if __name__ == "__main__":
    main()
