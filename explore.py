from rag_core import chunk_text
from rag_core import retrieve

with open("docs/product_faq.md", "r", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_text(text, "product_faq.md")
print(f"product_faq.md got split into {len(chunks)} chunks\n")

for i, c in enumerate(chunks):
    print(f"--- Chunk {i} ({len(c['text'])} characters) ---")
    print(c["text"])
    print()

results = retrieve("What's the warranty on the hardshell backpacks?")
print("\n=== Retrieval results ===")
for r in results:
    print(f"\nfrom {r.source} — distance {r.distance:.3f}")
    print(r.text)
    print("Contains 'warranty'? ", "warranty" in r.text.lower())
