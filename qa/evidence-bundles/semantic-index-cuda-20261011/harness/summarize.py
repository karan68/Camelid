"""Print the README's search and chat tables from data/."""
import json
import pathlib

data = pathlib.Path(__file__).resolve().parent.parent / "data"
runs = [
    ("main, library not indexed", "search/main-not-indexed.json"),
    ("this PR, indexed, before the tie-break", "search/indexed-before-tie-break.json"),
    ("this PR, indexed (final)", "search/indexed-final.json"),
]
names = {
    "collection_default": "Collection, default (hybrid once indexed)",
    "collection_semantic": "Collection, meaning only",
    "collection_keyword": "Collection, keyword only",
    "library": "Whole library",
}
loaded = [(label, json.loads((data / path).read_text())) for label, path in runs]
print("| Search, 50 questions | " + " | ".join(label for label, _ in loaded) + " |")
print("|---|" + "---|" * len(loaded))
for key, name in names.items():
    cells = [f"first {r[key]['first']}, top 5 {r[key]['top5']}, decoy first {r[key]['decoy_first']}" for _, r in loaded]
    print(f"| {name} | " + " | ".join(cells) + " |")
print()
print("| Search time, p50 | " + " | ".join(label for label, _ in loaded) + " |")
print("|---|" + "---|" * len(loaded))
for key, name in names.items():
    print(f"| {name} | " + " | ".join(f"{r[key]['ms_p50']:.0f} ms" for _, r in loaded) + " |")
chat = json.loads((data / "chat-hybrid-final.json").read_text())
print()
print({k: chat[k] for k in ("model", "questions", "states_right_number", "states_decoy_number", "right_and_cited_passage_resolves_to_it", "chat_ms_p50")})
