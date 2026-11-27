import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

load_dotenv()

REPEATS = 3            # run every question 3 times
CLOUD_DELAY_S = 3.0    # pause between cloud calls (free-tier rate limit)

# The pipeline under test: same prompt for every model
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a concise technical assistant. Answer using ONLY the provided context. "
     "If the answer is not in the context, reply exactly: I don't know. "
     "Show short working for calculations. Keep answers under 80 words."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

models = {
    "cloud:gpt-oss-20b": ChatGroq(model="openai/gpt-oss-20b", temperature=0, reasoning_effort="low"),
    "local:llama3.2-3b": ChatOllama(model="llama3.2:3b", temperature=0),
}


def timed_run(chain, inputs):
    """Stream one answer and time it (same as Lesson 2)."""
    t0 = time.perf_counter()
    ttft, full = None, None
    for chunk in chain.stream(inputs):
        if ttft is None and chunk.content:
            ttft = time.perf_counter() - t0
        full = chunk if full is None else full + chunk
    return full, ttft, time.perf_counter() - t0


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def p95(values):
    values = sorted(values)
    return values[int(round(0.95 * (len(values) - 1)))]


questions = load_questions("data/questions.jsonl")
rows = []

for name, llm in models.items():
    chain = prompt | llm
    is_cloud = name.startswith("cloud")
    print(f"\n=== {name} ===")
    chain.invoke({"context": "Warm-up.", "question": "Say OK."})   # not measured

    for rep in range(REPEATS):
        for q in questions:
            if is_cloud:
                time.sleep(CLOUD_DELAY_S)

            full, ttft, total = timed_run(chain, {"context": q["context"], "question": q["question"]})

            usage = full.usage_metadata or {}
            out_tokens = usage.get("output_tokens") or max(1, len(full.content) // 4)
            decode = (out_tokens - 1) / (total - ttft) if ttft and total > ttft else None

            rows.append({
                "model": name, "item_id": q["id"], "category": q["category"], "repeat": rep,
                "ttft_s": round(ttft or 0, 4), "total_s": round(total, 4),
                "out_tokens": out_tokens, "decode_tps": round(decode, 2) if decode else "",
                "answer": full.content.strip(),
            })
            print(f"  r{rep} {q['id']:<14} ttft={ttft or 0:5.2f}s  total={total:5.2f}s  "
                  f"tok/s={decode or 0:7.1f}")

# Save every row to CSV
Path("results").mkdir(exist_ok=True)
csv_path = Path("results") / f"run_{datetime.now():%Y%m%d_%H%M%S}.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Summary per model
print(f"\nSaved {len(rows)} rows to {csv_path}\n")
print(f"{'model':<20} {'TTFT med':>9} {'total med':>10} {'total p95':>10} {'tok/s med':>10}")
for name in models:
    r = [x for x in rows if x["model"] == name]
    ttfts = [x["ttft_s"] for x in r]
    totals = [x["total_s"] for x in r]
    tps = [x["decode_tps"] for x in r if x["decode_tps"] != ""]
    print(f"{name:<20} {statistics.median(ttfts):9.2f} {statistics.median(totals):10.2f} "
          f"{p95(totals):10.2f} {statistics.median(tps):10.1f}")
