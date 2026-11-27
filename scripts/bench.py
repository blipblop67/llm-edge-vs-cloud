#!/usr/bin/env python3
"""Local vs cloud LLM benchmark: same LangChain pipeline, different models.

Targets are written as provider:model, for example
  groq:openai/gpt-oss-20b    (cloud, free tier)
  ollama:llama3.2:3b         (local, on the Jetson GPU)

Run from the repo root:
  python scripts/bench.py                          # full run
  python scripts/bench.py --limit 2 --repeats 1    # quick test
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

from power import PowerSampler
from quality import keyword_score, make_judge

load_dotenv()

# The pipeline under test: identical for every target.
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a concise technical assistant. Answer using ONLY the provided context. "
     "If the answer is not in the context, reply exactly: I don't know. "
     "Show short working for calculations. Keep answers under 80 words."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

# Paid list prices, USD per 1M tokens (input, output). Check groq.com/pricing.
# On the free tier you are billed $0; this shows what production use would cost.
CLOUD_PRICES = {
    "groq:openai/gpt-oss-20b": (0.075, 0.30),
}

JUDGE_MODEL = "openai/gpt-oss-120b"


def build_llm(target):
    provider, model = target.split(":", 1)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0, reasoning_effort="low", max_retries=3)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=0)
    raise ValueError(f"Unknown provider in {target!r}")


def unload_ollama(model):
    """Free GPU memory before the next model loads (8 GB is shared)."""
    try:
        httpx.post("http://localhost:11434/api/generate",
                   json={"model": model, "keep_alive": 0}, timeout=30)
    except httpx.HTTPError:
        pass


def timed_run(chain, inputs):
    """Stream one answer. Returns (message, t_start, ttft, t_end)."""
    t0 = time.perf_counter()
    ttft, full = None, None
    for chunk in chain.stream(inputs):
        if ttft is None and chunk.content:
            ttft = time.perf_counter() - t0
        full = chunk if full is None else full + chunk
    return full, t0, ttft, time.perf_counter()


def load_questions(path, limit=None):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()][:limit]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+",
                    default=["groq:openai/gpt-oss-20b", "ollama:llama3.2:3b"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--cloud-delay", type=float, default=3.0, help="seconds between cloud calls")
    ap.add_argument("--kwh-price", type=float, default=0.15, help="electricity, USD per kWh")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge scoring")
    args = ap.parse_args()

    questions = load_questions("data/questions.jsonl", args.limit)
    power = PowerSampler().start()
    rows = []

    for target in args.targets:
        is_cloud = not target.startswith("ollama:")
        chain = PROMPT | build_llm(target)
        print(f"\n=== {target} ===")
        chain.invoke({"context": "Warm-up.", "question": "Say OK."})       # not measured

        idle_w = None
        if not is_cloud:
            idle_w = power.idle(5)                                         # baseline, model loaded
            print(f"  idle power: {idle_w:.2f} W")

        for rep in range(args.repeats):
            for q in questions:
                if is_cloud:
                    time.sleep(args.cloud_delay)

                full, t0, ttft, t1 = timed_run(chain, {"context": q["context"],
                                                       "question": q["question"]})
                total = t1 - t0
                usage = full.usage_metadata or {}
                in_tok = usage.get("input_tokens", 0)
                out_tok = usage.get("output_tokens") or max(1, len(full.content) // 4)
                reasoning = usage.get("output_token_details", {}).get("reasoning", 0)

                # Decode speed: Ollama's own GPU timing if available. Otherwise use our
                # timing, but only when tokens arrived over a real stretch of time (Lesson 3 bug).
                meta = full.response_metadata or {}
                if meta.get("eval_duration"):
                    decode = meta["eval_count"] / (meta["eval_duration"] / 1e9)
                elif ttft and total - ttft > 0.05:
                    decode = (out_tok - 1) / (total - ttft)
                else:
                    decode = None

                avg_w = energy = extra = None
                if is_cloud:
                    pin, pout = CLOUD_PRICES.get(target, (None, None))
                    cost = (in_tok * pin + out_tok * pout) / 1e6 if pin is not None else None
                else:
                    avg_w = power.average(t0, t1) or power.read_watts()
                    energy = avg_w * total
                    extra = (avg_w - idle_w) * total
                    cost = energy / 3.6e6 * args.kwh_price

                answer = full.content.strip()
                rows.append({
                    "target": target, "item_id": q["id"], "category": q["category"], "repeat": rep,
                    "ttft_s": ttft, "total_s": total,
                    "in_tokens": in_tok, "out_tokens": out_tok, "reasoning_tokens": reasoning,
                    "decode_tps": decode, "e2e_tps": out_tok / total,
                    "avg_power_w": avg_w, "energy_j": energy, "extra_energy_j": extra,
                    "cost_usd": cost, "keyword_score": keyword_score(answer, q["keywords"]),
                    "judge_score": None, "judge_reason": "", "answer": answer,
                })
                print(f"  r{rep} {q['id']:<14} ttft={ttft or 0:5.2f}s total={total:5.2f}s "
                      f"e2e={out_tok / total:7.1f} tok/s" + (f"  {avg_w:5.1f} W" if avg_w else ""))

        if not is_cloud:
            unload_ollama(target.split(":", 1)[1])

    power.stop()

    # Judge AFTER all timing, so judge traffic can't affect the speed numbers.
    if not args.no_judge:
        from langchain_groq import ChatGroq
        judge = make_judge(ChatGroq(model=JUDGE_MODEL, temperature=0, reasoning_effort="medium"))
        by_id = {q["id"]: q for q in questions}
        print(f"\nJudging first-repeat answers with {JUDGE_MODEL} ...")
        for r in rows:
            if r["repeat"] == 0:
                time.sleep(args.cloud_delay)
                r["judge_score"], r["judge_reason"] = judge(by_id[r["item_id"]], r["answer"])
                print(f"  {r['target']:<26} {r['item_id']:<14} judge={r['judge_score']}")

    Path("results").mkdir(exist_ok=True)
    csv_path = Path("results") / f"run_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {csv_path}")
    print(f"Next: python scripts/report.py {csv_path}")


if __name__ == "__main__":
    main()
