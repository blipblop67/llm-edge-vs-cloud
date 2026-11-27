import time
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise technical assistant. Answer in about 100 words."),
    ("human", "{question}"),
])

models = {
    "cloud": ChatGroq(model="openai/gpt-oss-20b", temperature=0, reasoning_effort="low"),
    "local": ChatOllama(model="llama3.2:3b", temperature=0),
}


def timed_run(chain, inputs):
    """Stream one answer and time it."""
    t0 = time.perf_counter()          # high-resolution clock for timing
    ttft = None
    full = None
    for chunk in chain.stream(inputs):
        if ttft is None and chunk.content:
            ttft = time.perf_counter() - t0                  # first visible text
        full = chunk if full is None else full + chunk       # add up the chunks
    total = time.perf_counter() - t0
    return full, ttft, total


question = "Why do edge devices like the Jetson use quantized models?"

for name, llm in models.items():
    chain = prompt | llm
    chain.invoke({"question": "Say OK."})     # warm-up, not measured

    full, ttft, total = timed_run(chain, {"question": question})

    usage = full.usage_metadata or {}
    out_tokens = usage.get("output_tokens", 0)
    reasoning = usage.get("output_token_details", {}).get("reasoning", 0)
    estimated = False
    if not out_tokens:                         # some providers skip usage when streaming
        out_tokens, estimated = len(full.content) // 4, True

    decode_tps = (out_tokens - 1) / (total - ttft)

    print(f"[{name}]")
    print(f"  TTFT        : {ttft:.2f} s")
    print(f"  total       : {total:.2f} s")
    print(f"  out tokens  : {out_tokens}{' (estimated)' if estimated else ''}"
          f"{f'  (incl. {reasoning} reasoning)' if reasoning else ''}")
    print(f"  decode tok/s: {decode_tps:.1f}")

    # Ollama reports its own GPU timing. Use it to check our number.
    meta = full.response_metadata or {}
    if meta.get("eval_duration"):
        print(f"  ollama tok/s: {meta['eval_count'] / (meta['eval_duration'] / 1e9):.1f}")
    print()
