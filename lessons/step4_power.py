import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from power import PowerSampler

KWH_PRICE_USD = 0.15   # change to your local electricity tariff

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise technical assistant."),
    ("human", "{question}"),
])
chain = prompt | ChatOllama(model="llama3.2:3b", temperature=0)

power = PowerSampler().start()
chain.invoke({"question": "Say OK."})               # warm-up: model is now loaded

print("Measuring idle power for 5 s...")
idle_w = power.idle(5)

t0 = time.perf_counter()
reply = chain.invoke({"question": "Explain in about 150 words how a Jetson runs a quantized LLM."})
t1 = time.perf_counter()
power.stop()

seconds = t1 - t0
tokens = reply.usage_metadata["output_tokens"]
avg_w = power.average(t0, t1)
peak_w = max(power.window(t0, t1))
energy_j = avg_w * seconds                          # joules = watts x seconds
extra_j = (avg_w - idle_w) * seconds                # energy the model itself added

print(f"idle power      : {idle_w:.2f} W")
print(f"during answer   : {avg_w:.2f} W avg, {peak_w:.2f} W peak")
print(f"answer          : {tokens} tokens in {seconds:.2f} s")
print(f"energy          : {energy_j:.1f} J total, {extra_j:.1f} J above idle")
print(f"energy per token: {energy_j / tokens:.3f} J")
print(f"cost / 1k answers at ${KWH_PRICE_USD}/kWh: ${energy_j * 1000 / 3.6e6 * KWH_PRICE_USD:.4f}")
