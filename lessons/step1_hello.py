from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

load_dotenv()  # reads GROQ_API_KEY from .env

# 1. The prompt: same for every model
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise technical assistant. Answer in one sentence."),
    ("human", "{question}"),
])

# 2. The models: one cloud, one local
models = {
    "cloud": ChatGroq(model="openai/gpt-oss-20b", temperature=0, reasoning_effort="low"),
    "local": ChatOllama(model="llama3.2:3b", temperature=0),
}

question = "Why do edge devices use quantized models?"

# 3. The pipeline: prompt | model. Only the model changes.
for name, llm in models.items():
    chain = prompt | llm
    reply = chain.invoke({"question": question})
    print(f"[{name}] {reply.content}")
    print(f"        tokens: {reply.usage_metadata}\n")
