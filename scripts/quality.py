"""Answer quality: keyword check + LLM-as-judge with structured output."""
import unicodedata

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


def keyword_score(answer, keywords):
    """Fraction of required keywords found in the answer (0.0 to 1.0)."""
    a = unicodedata.normalize("NFKC", answer)        # \u202f -> normal space, etc.
    a = a.lower().replace("\u2019", "'")             # curly apostrophe -> straight
    return sum(k.lower() in a for k in keywords) / len(keywords)


class Grade(BaseModel):
    """The shape of the judge's reply."""
    score: int = Field(ge=1, le=5, description="1 = wrong or invented, 5 = fully correct")
    reason: str = Field(description="one short sentence explaining the score")


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict grader. Compare the candidate answer to the reference answer.\n"
     "5 = fully correct and complete\n"
     "4 = correct with minor omissions\n"
     "3 = partially correct\n"
     "2 = mostly wrong\n"
     "1 = wrong, or invents facts not in the context\n"
     "If the reference says \"I don't know\", the candidate must also decline to answer."),
    ("human",
     "Context:\n{context}\n\nQuestion: {question}\n\n"
     "Reference answer: {reference}\n\nCandidate answer: {answer}"),
])


def make_judge(llm):
    """Build a judge function from any chat model."""
    chain = JUDGE_PROMPT | llm.with_structured_output(Grade, method="json_schema")

    def judge(question, answer):
        try:
            g = chain.invoke({
                "context": question["context"], "question": question["question"],
                "reference": question["reference"], "answer": answer,
            })
            return g.score, g.reason
        except Exception as e:          # rate limit, bad output, network...
            return None, f"judge error: {type(e).__name__}: {str(e)[:300]}"

    return judge
