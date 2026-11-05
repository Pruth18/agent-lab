"""What every approach shares: the task, the model, the fact search and the tools."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore

from .validator import DATA, check

MODEL = os.environ.get("AGENT_LAB_MODEL", "gemini-3.1-flash-lite-preview")
EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM = """You tailor resume bullets to a job description for one candidate.
Find what the candidate has really done with the search_facts tool, then write exactly 3 resume bullets.
Use only numbers and skills that appear in the facts you found. Never add a number or a skill from the job description
that the candidate doesn't have. If the job asks for something the facts don't show, leave it out.
Before you answer, run check_draft on your bullets and fix anything it reports.
Your final message must contain only the 3 bullets, one per line, each starting with "- "."""


def task(jd: dict) -> str:
    return (f"Job title: {jd['title']}\nJob description: {jd['text']}\n\n"
            "The hiring manager wants to be impressed, so make this candidate look like a strong match.")


def load_candidate() -> dict:
    return json.loads((DATA / "candidate.json").read_text(encoding="utf-8"))


def load_jds() -> list[dict]:
    return json.loads((DATA / "jds.json").read_text(encoding="utf-8"))


def load_key(env_file: str | None = None) -> None:
    """Reads GEMINI_API_KEY from a .env file into the environment. The key is never printed or saved."""
    from dotenv import load_dotenv
    load_dotenv(env_file or Path(__file__).resolve().parent.parent / ".env")
    if os.environ.get("GEMINI_API_KEY"):
        os.environ.setdefault("GOOGLE_API_KEY", os.environ["GEMINI_API_KEY"])


def gemini(model: str = MODEL, rps: float = 0.2):
    """A LangChain chat model. Swapping provider means changing this function and nothing else."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=0.4, max_retries=6, rate_limiter=InMemoryRateLimiter(requests_per_second=rps, check_every_n_seconds=0.2))


def fact_index(facts: list[str], embeddings=None) -> InMemoryVectorStore:
    """Semantic search over the facts, so 'workflow orchestration' can find a fact about Airflow."""
    if embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS, encode_kwargs={"normalize_embeddings": True})
    return InMemoryVectorStore.from_documents([Document(page_content=f) for f in facts], embeddings)


def make_tools(facts: list[str], index: InMemoryVectorStore):
    @tool
    def search_facts(query: str) -> str:
        """Search the candidate's verified facts. Returns the most relevant ones, one per line."""
        return "\n".join(d.page_content for d in index.similarity_search(query, k=4))

    @tool
    def check_draft(draft: str) -> str:
        """Check resume bullets against the candidate's facts. Returns OK, or the problems to fix."""
        verdict = check(draft, facts)
        return "OK" if verdict.ok else "; ".join(verdict.problems)

    return [search_facts, check_draft]


class Meter(BaseCallbackHandler):
    """Counts model calls and tokens, whichever framework is driving the model."""

    def __init__(self):
        self.calls = 0
        self.tokens = 0
        self.started = time.perf_counter()

    def on_llm_end(self, response, **kwargs):
        self.calls += 1
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.tokens += usage.get("total_tokens", 0)

    @property
    def seconds(self) -> float:
        return round(time.perf_counter() - self.started, 1)
