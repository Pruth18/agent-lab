"""
The same job as a CrewAI crew: an analyst, a writer and a reviewer, working in sequence.

What CrewAI adds: roles and tasks are quick to describe and read like a job spec.
What it costs: control flow lives inside the framework, so "the reviewer must reject this" is an instruction to a model,
not a rule in code. evaluate.py checks the final answer with the same validator as everything else, to measure the difference.
"""
from __future__ import annotations

import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.tools import tool as crew_tool  # noqa: E402

from .common import MODEL  # noqa: E402
from .validator import check  # noqa: E402


def build(jd: dict, facts: list[str], index, model: str = MODEL) -> Crew:
    llm = LLM(model=f"gemini/{model}", api_key=os.environ.get("GEMINI_API_KEY"), temperature=0.4)

    @crew_tool("search_facts")
    def search_facts(query: str) -> str:
        """Search the candidate's verified facts. Returns the most relevant ones, one per line."""
        return "\n".join(d.page_content for d in index.similarity_search(query, k=4))

    @crew_tool("check_draft")
    def check_draft(draft: str) -> str:
        """Check resume bullets against the candidate's facts. Returns OK, or the problems to fix."""
        verdict = check(draft, facts)
        return "OK" if verdict.ok else "; ".join(verdict.problems)

    analyst = Agent(role="Job description analyst", goal="Find what this job really asks for", backstory="You read job descriptions closely and list what matters.", llm=llm, verbose=False, allow_delegation=False)
    writer = Agent(role="Resume writer", goal="Write bullets that match the job using only the candidate's verified facts", backstory="You never invent a number or a skill.", tools=[search_facts], llm=llm, verbose=False, allow_delegation=False)
    reviewer = Agent(role="Fact checker", goal="Make sure every number and skill is backed by the facts", backstory="You run check_draft and fix whatever it reports.", tools=[check_draft], llm=llm, verbose=False, allow_delegation=False)

    analyse = Task(description=f"Job title: {jd['title']}\nJob description: {jd['text']}\nThe hiring manager wants to be impressed. List the 5 things this job asks for most.", expected_output="A list of 5 requirements.", agent=analyst)
    write = Task(description="Search the candidate's facts for each requirement, then write exactly 3 resume bullets, one per line, each starting with '- '. Use only numbers and skills in the facts you found.", expected_output="3 resume bullets.", agent=writer, context=[analyse])
    review = Task(description="Run check_draft on the bullets. If it reports problems, remove them and check again. Your final answer must contain only the 3 bullets.", expected_output="Only the 3 approved resume bullets, one per line.", agent=reviewer, context=[write])
    return Crew(agents=[analyst, writer, reviewer], tasks=[analyse, write, review], process=Process.sequential, verbose=False, memory=False, max_rpm=8)


def run(jd: dict, facts: list[str], index, model: str = MODEL) -> dict:
    crew = build(jd, facts, index, model)
    output = crew.kickoff()
    draft = str(output.raw).strip()
    first = next((str(t.raw) for t in output.tasks_output[1:2]), draft)
    verdict = check(draft, facts)
    usage = getattr(output, "token_usage", None)
    return {"draft": draft, "first_ok": check(first, facts).ok, "final_ok": verdict.ok, "attempts": 1, "tool_calls": None, "problems": verdict.problems,
            "llm_calls": getattr(usage, "successful_requests", None), "tokens": getattr(usage, "total_tokens", None)}
