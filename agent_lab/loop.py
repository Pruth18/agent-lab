"""
The baseline: the same job in plain Python, no agent framework. This is the shape of JobHunter's resume writer.
Find the relevant facts first, ask once, check, retry with the reasons. The model has no tools and no choices.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .common import task
from .validator import check

PROMPT = """You tailor resume bullets to a job description for one candidate.
Write exactly 3 resume bullets, one per line, each starting with "- ".
Use only numbers and skills that appear in the facts below. Never add a number or a skill from the job description
that the candidate doesn't have. If the job asks for something the facts don't show, leave it out.

Facts:
{facts}"""


def text_of(message) -> str:
    content = message.content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content or ""


def run(llm, jd: dict, facts: list[str], index, config=None, max_attempts: int = 3) -> dict:
    found = [d.page_content for d in index.similarity_search(f"{jd['title']}. {jd['text']}", k=6)]
    messages = [SystemMessage(PROMPT.format(facts="\n".join(found))), HumanMessage(task(jd))]
    first_ok, draft, verdict, attempts = None, "", None, 0
    for attempts in range(1, max_attempts + 1):
        reply = llm.invoke(messages, config=config)
        draft = text_of(reply).strip()
        verdict = check(draft, facts)
        first_ok = verdict.ok if first_ok is None else first_ok
        if verdict.ok:
            break
        messages += [AIMessage(draft), HumanMessage("Rejected: " + "; ".join(verdict.problems) + ". Rewrite without them.")]
    return {"draft": draft, "first_ok": first_ok, "final_ok": verdict.ok, "attempts": attempts, "tool_calls": 0, "problems": verdict.problems}
