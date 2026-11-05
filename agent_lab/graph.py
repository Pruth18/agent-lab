"""
The same job as a LangGraph agent.

    START -> agent <-> tools
               |
               v
             guard -> agent        rejected: back to the model with the reasons
               |
               +----> human -> END a person approves or rejects
               |
               +----> END          out of attempts: no draft goes out

What LangGraph adds over the plain loop:
  - The model chooses when to search and when to check (tool calling).
  - `guard` is a node, not a suggestion. The model's own check_draft call is advice; this one can't be skipped.
  - `interrupt` pauses the run for a person, and the checkpointer lets it resume later, from another process if needed.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from .common import SYSTEM, make_tools, task
from .loop import text_of
from .validator import check

MAX_ATTEMPTS = 3


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    attempts: int
    first_ok: bool | None
    draft: str
    problems: list[str]
    approved: bool | None


def build(llm, facts: list[str], index, checkpointer=None):
    tools = make_tools(facts, index)
    model = llm.bind_tools(tools)

    def agent(state: State):
        return {"messages": [model.invoke(state["messages"])]}

    def guard(state: State):
        draft = text_of(state["messages"][-1]).strip()
        verdict = check(draft, facts)
        update = {"draft": draft, "attempts": state["attempts"] + 1, "problems": verdict.problems}
        if state["first_ok"] is None:
            update["first_ok"] = verdict.ok
        if not verdict.ok:
            update["messages"] = [HumanMessage("Rejected: " + "; ".join(verdict.problems) + ". Rewrite without them.")]
        return update

    def human(state: State):
        decision = interrupt({"draft": state["draft"], "question": "Send this to the candidate?"})
        return {"approved": decision == "approve"}

    def after_agent(state: State):
        return "tools" if state["messages"][-1].tool_calls else "guard"

    def after_guard(state: State):
        if not state["problems"]:
            return "human"
        return "agent" if state["attempts"] < MAX_ATTEMPTS else END

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_node("guard", guard)
    g.add_node("human", human)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", "guard"])
    g.add_edge("tools", "agent")
    g.add_conditional_edges("guard", after_guard, ["agent", "human", END])
    g.add_edge("human", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def start_state(jd: dict) -> dict:
    return {"messages": [SystemMessage(SYSTEM), HumanMessage(task(jd))], "attempts": 0, "first_ok": None, "draft": "", "problems": [], "approved": None}


def run(llm, jd: dict, facts: list[str], index, config=None, decide=lambda draft: "approve", on_step=None) -> dict:
    """Runs to the end. `decide` plays the person at the interrupt, and `on_step` sees every step (the trace uses it)."""
    app = build(llm, facts, index)
    config = {**(config or {}), "configurable": {"thread_id": jd["id"]}, "recursion_limit": 40}
    tool_calls = 0
    payload = start_state(jd)
    while True:
        for chunk in app.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    if on_step:
                        on_step("human", {"waiting": True})
                    continue
                if node == "agent":
                    tool_calls += len(update["messages"][-1].tool_calls)
                if on_step:
                    on_step(node, update)
        snapshot = app.get_state(config)
        if not snapshot.next:
            break
        payload = Command(resume=decide(snapshot.values["draft"]))
    values = app.get_state(config).values
    return {"draft": values["draft"], "first_ok": values["first_ok"], "final_ok": not values["problems"], "attempts": values["attempts"],
            "tool_calls": tool_calls, "problems": values["problems"], "approved": values["approved"]}
