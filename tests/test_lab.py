"""Runs without a network or a key: a scripted model and fake embeddings stand in, and everything else is the real code."""
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent_lab import graph, loop
from agent_lab.common import fact_index, load_candidate, load_jds, make_tools
from agent_lab.validator import check

FACTS = load_candidate()["facts"]
JD = load_jds()[0]
CLEAN = "- Built PySpark pipelines processing 2 million daily orders.\n- Reduced batch runtime by 30%.\n- Deployed Airflow DAGs for 12 jobs."
INVENTED = "- Cut warehouse costs by 73% on Kubernetes.\n- Reduced batch runtime by 30%.\n- Deployed Airflow DAGs for 12 jobs."


class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def model(*replies):
    return Scripted(messages=iter([r if isinstance(r, AIMessage) else AIMessage(r) for r in replies]))


def index():
    return fact_index(FACTS, DeterministicFakeEmbedding(size=32))


def test_validator_catches_invented_number_and_skill():
    v = check(INVENTED, FACTS)
    assert v.bad_numbers == ["73"] and v.bad_skills == ["Kubernetes"] and not v.ok


def test_validator_passes_true_draft():
    assert check(CLEAN, FACTS).ok


def test_skill_matching_is_whole_word():
    assert check("Scalable pipelines", FACTS).ok
    assert check("Scala pipelines", FACTS).bad_skills == ["Scala"]


def test_check_draft_tool_reports_problems():
    tools = {t.name: t for t in make_tools(FACTS, index())}
    assert tools["check_draft"].invoke({"draft": CLEAN}) == "OK"
    assert "73" in tools["check_draft"].invoke({"draft": INVENTED})
    assert tools["search_facts"].invoke({"query": "orchestration"})


def test_loop_retries_with_reasons_then_passes():
    out = loop.run(model(INVENTED, CLEAN), JD, FACTS, index())
    assert (out["first_ok"], out["final_ok"], out["attempts"]) == (False, True, 2)


def test_loop_gives_up_after_three_attempts():
    out = loop.run(model(INVENTED, INVENTED, INVENTED), JD, FACTS, index())
    assert out["final_ok"] is False and out["attempts"] == 3


def test_graph_guard_sends_a_bad_draft_back():
    out = graph.run(model(INVENTED, CLEAN), JD, FACTS, index())
    assert out["first_ok"] is False and out["final_ok"] is True and out["attempts"] == 2 and out["approved"] is True


def test_graph_guard_runs_even_when_the_model_skips_check_draft():
    # The model never calls a tool here, and the invented draft is still caught
    out = graph.run(model(INVENTED, INVENTED, INVENTED), JD, FACTS, index())
    assert out["final_ok"] is False and out["attempts"] == 3 and out["approved"] is None


def test_graph_runs_the_tools_the_model_asks_for():
    call = AIMessage("", tool_calls=[{"name": "search_facts", "args": {"query": "spark"}, "id": "1"}])
    steps = []
    out = graph.run(model(call, CLEAN), JD, FACTS, index(), on_step=lambda node, update: steps.append(node))
    assert out["tool_calls"] == 1 and steps[:3] == ["agent", "tools", "agent"]


def test_graph_pauses_for_a_person_who_can_say_no():
    seen = []

    def decide(draft):
        seen.append(draft)
        return "reject"

    out = graph.run(model(CLEAN), JD, FACTS, index(), decide=decide)
    assert seen == [CLEAN] and out["approved"] is False
