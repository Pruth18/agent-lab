# Agent Lab

One agent job, built three ways, to see what each framework actually adds.

**The job:** write 3 resume bullets for a job description, for a made-up candidate, under a prompt that pushes the model to exaggerate. The bullets may only use numbers and skills from the candidate's facts.

**The rule:** one deterministic validator marks every approach, so no framework grades its own work.

| Approach | File | What it shows |
|---|---|---|
| Plain Python loop | `agent_lab/loop.py` | Find facts, ask, check, retry with the reasons. The shape of JobHunter's resume writer |
| LangGraph agent | `agent_lab/graph.py` | Tool calling, a guard node the model can't skip, and a pause for human approval with a checkpointer |
| CrewAI crew | `agent_lab/crew.py` | An analyst, a writer and a fact checker working in sequence |
| Local model | `agent_lab/local.py` | An open Hugging Face model on a local CPU, through the same loop and the same check |

Facts are found by meaning with `sentence-transformers/all-MiniLM-L6-v2`: "workflow orchestration" finds the fact about Airflow, and "message queues and event streaming" finds Kafka.

## The graph

```
START -> agent <-> tools (search_facts, check_draft)
           |
           v
         guard --rejected--> agent
           |
           +--passes--> human (interrupt) -> END
           |
           +--out of attempts--> END
```

The model's own `check_draft` call is advice. `guard` is a node, so it runs whether the model checked or not. `tests/test_lab.py` proves that with a scripted model that never checks.

## Results

| Approach | First draft clean | Final answer clean | Model calls per run | Tokens per run |
|---|---|---|---|---|
| Gemini alone, nothing checks it | 4 of 5 | 4 of 5 | 1.0 | 338 |
| Plain Python loop | 4 of 5 | 5 of 5 | 1.2 | 423 |
| LangGraph agent | 5 of 5 | 5 of 5 | 3.8 | 2018 |
| CrewAI crew | 2 of 5 | 5 of 5 | 18.6 | 9884 |
| Plain loop on Qwen 2.5 0.5B, local CPU | 1 of 5 | 1 of 5 | 2.6 | n/a |

Five made-up job descriptions, one run each, on `gemini-3.1-flash-lite-preview` unless the row says otherwise. A small sample, so this is a first look. Time is left out because the free API's rate limits would dominate it.

- Left alone, the model slipped an unsupported skill into 1 of 5 answers. A check and one retry fixed it.
- All three approaches ended clean, at very different cost. CrewAI used about 5 times the tokens of LangGraph, and over 20 times the plain loop.
- For a fixed job like this I'd ship the plain loop. LangGraph earns its extra calls when the model has to choose its tools, or a person has to approve and resume later.
- In LangGraph the model's own check caught every slip, so the guard never had to fire in these runs. The tests show it does when the model skips the check.
- A tiny open model on a local CPU stayed clean on only 1 of 5, even with retries. The check stopped the other 4.

`results/trace_*.json` holds real runs, step by step.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt
pytest                                               # offline: no key or network needed

cp .env.example .env                                 # add a free Gemini API key
python -m agent_lab.evaluate --only raw loop graph crew
python -m agent_lab.evaluate --only local            # downloads the model, no key needed
python -m agent_lab.evaluate --trace analytics       # one run, step by step
```

## Honest limits

- The validator checks numbers and skills, not meaning. A false sentence built from true facts gets through.
- It's strict: it rejected "data models" for a candidate who "modelled marts".
- The sample is small. This is a learning project, not a benchmark.
- Built with Claude Code.
