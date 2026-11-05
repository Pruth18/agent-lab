"""
A small open model from Hugging Face, running on the local CPU through LangChain. It uses the plain loop in loop.py,
so the only thing that changes is the model. This is the harness I'd point at any new model: same task, same check.
"""
from __future__ import annotations

import os

LOCAL_MODEL = os.environ.get("AGENT_LAB_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def load(model_id: str = LOCAL_MODEL):
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    pipe = HuggingFacePipeline.from_model_id(
        model_id=model_id, task="text-generation", device_map="auto",
        pipeline_kwargs={"max_new_tokens": 220, "do_sample": False, "return_full_text": False},
    )
    return ChatHuggingFace(llm=pipe)
