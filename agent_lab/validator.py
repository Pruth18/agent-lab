"""
The part of the system no model can talk its way past.

Two deterministic rules, taken from JobHunter (https://getaaroh.in's engine):
  1. Every number in a draft must appear in the candidate's facts.
  2. Every known skill in a draft must appear in the candidate's facts.

It checks numbers and skills, not meaning. A false sentence built from true facts gets through,
which is why a person still approves the result at the end of the graph.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
NUMBER_RX = re.compile(r"\d+(?:\.\d+)?")


@lru_cache(maxsize=1)
def _skills() -> dict[str, re.Pattern]:
    taxonomy = json.loads((DATA / "skills.json").read_text(encoding="utf-8"))
    # Whole-word match: no letter or digit directly before or after
    return {name: re.compile("|".join(rf"(?<!\w)(?:{p})(?!\w)" for p in pats), re.IGNORECASE) for name, pats in taxonomy.items()}


def find_skills(text: str) -> list[str]:
    return [name for name, rx in _skills().items() if rx.search(text or "")]


@dataclass
class Verdict:
    bad_numbers: list[str] = field(default_factory=list)
    bad_skills: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.bad_numbers and not self.bad_skills

    @property
    def problems(self) -> list[str]:
        out = []
        if self.bad_numbers:
            out.append(f"numbers not in the candidate facts (invented?): {', '.join(self.bad_numbers)}")
        if self.bad_skills:
            out.append(f"skills the candidate cannot evidence: {', '.join(self.bad_skills)}")
        return out


def check(draft: str, facts: list[str]) -> Verdict:
    source = "\n".join(facts)
    numbers, skills = set(NUMBER_RX.findall(source)), set(find_skills(source))
    return Verdict(
        bad_numbers=sorted({n for n in NUMBER_RX.findall(draft or "") if n not in numbers}),
        bad_skills=[s for s in find_skills(draft or "") if s not in skills],
    )
