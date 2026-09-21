"""
llm_service.py
---------------
The LangChain layer. Two real LLM-powered pipelines live here:

1. extract_conditions_from_text()
       Natural language  ->  ChatPromptTemplate  ->  LLM  ->
       structured JSON  ->  Pydantic validation  ->  ClassroomConditions

2. generate_explanation()
       Fuzzy engine result (score, category, fired rules)  ->
       ChatPromptTemplate  ->  LLM  ->  plain-English explanation

Neither function ever lets the LLM invent the final quality score — that
number always comes from fuzzy_engine.py. The LLM only (a) turns language
into numbers, and (b) turns numbers back into language.

The LLM provider is OpenAI-compatible and configured entirely through
environment variables (see .env.example), so the same code works with
OpenAI itself or with any OpenAI-compatible endpoint (Groq, OpenRouter,
Together AI, etc.) by changing LLM_BASE_URL.
"""

import json
import os
import re
from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI

from models import ClassroomConditions, ExplanationResult
from prompts import EXTRACTION_PROMPT, EXPLANATION_PROMPT


class LLMConfigError(Exception):
    """Raised when the LLM cannot be configured (e.g. missing API key)."""


class LLMExtractionError(Exception):
    """Raised when the LLM response cannot be parsed into valid conditions."""


def _get_api_key() -> str:
    """Read the API key from env / Streamlit secrets. Streamlit secrets are
    injected into os.environ by app.py at startup (see app.py), so this
    function only needs to look at the environment."""
    key = os.environ.get("LLM_API_KEY", "").strip()
    if not key:
        raise LLMConfigError(
            "No LLM_API_KEY found. Set it in your .env file (local) or in "
            "Streamlit secrets (cloud) — see .env.example."
        )
    return key


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Build a configured ChatOpenAI client.

    LLM_MODEL and LLM_BASE_URL are optional; sensible defaults are used so
    the app works out of the box with a plain OpenAI key, while still
    allowing students on a free-tier OpenAI-compatible provider (e.g. Groq)
    to just change two env vars.
    """
    api_key = _get_api_key()
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("LLM_BASE_URL") or None

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=30,
    )


def _strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` even when told not to.
    Strip that defensively before parsing."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def extract_conditions_from_text(description: str) -> ClassroomConditions:
    """LangChain pipeline #1: Natural language -> structured numeric conditions.

    Pipeline: ChatPromptTemplate -> ChatOpenAI -> JsonOutputParser -> Pydantic validation.
    """
    if not description or not description.strip():
        raise LLMExtractionError("Please describe the classroom before analyzing.")

    llm = get_llm(temperature=0.1)
    parser = JsonOutputParser()
    chain = EXTRACTION_PROMPT | llm | parser

    try:
        raw = chain.invoke({"description": description})
    except OutputParserException:
        # Fallback: ask the raw chain (no parser) and parse manually,
        # stripping any accidental markdown fences.
        try:
            raw_chain = EXTRACTION_PROMPT | llm
            raw_message = raw_chain.invoke({"description": description})
            cleaned = _strip_code_fences(raw_message.content)
            raw = json.loads(cleaned)
    except Exception as exc:  # noqa: BLE001
            raise LLMExtractionError(
                f"The AI response could not be parsed as valid JSON: {exc}"
            ) from exc
    except Exception as exc:  # noqa: BLE001
        raise LLMExtractionError(f"LLM request failed: {exc}") from exc

    required_keys = {"temperature", "humidity", "co2", "noise", "light", "occupancy"}
    missing = required_keys - set(raw.keys())
    if missing:
        raise LLMExtractionError(f"AI response is missing fields: {', '.join(sorted(missing))}")

    try:
        return ClassroomConditions(**{k: raw[k] for k in required_keys})
    except Exception as exc:  # noqa: BLE001
        raise LLMExtractionError(f"Extracted values failed validation: {exc}") from exc


def generate_explanation(
    conditions: ClassroomConditions,
    score: float,
    category: str,
    fired_rule_descriptions: List[str],
) -> ExplanationResult:
    """LangChain pipeline #2: Fuzzy result -> plain-English explanation.

    Falls back to a simple templated explanation (no LLM call) if the API
    key is missing or the request fails, so the app never breaks the user's
    flow just because the explanation step is unavailable.
    """
    try:
        llm = get_llm(temperature=0.4)
        chain = EXPLANATION_PROMPT | llm

        readings_str = ", ".join(f"{k}={v}" for k, v in conditions.model_dump().items())
        rules_str = "; ".join(fired_rule_descriptions) if fired_rule_descriptions else "none strongly activated"

        response = chain.invoke(
            {
                "readings": readings_str,
                "score": score,
                "category": category,
                "fired_rules": rules_str,
            }
        )
        return ExplanationResult(explanation=response.content.strip(), source="llm")

       
    except Exception as exc:
        import logging
        logging.exception("LLM explanation request failed")

        return ExplanationResult(explanation=response.content.strip(), source="llm")

    except Exception as exc:


def _fallback_explanation(conditions: ClassroomConditions, score: float, category: str) -> str:
    """A simple, non-LLM explanation used when the API is unavailable, so
    Manual Assessment mode keeps working even with no internet/API key."""
    c = conditions
    notes = []
    if c.temperature > 27:
        notes.append("the temperature is on the warm side")
    elif c.temperature < 18:
        notes.append("the temperature is on the cold side")
    if c.co2 > 1200:
        notes.append("CO2 levels are elevated, suggesting poor ventilation")
    if c.noise > 65:
        notes.append("the noise level is high")
    if c.occupancy > 80:
        notes.append("the room is heavily occupied")
    if not notes:
        notes.append("most readings are within comfortable ranges")

    suggestion = ""
    if category in ("Poor", "Very Poor"):
        suggestion = " Improving ventilation and reducing noise would likely help the most."
    elif category == "Average":
        suggestion = " A few small adjustments could push this into the 'Good' range."

    return (
        f"The classroom environment scored {score}/100 ({category}). "
        f"This is mainly because {', and '.join(notes)}.{suggestion}"
    )
