"""
llm_service.py
---------------
LangChain layer for the Classroom Environment Advisor.

Two LLM-powered pipelines:

1. extract_conditions_from_text()
   Natural language -> ChatPromptTemplate -> LLM
   -> JSON -> validation -> ClassroomConditions

2. generate_explanation()
   Fuzzy result -> ChatPromptTemplate -> LLM
   -> plain-English explanation

Important:
- The LLM NEVER calculates the final classroom score.
- The final score always comes from fuzzy_engine.py.
- The LLM only extracts readings and explains the fuzzy result.
- API credentials are never printed or exposed in logs.
"""

import json
import logging
import os
import re
from typing import List

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from models import ClassroomConditions, ExplanationResult
from prompts import EXTRACTION_PROMPT, EXPLANATION_PROMPT


logger = logging.getLogger(__name__)


class LLMConfigError(Exception):
    """Raised when the LLM cannot be configured."""


class LLMExtractionError(Exception):
    """Raised when the LLM response cannot be parsed or validated."""


def _get_api_key() -> str:
    """
    Read the API key from environment variables.

    Streamlit secrets are loaded into os.environ by app.py/utils.py.
    """
    key = os.environ.get("LLM_API_KEY", "").strip()

    if not key:
        raise LLMConfigError(
            "No LLM_API_KEY found. Set it in your .env file locally "
            "or in Streamlit Secrets on Streamlit Cloud."
        )

    return key


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """
    Build the configured OpenAI-compatible ChatOpenAI client.

    Supported configuration:
        LLM_API_KEY
        LLM_MODEL
        LLM_BASE_URL
    """

    api_key = _get_api_key()

    model = os.environ.get(
        "LLM_MODEL",
        "gpt-4o-mini",
    ).strip()

    base_url = os.environ.get(
        "LLM_BASE_URL",
        "",
    ).strip() or None

    return ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=base_url,
    temperature=temperature,
    timeout=30,
    max_tokens=1000,
    )


def _strip_code_fences(text: str) -> str:
    """
    Remove accidental Markdown code fences around JSON.

    Example:
        ```json
        {"temperature": 24}
        ```

    becomes:
        {"temperature": 24}
    """

    if not text:
        return ""

    text = text.strip()

    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return text


def extract_conditions_from_text(
    description: str,
) -> ClassroomConditions:
    """
    LangChain pipeline #1:

    Natural language
        -> ChatPromptTemplate
        -> ChatOpenAI
        -> JSON parser
        -> Pydantic validation
        -> ClassroomConditions
    """

    if not description or not description.strip():
        raise LLMExtractionError(
            "Please describe the classroom before analyzing."
        )

    try:
        llm = get_llm(temperature=0.1)

        parser = JsonOutputParser()

        chain = EXTRACTION_PROMPT | llm | parser

        try:
            raw = chain.invoke(
                {
                    "description": description.strip(),
                }
            )

        except OutputParserException as exc:
            """
            Sometimes the LLM returns valid JSON wrapped in Markdown.

            Retry once without the JSON parser and clean the response
            manually.
            """

            logger.warning(
                "LLM returned output that could not be parsed directly. "
                "Attempting manual JSON recovery."
            )

            try:
                raw_chain = EXTRACTION_PROMPT | llm

                raw_message = raw_chain.invoke(
                    {
                        "description": description.strip(),
                    }
                )

                cleaned = _strip_code_fences(
                    raw_message.content
                )

                raw = json.loads(cleaned)

            except Exception as recovery_exc:
                logger.exception(
                    "Manual JSON recovery failed: %s",
                    recovery_exc,
                )

                raise LLMExtractionError(
                    "The AI response could not be converted into valid "
                    "classroom data."
                ) from recovery_exc

        except Exception as exc:
            logger.exception(
                "LLM extraction request failed: %s",
                exc,
            )

            raise LLMExtractionError(
                f"LLM request failed: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise LLMExtractionError(
                "The AI returned an unexpected response format."
            )

        required_keys = {
            "temperature",
            "humidity",
            "co2",
            "noise",
            "light",
            "occupancy",
        }

        missing = required_keys - set(raw.keys())

        if missing:
            raise LLMExtractionError(
                "AI response is missing fields: "
                + ", ".join(sorted(missing))
            )

        try:
            values = {
                key: raw[key]
                for key in required_keys
            }

            return ClassroomConditions(**values)

        except Exception as exc:
            logger.exception(
                "Extracted classroom values failed validation: %s",
                exc,
            )

            raise LLMExtractionError(
                f"Extracted values failed validation: {exc}"
            ) from exc

    except LLMExtractionError:
        raise

    except LLMConfigError:
        raise

    except Exception as exc:
        logger.exception(
            "Unexpected error during classroom condition extraction: %s",
            exc,
        )

        raise LLMExtractionError(
            f"Unexpected LLM error: {exc}"
        ) from exc


def generate_explanation(
    conditions: ClassroomConditions,
    score: float,
    category: str,
    fired_rule_descriptions: List[str],
) -> ExplanationResult:
    """
    LangChain pipeline #2:

    Fuzzy result
        -> readings + score + category + fired rules
        -> ChatPromptTemplate
        -> ChatOpenAI
        -> plain-English explanation

    If the LLM is unavailable, the function safely returns a
    non-LLM fallback explanation.
    """

    try:
        llm = get_llm(temperature=0.4)

        chain = EXPLANATION_PROMPT | llm

        readings_str = ", ".join(
            f"{key}={value}"
            for key, value in conditions.model_dump().items()
        )

        rules_str = (
            "; ".join(fired_rule_descriptions)
            if fired_rule_descriptions
            else "none strongly activated"
        )

        response = chain.invoke(
            {
                "readings": readings_str,
                "score": score,
                "category": category,
                "fired_rules": rules_str,
            }
        )

        explanation = getattr(
            response,
            "content",
            "",
        )

        if not explanation or not explanation.strip():
            raise ValueError(
                "The LLM returned an empty explanation."
            )

        return ExplanationResult(
            explanation=explanation.strip(),
            source="llm",
        )

    except Exception as exc:
        """
        IMPORTANT:
        Do not expose the API key or sensitive configuration.

        logging.exception() records the actual error and traceback in
        Streamlit Cloud logs, making debugging much easier.
        """

        logger.exception(
            "LLM explanation request failed: %s",
            exc,
        )

        return ExplanationResult(
            explanation=_fallback_explanation(
                conditions,
                score,
                category,
            ),
            source="fallback",
        )


def _fallback_explanation(
    conditions: ClassroomConditions,
    score: float,
    category: str,
) -> str:
    """
    Generate a simple explanation without using an LLM.

    This guarantees that Manual Assessment mode continues working
    even if the API is unavailable.
    """

    c = conditions

    notes = []

    if c.temperature > 27:
        notes.append(
            "the temperature is on the warm side"
        )

    elif c.temperature < 18:
        notes.append(
            "the temperature is on the cold side"
        )

    if c.humidity > 70:
        notes.append(
            "humidity is relatively high"
        )

    elif c.humidity < 30:
        notes.append(
            "humidity is relatively low"
        )

    if c.co2 > 1200:
        notes.append(
            "CO2 levels are elevated, suggesting poor ventilation"
        )

    if c.noise > 65:
        notes.append(
            "the noise level is high"
        )

    if c.light < 300:
        notes.append(
            "lighting may be lower than recommended"
        )

    if c.occupancy > 80:
        notes.append(
            "the room is heavily occupied"
        )

    if not notes:
        notes.append(
            "most readings are within comfortable ranges"
        )

    reason_text = ", and ".join(notes)

    suggestion = ""

    if category in ("Poor", "Very Poor"):
        suggestion = (
            " Improving ventilation and reducing noise "
            "would likely help the most."
        )

    elif category == "Average":
        suggestion = (
            " A few small adjustments could improve "
            "the classroom environment."
        )

    elif category == "Good":
        suggestion = (
            " The current conditions are generally comfortable, "
            "with only minor improvements potentially needed."
        )

    elif category == "Excellent":
        suggestion = (
            " The current readings indicate a generally "
            "comfortable classroom environment."
        )

    return (
        f"The classroom environment scored {score}/100 "
        f"({category}). "
        f"This is mainly because {reason_text}."
        f"{suggestion}"
    )
