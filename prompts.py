"""
prompts.py
----------
Prompt templates used by the two LangChain calls in llm_service.py:

1. EXTRACTION_PROMPT — turns a free-text classroom description into the six
   structured numeric fields defined by ClassroomConditions.
2. EXPLANATION_PROMPT — turns the fuzzy engine's numeric result into a short
   human-readable explanation, grounded strictly in the numbers it is given.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# 1. Extraction prompt
# ---------------------------------------------------------------------------
# This is the "real LangChain work": mapping vague natural language onto a
# numeric scale the fuzzy engine understands. The LLM is explicitly told
# the valid ranges so it grounds its estimates sensibly, and it is told to
# guess a reasonable default (not fail) when a factor isn't mentioned.

EXTRACTION_SYSTEM_PROMPT = """You are a classroom-environment data extraction assistant.

Read the user's natural-language description of a classroom and estimate six
numeric readings. Use your judgement to convert descriptive language (e.g.
"quite warm", "very crowded", "a bit noisy") into a specific number within
the given range. If a factor is not mentioned at all, use a typical/default
value roughly in the middle of its range rather than an extreme value.

Valid ranges (use these to keep your estimates realistic):
- temperature: 10 to 45 (degrees Celsius). Typical comfortable classroom: ~22-24.
- humidity: 0 to 100 (percent relative humidity). Typical comfortable: ~40-55.
- co2: 300 to 3000 (ppm). Outdoor/fresh air baseline ~400. Poorly ventilated, crowded room can exceed 1500.
- noise: 20 to 100 (decibels). A quiet room is ~30-40 dB, a noisy room is ~75+ dB.
- light: 0 to 1000 (lux). Comfortable classroom lighting is roughly 300-500 lux.
- occupancy: 0 to 100 (percent of room capacity currently filled).

Respond with ONLY a JSON object with exactly these six keys and numeric
values, no explanation, no markdown fences:
{{"temperature": <number>, "humidity": <number>, "co2": <number>, "noise": <number>, "light": <number>, "occupancy": <number>}}
"""

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACTION_SYSTEM_PROMPT),
        ("human", "Classroom description:\n{description}"),
    ]
)


# ---------------------------------------------------------------------------
# 2. Explanation prompt
# ---------------------------------------------------------------------------
# Crucially, the LLM is given the *already computed* fuzzy score/category and
# the rules that fired. It is instructed never to change or invent a new
# score — it is only explaining a result the fuzzy system already produced.

EXPLANATION_SYSTEM_PROMPT = """You are a classroom environment advisor.

You will be given the numeric readings of a classroom, a quality score
(0-100) and category that a fuzzy logic system has already calculated, and
the fuzzy rules that fired during that calculation. Your job is ONLY to
explain the result in 2-4 plain-English sentences: name the 1-3 factors that
most likely drove the score, and give one brief practical suggestion if the
score is not "Excellent".

Rules:
- Do NOT invent or restate a different numeric score. Treat the given score
  and category as fixed, correct facts.
- Do NOT use technical fuzzy-logic jargon (no "membership", "defuzzification", etc).
- Keep it concise and easy for a student/teacher to read.
"""

EXPLANATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", EXPLANATION_SYSTEM_PROMPT),
        (
            "human",
            "Readings: {readings}\n"
            "Score: {score}/100\n"
            "Category: {category}\n"
            "Rules that fired: {fired_rules}\n\n"
            "Explain this result.",
        ),
    ]
)
