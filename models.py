"""
models.py
---------
Pydantic data models used across the application.

ClassroomConditions holds the six crisp (numeric) inputs that both the
Manual Assessment mode and the AI Natural Language mode eventually produce.
These are the values that get fed into the fuzzy inference system.

FuzzyResult holds the output of the fuzzy engine: the 0-100 score, the
category label, and (for transparency / viva purposes) the individual
rule-firing strengths that contributed to the result.
"""

from pydantic import BaseModel, Field, field_validator


# Valid numeric ranges for each variable. These are used both for Streamlit
# slider bounds and to validate whatever LangChain extracts from free text.
RANGES = {
    "temperature": (10.0, 45.0),   # degrees Celsius
    "humidity": (0.0, 100.0),      # percent relative humidity
    "co2": (300.0, 3000.0),        # ppm
    "noise": (20.0, 100.0),        # dB
    "light": (0.0, 1000.0),        # lux
    "occupancy": (0.0, 100.0),     # percent of room capacity in use
}


class ClassroomConditions(BaseModel):
    """Structured, validated classroom environment readings."""

    temperature: float = Field(..., description="Room temperature in Celsius")
    humidity: float = Field(..., description="Relative humidity in percent")
    co2: float = Field(..., description="CO2 concentration in ppm")
    noise: float = Field(..., description="Ambient noise level in dB")
    light: float = Field(..., description="Illuminance in lux")
    occupancy: float = Field(..., description="Occupancy as percent of room capacity")

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        lo, hi = RANGES["temperature"]
        return _clamp(v, lo, hi)

    @field_validator("humidity")
    @classmethod
    def _check_humidity(cls, v: float) -> float:
        lo, hi = RANGES["humidity"]
        return _clamp(v, lo, hi)

    @field_validator("co2")
    @classmethod
    def _check_co2(cls, v: float) -> float:
        lo, hi = RANGES["co2"]
        return _clamp(v, lo, hi)

    @field_validator("noise")
    @classmethod
    def _check_noise(cls, v: float) -> float:
        lo, hi = RANGES["noise"]
        return _clamp(v, lo, hi)

    @field_validator("light")
    @classmethod
    def _check_light(cls, v: float) -> float:
        lo, hi = RANGES["light"]
        return _clamp(v, lo, hi)

    @field_validator("occupancy")
    @classmethod
    def _check_occupancy(cls, v: float) -> float:
        lo, hi = RANGES["occupancy"]
        return _clamp(v, lo, hi)


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value into [lo, hi] instead of raising, so a slightly
    out-of-range LLM extraction doesn't crash the whole pipeline."""
    return max(lo, min(hi, float(value)))


class FuzzyResult(BaseModel):
    """Output of the fuzzy inference engine."""

    score: float = Field(..., description="Environment quality score, 0-100")
    category: str = Field(..., description="Very Poor / Poor / Average / Good / Excellent")
    rule_strengths: dict = Field(
        default_factory=dict,
        description="Firing strength (0-1) of each rule that activated, for transparency",
    )


class ExplanationResult(BaseModel):
    """Output of the second LangChain call (natural-language explanation)."""

    explanation: str
    source: str = Field(default="llm", description="'llm' or 'fallback' if the API was unavailable")
