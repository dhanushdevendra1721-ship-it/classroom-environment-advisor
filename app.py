"""
app.py
------
Streamlit entry point for the AI-Based Classroom Environment Quality Advisor.

Two modes:
  - Manual Assessment: sliders -> fuzzy engine -> score
  - AI Natural Language Assessment: free text -> LangChain extraction ->
    fuzzy engine -> score -> LangChain explanation

Run locally with:  streamlit run app.py
"""

import streamlit as st

from models import ClassroomConditions, RANGES
from fuzzy_engine import (
    run_fuzzy_inference_detailed,
    get_fuzzification_breakdown,
    get_fired_rules_readable,
    RULES,
)
from llm_service import (
    extract_conditions_from_text,
    generate_explanation,
    LLMConfigError,
    LLMExtractionError,
)
from utils import (
    load_secrets_into_env,
    score_gauge,
    inputs_radar_chart,
    membership_function_plot,
    aggregated_output_plot,
)

st.set_page_config(
    page_title="Classroom Environment Quality Advisor",
    page_icon="🏫",
    layout="wide",
)

load_secrets_into_env()


# ---------------------------------------------------------------------------
# Shared display logic
# ---------------------------------------------------------------------------

def render_result(conditions: ClassroomConditions, explanation_text: str, explanation_source: str):
    """Runs the fuzzy engine on `conditions` and renders score, gauge,
    rule trace, and the explanation text. Used by both modes."""
    result, aggregated = run_fuzzy_inference_detailed(conditions)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(score_gauge(result.score, result.category), use_container_width=True)
    with col2:
        st.plotly_chart(inputs_radar_chart(conditions.model_dump()), use_container_width=True)

    st.subheader("📋 Input Values Used")
    vals = conditions.model_dump()
    cols = st.columns(6)
    labels = {
        "temperature": ("Temperature", "°C"),
        "humidity": ("Humidity", "%"),
        "co2": ("CO2", "ppm"),
        "noise": ("Noise", "dB"),
        "light": ("Light", "lux"),
        "occupancy": ("Occupancy", "%"),
    }
    for i, key in enumerate(labels):
        name, unit = labels[key]
        cols[i].metric(name, f"{vals[key]:.0f} {unit}")

    st.subheader("💡 Explanation")
    if explanation_source == "fallback":
        st.info(explanation_text + "\n\n_(Generated without the LLM — API key missing or request failed.)_")
    else:
        st.success(explanation_text)

    with st.expander("🔍 See the fuzzy reasoning behind this score"):
        st.markdown("**Rules that fired** (fuzzy AND = minimum of membership degrees):")
        fired = get_fired_rules_readable(conditions)
        if fired:
            for r in sorted(fired, key=lambda x: -x["strength"]):
                st.write(f"- `{r['id']}` (strength **{r['strength']}**): {r['description']}")
        else:
            st.write("No rule fired strongly for these inputs — score defaults toward the middle.")

        st.markdown("**Aggregation & Defuzzification:**")
        st.plotly_chart(aggregated_output_plot(aggregated, result.score), use_container_width=True)

        st.markdown("**Fuzzification breakdown** (membership degree of each input in each linguistic term):")
        breakdown = get_fuzzification_breakdown(conditions)
        st.json(breakdown)

    return result


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🏫 AI-Based Classroom Environment Quality Advisor")
st.caption("LangChain + Fuzzy Logic")

mode = st.radio(
    "Choose a mode:",
    ["🎛️ Manual Assessment", "🤖 AI Natural Language Assessment"],
    horizontal=True,
)

st.divider()

# ---------------------------------------------------------------------------
# MODE 1: Manual Assessment
# ---------------------------------------------------------------------------
if mode == "🎛️ Manual Assessment":
    st.subheader("🎛️ Manual Assessment")
    st.write("Set the classroom readings using the sliders, then click **Analyze Classroom**.")

    c1, c2, c3 = st.columns(3)
    with c1:
        temperature = st.slider("Temperature (°C)", *RANGES["temperature"], value=24.0, step=0.5)
        humidity = st.slider("Humidity (%)", *RANGES["humidity"], value=50.0, step=1.0)
    with c2:
        co2 = st.slider("CO2 (ppm)", *RANGES["co2"], value=700.0, step=10.0)
        noise = st.slider("Noise (dB)", *RANGES["noise"], value=50.0, step=1.0)
    with c3:
        light = st.slider("Light (lux)", *RANGES["light"], value=400.0, step=10.0)
        occupancy = st.slider("Occupancy (%)", *RANGES["occupancy"], value=60.0, step=1.0)

    if st.button("🔍 Analyze Classroom", type="primary"):
        conditions = ClassroomConditions(
            temperature=temperature, humidity=humidity, co2=co2,
            noise=noise, light=light, occupancy=occupancy,
        )
        # Compute fuzzy result first so the explanation is grounded in it
        result_preview, _ = run_fuzzy_inference_detailed(conditions)
        fired_descriptions = [r["description"] for r in get_fired_rules_readable(conditions)]

        with st.spinner("Generating explanation..."):
            explanation = generate_explanation(
                conditions, result_preview.score, result_preview.category, fired_descriptions
            )

        render_result(conditions, explanation.explanation, explanation.source)

# ---------------------------------------------------------------------------
# MODE 2: AI Natural Language Assessment
# ---------------------------------------------------------------------------
else:
    st.subheader("🤖 AI Natural Language Assessment")
    st.write("Describe the classroom in your own words. LangChain will extract the numbers, "
             "then the fuzzy engine will score the environment.")

    description = st.text_area(
        "Describe your classroom environment in natural language...",
        placeholder=(
            "The room is slightly hot, around 29 degrees, with about 45 students. "
            "It is noisy and CO2 is around 1200 ppm."
        ),
        height=120,
    )

    if st.button("🤖 Analyze with AI", type="primary"):
        if not description.strip():
            st.warning("Please enter a description first.")
        else:
            try:
                with st.spinner("LangChain is extracting classroom parameters..."):
                    conditions = extract_conditions_from_text(description)

                st.subheader("🧩 Extracted Parameters")
                st.json(conditions.model_dump())

                result_preview, _ = run_fuzzy_inference_detailed(conditions)
                fired_descriptions = [r["description"] for r in get_fired_rules_readable(conditions)]

                with st.spinner("Generating explanation..."):
                    explanation = generate_explanation(
                        conditions, result_preview.score, result_preview.category, fired_descriptions
                    )

                render_result(conditions, explanation.explanation, explanation.source)

            except LLMConfigError as e:
                st.error(f"⚠️ {e}")
                st.info("Tip: You can still use **Manual Assessment** mode without an API key.")
            except LLMExtractionError as e:
                st.error(f"⚠️ Could not extract classroom parameters: {e}")
            except Exception as e:  # noqa: BLE001
                st.error(f"⚠️ Unexpected error: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Explanation sections (viva-friendly)
# ---------------------------------------------------------------------------
with st.expander("📘 How the Fuzzy Logic Works"):
    st.markdown(
        """
1. **Crisp Input** — a real number, e.g. Temperature = 29°C.
2. **Fuzzification** — that number is converted into membership degrees in
   linguistic terms (e.g. 0.3 "Comfortable", 0.7 "Hot") using membership
   function curves.
3. **Membership Functions** — triangular/trapezoidal curves define how
   much each term applies across the variable's range.
4. **Rule Evaluation** — each rule's antecedents are combined with fuzzy
   AND (minimum) to get a firing strength.
5. **Aggregation** — all rules pointing to the same output term are
   combined with fuzzy OR (maximum), then all output terms are combined
   into one aggregated curve.
6. **Defuzzification** — the centroid (center of gravity) of the
   aggregated curve gives the final crisp score (0-100).
        """
    )
    st.markdown("**Sample fuzzy rules used in this system:**")
    for rule in RULES[:5]:
        st.write(f"- `{rule.id}`: {rule.description}")

    st.markdown("**Membership function visualizations:**")
    var_choice = st.selectbox(
        "Choose a variable to visualize:",
        ["temperature", "humidity", "co2", "noise", "light", "occupancy", "quality"],
        key="mf_viewer",
    )
    st.plotly_chart(membership_function_plot(var_choice), use_container_width=True)

with st.expander("📘 How LangChain Works"):
    st.markdown(
        """
**Pipeline:**

```
Natural Language
      ↓
LangChain Prompt Template
      ↓
LLM (via ChatOpenAI)
      ↓
Structured Extraction (JSON → Pydantic validation)
      ↓
Fuzzy Inference Engine
      ↓
Quality Score (0-100)
      ↓
LangChain Prompt Template (explanation)
      ↓
AI-Generated Explanation
```

LangChain is used for **two** distinct tasks:
1. **Extraction** — reading a free-text classroom description and turning
   vague language ("quite warm", "very crowded") into the six numeric
   fields the fuzzy engine needs.
2. **Explanation** — turning the fuzzy engine's numeric result back into a
   short, plain-English explanation.

The LLM **never decides the final score** — that always comes from the
fuzzy inference system. This keeps the numeric result explainable,
consistent, and reproducible, while still letting the app understand
natural language.
        """
    )

st.caption("Built for an academic mini-project — LangChain + Fuzzy Logic demo.")
