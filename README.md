# 🏫 AI-Based Classroom Environment Quality Advisor

**LangChain + Fuzzy Logic mini-project**

A Streamlit web app that scores classroom environment quality (0–100) using
a genuine **Mamdani Fuzzy Inference System**, with a **LangChain**-powered
natural-language front end that lets a user describe a classroom in plain
English instead of setting sliders.

---

## 1. Problem Statement

Teachers and students have an intuitive sense of when a classroom "feels
off" — too hot, too noisy, stuffy air — but no simple tool to quantify it.
Environmental factors like temperature, humidity, CO2, noise, lighting and
occupancy interact in ways that are naturally *vague* ("somewhat warm",
"a bit crowded"), which makes classical if-else scoring awkward and
brittle. This is a textbook use case for **fuzzy logic**, which is designed
to reason with degrees of truth rather than sharp thresholds.

## 2. Objective

Build a tool that:
1. Accepts classroom readings either manually (sliders) or as a free-text
   description.
2. Uses a **fuzzy inference system** — not if/else rules — to compute a
   0–100 "Environment Quality" score and category.
3. Uses **LangChain + an LLM** to (a) turn natural language into numeric
   readings, and (b) turn the numeric result back into a plain-English
   explanation.

## 3. Features

- 🎛️ **Manual Assessment** — sliders for all 6 environmental factors.
- 🤖 **AI Natural Language Assessment** — describe the room in your own
  words; LangChain extracts the numbers.
- A genuine **Mamdani fuzzy inference engine**: fuzzification → rule
  evaluation → aggregation → centroid defuzzification.
- 15-rule fuzzy rule base, fully visible in the UI (not hidden).
- LangChain-generated plain-English explanation of every result, grounded
  in the fuzzy engine's actual numbers.
- Visualizations: quality gauge, input radar chart, membership function
  plots, and the aggregated-output/defuzzification plot.
- Works offline in Manual mode even without an API key (fallback
  explanation), so a missing/expired key never fully breaks the app.
- No hardcoded API keys — `.env` locally, `st.secrets` on Streamlit Cloud.

## 4. Technologies Used

| Purpose            | Technology                         |
|---------------------|-------------------------------------|
| UI                  | Streamlit                           |
| LLM orchestration   | LangChain (`langchain`, `langchain-openai`, `langchain-core`) |
| LLM                 | Any OpenAI-compatible model (default `gpt-4o-mini`) |
| Fuzzy logic         | scikit-fuzzy (`skfuzzy`) + NumPy    |
| Data validation     | Pydantic v2                         |
| Charts              | Plotly                              |
| Secrets             | python-dotenv / `st.secrets`        |

## 5. System Architecture

```
                    ┌─────────────────────┐
   Manual sliders → │                     │
                     │   Streamlit UI      │
 Free-text input →  │      (app.py)       │
                     └──────────┬──────────┘
                                │
                 ┌──────────────┴───────────────┐
                 │ (AI mode only)                │
                 ▼                               │
     ┌───────────────────────┐                   │
     │   llm_service.py       │                   │
     │  (LangChain extraction)│                   │
     └───────────┬───────────┘                   │
                 │ ClassroomConditions            │
                 ▼                               ▼
        ┌───────────────────────────────────────┐
        │            fuzzy_engine.py              │
        │  Fuzzification → Rules → Aggregation →  │
        │            Defuzzification               │
        └───────────────────┬──────────────────────┘
                            │ FuzzyResult (score, category)
                            ▼
                ┌───────────────────────┐
                │   llm_service.py       │
                │ (LangChain explanation)│
                └───────────┬───────────┘
                            ▼
                    Displayed in Streamlit
```

## 6. LangChain Component

LangChain is used for **two** real language-understanding tasks (see
`llm_service.py` and `prompts.py`):

1. **Extraction** (`extract_conditions_from_text`) — a `ChatPromptTemplate`
   instructs the LLM to read a free-text classroom description and output
   the six numeric readings as JSON. The response is parsed with
   `JsonOutputParser` (with a manual fallback if the LLM wraps its answer in
   markdown) and validated into a `ClassroomConditions` Pydantic model,
   which clamps any out-of-range values.
2. **Explanation** (`generate_explanation`) — a second `ChatPromptTemplate`
   is given the *already-computed* fuzzy score, category, and list of fired
   rules, and asked to explain the result in plain English. The prompt
   explicitly forbids the LLM from inventing a different score — it is only
   allowed to explain the number it's given.

The LLM **never produces the final score itself** — the score always comes
from the fuzzy engine. This is what makes the "AI" component and the
"fuzzy" component two independent, verifiable parts of the pipeline (see
Section 21 "Final Audit" for why this separation matters academically).

## 7. Fuzzy Logic Component

Implemented from scratch in `fuzzy_engine.py` using `scikit-fuzzy`'s
membership-function primitives — **not** wrapped in a black-box
`ControlSystem`, so every step is inspectable:

1. **Fuzzification** — `fuzzify()` converts a crisp number into a
   membership degree (0–1) in each linguistic term, via
   `skfuzzy.interp_membership`.
2. **Membership Functions** — `build_membership_functions()` defines
   triangular/trapezoidal curves for every variable.
3. **Fuzzy Rules** — `RULES`, a list of 15 `Rule` data objects (not
   if/else code).
4. **Rule Evaluation** — `evaluate_rules()` computes each rule's firing
   strength as the **minimum** (fuzzy AND) of its antecedents' membership
   degrees.
5. **Aggregation** — `aggregate_output()` clips each output term's curve at
   its strongest firing rule, then combines all clipped curves with
   **maximum** (fuzzy OR).
6. **Defuzzification** — `defuzzify()` computes the **centroid** (center of
   gravity) of the aggregated curve via `skfuzzy.defuzz(..., 'centroid')`.

## 8. Fuzzy Variables

| Variable    | Range          | Terms                          |
|-------------|----------------|----------------------------------|
| Temperature | 10–45 °C       | Cold, Comfortable, Hot          |
| Humidity    | 0–100 %        | Low, Comfortable, High          |
| CO2         | 300–3000 ppm   | Low, Moderate, High             |
| Noise       | 20–100 dB      | Quiet, Moderate, Noisy          |
| Light       | 0–1000 lux     | Dim, Comfortable, Bright        |
| Occupancy   | 0–100 %        | Low, Moderate, High             |
| **Quality (output)** | 0–100 | Very Poor, Poor, Average, Good, Excellent |

## 9. Membership Functions

Open-ended terms (e.g. "Cold", "Hot", "Low", "High") use **trapezoidal**
functions; middle terms (e.g. "Comfortable", "Moderate") use **triangular**
functions. Exact breakpoints are in `build_membership_functions()` in
`fuzzy_engine.py`, and can be visualized live in the app under "How the
Fuzzy Logic Works" → membership function viewer.

## 10. Sample Fuzzy Rules

```
R1:  IF temperature is Comfortable AND humidity is Comfortable
     AND CO2 is Low AND noise is Quiet
     THEN quality is Excellent

R2:  IF temperature is Hot AND CO2 is High AND noise is Noisy
     THEN quality is Very Poor

R4:  IF CO2 is High AND occupancy is High
     THEN quality is Poor

R6:  IF temperature is Cold AND humidity is High
     THEN quality is Poor
```
Full list of 15 rules: see `RULES` in `fuzzy_engine.py`, or the in-app
"How the Fuzzy Logic Works" expander.

## 11. Application Workflow

**Manual mode:** sliders → `ClassroomConditions` → fuzzy engine → score +
category → LangChain explanation → results displayed.

**AI mode:** free text → LangChain extraction → `ClassroomConditions` →
fuzzy engine → score + category → LangChain explanation → results
displayed (extracted values also shown, for transparency).

## 12. Installation

```bash
git clone <your-repo-url>
cd classroom-environment-advisor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 13. Environment Variables

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

```env
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini        # optional, this is the default
LLM_BASE_URL=                # optional, only for non-OpenAI OpenAI-compatible providers
```

`LLM_BASE_URL` lets you point at any OpenAI-compatible endpoint (Groq,
OpenRouter, Together AI, etc.) if you don't have an OpenAI key — just set
`LLM_BASE_URL` and use that provider's key/model name.

## 14. Running Locally

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

Manual Assessment mode works even without setting up an API key (with a
templated fallback explanation); AI Natural Language mode requires
`LLM_API_KEY` to be set.

## 15. Streamlit Cloud Deployment

1. Push this project to a **public or private GitHub repo** (make sure
   `.env` is **not** committed — it's already in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click
   **"New app"**.
3. Select your repo, branch, and set the main file path to `app.py`.
4. Under **"Advanced settings" → Secrets**, add:
   ```toml
   LLM_API_KEY = "your_api_key_here"
   LLM_MODEL = "gpt-4o-mini"
   LLM_BASE_URL = ""
   ```
5. Click **Deploy**.

`utils.load_secrets_into_env()` automatically reads `st.secrets` on
Streamlit Cloud and injects them into `os.environ`, so the rest of the code
doesn't need to know whether it's running locally or in the cloud.

## 16. GitHub Setup

```bash
git init
git add .
git commit -m "Initial commit: Classroom Environment Quality Advisor"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

Double-check `.env` is **not** in `git status` before your first commit.

## 17. Screenshots

*(Add screenshots of the Manual Assessment tab, AI Assessment tab, and the
gauge/rule-trace expander here after running the app locally.)*

## 18. Future Scope

- Add real-time IoT sensor integration (temperature/CO2/noise sensors) to
  auto-fill the Manual Assessment sliders.
- Support multi-classroom comparison dashboards.
- Add adjustable/user-tunable fuzzy rules and membership functions from the
  UI (a rule/MF editor) for experimentation.
- Log historical scores over a school term to spot trends.
- Support additional languages in the natural-language extraction step.

## 19. Viva Preparation

**Q: What is fuzzification?**
A: Converting a precise ("crisp") number, like 29°C, into degrees of
membership in linguistic categories, like 0.3 "Comfortable" and 0.7 "Hot" —
capturing that a value can partially belong to more than one category.

**Q: What are membership functions?**
A: Mathematical curves (triangular, trapezoidal, etc.) that define how
strongly a value belongs to a linguistic term, across the variable's whole
range.

**Q: What are fuzzy rules?**
A: IF-THEN statements written in linguistic terms (e.g. "IF temperature is
Hot AND CO2 is High THEN quality is Very Poor"), evaluated using fuzzy
logic operators instead of exact comparisons.

**Q: What is defuzzification?**
A: Converting the aggregated fuzzy output back into one crisp number. This
project uses the **centroid** method — the "center of gravity" of the
aggregated output curve.

**Q: Why use fuzzy logic instead of if-else?**
A: Real-world environmental comfort is inherently gradual, not binary — 24°C
isn't suddenly "uncomfortable" one degree past a cutoff. Fuzzy logic models
that gradualness and lets multiple rules contribute proportionally to the
final score, instead of one rigid threshold overriding everything.

**Q: What does LangChain do here?**
A: It orchestrates two LLM calls: (1) turning a free-text classroom
description into structured numeric readings, and (2) turning the fuzzy
engine's numeric result into a plain-English explanation.

**Q: Why is an LLM needed at all?**
A: Users naturally describe environments in vague language ("kind of warm",
"pretty crowded"). An LLM can interpret that language and estimate
reasonable numeric values, which a rule-based parser could not do reliably
for open-ended phrasing.

**Q: How does natural language become numerical input?**
A: The LLM is prompted with the valid range and typical value for each of
the six variables, and asked to return a JSON object with its best numeric
estimate for each — which is then validated by a Pydantic model before use.

**Q: Why doesn't the LLM directly decide the final score?**
A: To keep the score consistent, explainable, and reproducible. A fuzzy
system based on fixed membership functions and rules will always give the
same score for the same numeric inputs, while an LLM's output can vary
between runs. Keeping the LLM out of the scoring itself also makes the
fuzzy logic verifiable and demonstrable on its own.

**Q: How are the two components connected?**
A: The LangChain extraction step produces a `ClassroomConditions` object;
that exact object is passed straight into the fuzzy engine's
`run_fuzzy_inference()` function. The fuzzy engine's `FuzzyResult` (score +
category + fired rules) is then passed into the second LangChain step as
plain data for the explanation prompt.

## 20. Project Structure

```
classroom-environment-advisor/
│
├── app.py                    # Streamlit UI, both modes
├── fuzzy_engine.py           # Mamdani fuzzy inference system
├── llm_service.py            # LangChain extraction + explanation
├── models.py                 # Pydantic models (ClassroomConditions, FuzzyResult)
├── prompts.py                 # LangChain prompt templates
├── utils.py                   # Secrets loading + Plotly visualizations
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── tests/
│   ├── test_fuzzy_engine.py  # Fuzzy engine unit tests
│   └── test_llm_service.py   # LangChain service error-handling tests
└── assets/                    # (optional diagrams/screenshots)
```

## 21. Testing

```bash
pip install pytest
pytest tests/ -v
```

Or, without pytest installed:
```bash
python tests/test_fuzzy_engine.py
```

Test coverage includes:
- Comfortable classroom → high score (Test Case 1)
- Hot + high CO2 + noisy classroom → low score (Test Case 2)
- Moderate conditions → medium score (Test Case 3)
- Near-ideal conditions → high score (Test Case 4)
- Score always stays within [0, 100]
- Rule strengths are valid membership degrees
- Out-of-range inputs are clamped, not rejected
- Missing required fields are rejected by Pydantic
- Missing API key raises a clear config error
- Empty/whitespace descriptions are rejected before calling the LLM
- Explanation generation falls back gracefully with no API key

> **Note on assertions:** tests check score *ranges/behavior* (e.g. "a hot,
> noisy, crowded room scores low"), not one exact number — since a fuzzy
> system's precise output depends on membership-function shape and isn't
> meant to be pinned to a single "magic" value (see project brief §17).

## Final note

**The LLM is used for natural-language understanding and explanation,
while the final classroom quality score is produced by the fuzzy
inference system.**
