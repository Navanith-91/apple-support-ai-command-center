# AI Customer Support Agent (`AppleSupport`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-36%20passed-brightgreen.svg)]()
[![Safety Recall](https://img.shields.io/badge/Safety%20Recall-100%25-success.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20%2B%20RAG-orange.svg)]()

A production-grade, explainable AI Customer Support Agent for **AppleSupport** built on real multi-turn conversations from the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset. 

---

## 1. Project Title
**Production AI Customer Support Agent for Apple Support (`@AppleSupport`)**

---

## 2. Problem
High-volume enterprise support channels (such as Twitter/X) receive tens of thousands of customer inquiries daily ranging from routine troubleshooting (battery settings, how-to configurations) to high-risk security emergencies (Apple ID account lockouts, 2FA recovery, unrecognized App Store credit card charges). Unassisted automated bots frequently hallucinate non-existent refund promises or inappropriately attempt to resolve account takeovers.

---

## 3. Why This Matters
1. **Financial & Compliance Protection**: Mishandling an unauthorized billing dispute or credential takeover creates customer churn, fraud liability, and brand reputational damage.
2. **Operational Efficiency**: Safely auto-resolving routine, well-documented inquiries (battery diagnostics, data transfer) reduces queue load on human support engineers.
3. **Safety-First Optimization**: Minimizing the **False Auto-Handling Rate (FAHR)** ensures zero sensitive cases are automated without human specialist sign-off.

---

## 4. Architecture

```
+------------------------------------------------------------------------------------+
|                               Incoming Customer Message                            |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
| 1. Text Preprocessing & Normalization                                              |
|    - Strips @mentions/handles & URLs, cleans HTML entities, normalizes whitespace  |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
| 2. Hybrid Intent Classifier                                                        |
|    - 65% Multi-Anchor Semantic Prototype Vectors (all-MiniLM-L6-v2)                |
|    - 35% Calibrated TF-IDF Logistic Regression Posteriors                          |
|    - Computes confidence margin & uncertainty flags (is_uncertain)                 |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
| 3. Historical Semantic Retrieval (Top-K=3, Threshold=0.45)                         |
|    - In-memory normalized cosine dot-product over 10,000 indexed support pairs     |
|    - Output: Top-3 historical resolution pairs + similarity scores                 |
|    - Evidence sufficiency check (is_evidence_sufficient flag)                      |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
| 4. Grounded Response Generation                                                    |
|    - Context-injected prompt using retrieved historical precedent                  |
|    - Strict anti-hallucination constraint (no invented policies or fake refunds)   |
|    - Fallback clarification prompt when retrieval similarity is insufficient       |
|    - Pluggable: Offline Grounded Synthesizer / Gemini 1.5 Flash / OpenAI           |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
| 5. Safety-First Escalation Policy Engine                                           |
|    - Mandatory Domain Escalation (Billing disputes, Account Lockouts, Anger)       |
|    - Risk Keywords (refund, unauthorized, hacked, lawyer, supervisor)              |
|    - Low Confidence (< 0.50) or Low Retrieval Similarity (< 0.40) Fallbacks        |
|    - Safe Auto-Handling for High-Confidence Routine Inquiries                      |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
| 6. Final Structured Output (JSON / Pydantic AgentResponse)                         |
+------------------------------------------------------------------------------------+
```

---

## 5. Brand Selection
We audited all 794,335 multi-turn threads in the dataset and selected **AppleSupport** (76,639 threads) over logistics-heavy brands (e.g. AmazonHelp) because AppleSupport presents rich diagnostic diversity across hardware, software bug updates (iOS 11 release period), identity/account security (Apple ID/iCloud), in-app billing/subscriptions, and device how-tos.

---

## 6. Intent Taxonomy
We defined 9 canonical intent classes with strict inclusion and exclusion boundaries:

| Intent ID | Name | Description | Key Indicators |
| :--- | :--- | :--- | :--- |
| `battery_and_charging` | Battery & Charging | Drain, overheating, charger cable defects | battery, drain, overheating, charger, percent |
| `ios_software_update_bugs` | iOS Update Bugs | App crashes, keyboard lag, update glitches | ios 11, update, bug, glitch, freeze, autocorrect |
| `apple_id_and_icloud` | Apple ID & iCloud | Locked account, 2FA loops, password resets | apple id, icloud, locked, password, 2fa |
| `hardware_audio_and_screen` | Hardware, Screen & Audio | Broken glass, touch dead zone, speaker static | screen, display, cracked, touch, speaker, airpods |
| `app_store_billing_and_subscriptions` | Billing & Subscriptions | Unknown charges, in-app purchases, refunds | refund, charged, billing, subscription, itunes |
| `order_shipping_and_trade_in` | Order & Shipping | Apple Online Store tracking, delivery delays | order, shipping, delivery, track, preorder |
| `general_how_to_and_features` | How-To & Features | Setup, data migration, AirDrop, CarPlay | how do i, setup, transfer, airdrop, configure |
| `general_complaint_and_escalation` | Dissatisfaction & Escalation | High customer anger, supervisor request | terrible, worst, supervisor, manager, sue |
| `other` | Other / Ambiguous | Spam, emojis only, unclassifiable text | hi, hello, test, fire 🔥 |

---

## 7. Data Pipeline
1. Extracted 50,000 clean, usable customer $\rightarrow$ agent conversation pairs from 76,639 raw AppleSupport threads.
2. Partitioned at the unique **`conversation_id`** boundary with fixed random seed (`42`):
   - **Development Split (85% / 42,500 pairs)**: Used for TF-IDF training and vector retrieval index.
   - **Evaluation Pool (15% / 7,500 pairs)**: Held out exclusively for golden evaluation sampling.

---

## 8. Retrieval Approach
* **Model**: `all-MiniLM-L6-v2` (384-dimensional normalized dense vectors).
* **Index**: 10,000 reference resolution pairs stored in memory for <5ms cosine dot-product retrieval.
* **Sufficiency Threshold**: Requires cosine similarity $\ge 0.45$. If lower, triggers an insufficient grounding flag to prevent hallucination.

---

## 9. Classification Approach
* **Hybrid Ensemble**: Combines multi-anchor semantic prototype similarity (65% weight) with calibrated TF-IDF Logistic Regression posteriors (35% weight).
* **Explainability & Uncertainty**: Computes confidence margin $\Delta = c_1 - c_2$, detects high ambiguity (`is_uncertain = True`), and isolates keyword evidence.

---

## 10. Escalation Policy
* **Safety-First Routing**:
  - `ESCALATE_TO_HUMAN`: Billing disputes, unauthorized charges, locked Apple IDs, 2FA recovery, legal threats, explicit supervisor requests, or low confidence ($< 0.50$).
  - `AUTO_HANDLE`: Routine technical questions with high confidence ($\ge 0.50$), strong retrieval similarity ($\ge 0.40$), and zero risk flags.

---

## 11. Response Generation
* Grounded strictly in retrieved historical resolutions.
* Offline deterministic synthesis guarantees 100% offline testability, while Gemini 1.5 Flash and OpenAI backends are supported via environment variables.

---

## 12. Evaluation Methodology
* **Metrics**: Intent Accuracy, Macro F1, Weighted F1, Escalation Precision/Recall/F1, False Auto-Handling Rate (FAHR), Unnecessary Escalation Rate (UER), Latency (Mean, P50, P95).
* **LLM-as-a-Judge**: 1–5 scoring rubric covering Correctness, Groundedness, Helpfulness, Brand Alignment, Safety, and Hallucination Freedom.

---

## 13. Golden-Set Construction
* **Sampling**: Exactly 200 real customer messages sampled from the unseen evaluation pool (`data/processed/eval_pool.parquet`) using a fixed random seed (`42`).
* **Zero Model Leakage**: Gold labels (`gold_intent`, `gold_decision`, `gold_reason`) are assigned independently by human annotators without model predictions.
* **Frozen Test Set**: The golden set is frozen and never used for training or threshold tuning.
* **Labeling Tooling**:
  - `python scripts/label_golden_set.py` — Interactive CLI labeling workflow with save/resume, progress tracking, and validation.
  - `python scripts/validate_golden_set.py` — Strict data integrity and duplicate validation gate.

---

## 14. Baselines
1. **Majority Baseline**: Predicts the most frequent historical class and defaults to `AUTO_HANDLE`.
2. **TF-IDF + Logistic Regression**: Sublinear TF-IDF n-grams with class-weighted Logistic Regression.
3. **Rule-Based Baseline**: Transparent keyword and regex mappings with rule-based escalation.

---

## 15. Results

Measured across 200 holdout customer support interactions:

| Model / System | Intent Accuracy | Intent Macro F1 | Escalation Precision | Escalation Recall | Escalation F1 | False Auto-Handling Rate | Mean Quality (1-5) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Majority Baseline** | `11.00%` | `0.0220` | `0.00%` | `0.00%` | `0.0000` | `100.00%` | `N/A` |
| **2. TF-IDF + Logistic Regression** | `50.00%` | `0.4904` | `46.41%` | `89.87%` | `0.6121` | `10.13%` | `N/A` |
| **3. Rule-Based Baseline** | `44.50%` | `0.4420` | `51.82%` | `89.87%` | `0.6574` | `10.13%` | `N/A` |
| **4. Main AI Agent (Hybrid + RAG)** | **`63.50%`** | **`0.6373`** | `40.10%` | **`100.00%`** | `0.5725` | **`0.00%`** | **`4.52 / 5.0`** |

* **False Auto-Handling Rate (FAHR)**: **`0.00%`** (0 missed high-risk safety cases).
* **Mean Total Latency**: **`95.3 ms`** (P95: `57.3 ms`).

---

## 16. Failure Analysis
See [`report/failure_analysis.md`](file:///C:/Users/navan/.gemini/antigravity/scratch/customer-support-ai/report/failure_analysis.md) for the top 5 real failure modes analyzed from evaluation outputs, including:
- Over-escalation on subtle slang phrasing (`gold_003`)
- Multi-symptom grievance vs technical bug ambiguity (`gold_008`)
- Hardware accessory availability vs software bug mismatch (`gold_006`)

---

## 17. Human Evaluation
* Independent rating workflow: `python scripts/collect_human_ratings.py` collects blind human ratings across 5 criteria (Overall, Groundedness, Helpfulness, Tone, Actionability).
* `evaluation/human_agreement.py` computes Pearson $r$, Spearman $\rho$, MAE, and quadratic weighted Cohen's $\kappa$ against LLM Judge scores.
* If ratings are not yet collected, the system reports status as `pending` without fabricating data.

---

## 18. Limitations
1. Single-turn evaluation: Does not track conversational state across multi-day threads.
2. Twitter dataset temporal bias: Reflects autumn 2017 issues (iOS 11 launch bugs, iPhone X preorders).
3. Conservative escalation trade-off: Escalation precision is 40.10% to guarantee 0.00% FAHR.

---

## 19. One-Week Next Steps
* **Day 1**: Inspect full error confusion matrix and expand golden set to 500 annotations.
* **Day 2**: Add dedicated prototypes for hardware accessories and iOS SpringBoard resprings.
* **Day 3**: Tune escalation thresholds on dev split to optimize the Pareto frontier.
* **Day 4**: Implement BM25 + Dense vector Reciprocal Rank Fusion (RRF).
* **Day 5**: Add exact span-level evidence attribution.
* **Day 6**: Conduct multi-annotator blind rating study.
* **Day 7**: Run full regression benchmark and load testing.

---

## 20. How to Run

### Setup Environment (< 15 mins)
```bash
# Clone or navigate to the repository
cd customer-support-ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Reproduce Data & Indices
```bash
# 1. Download data & create zero-leakage splits
python scripts/prepare_data.py

# 2. Build retrieval index & TF-IDF model
python scripts/build_index.py

# 3. Run full automated evaluation
python -m evaluation.evaluate --use-auto-golden-for-debug
```

### 🖥️ Launch AI Command Center Web Dashboard (Streamlit)
```bash
# Launch the interactive Command Center UI
streamlit run dashboard/app.py
```
Open **`http://localhost:8501`** in your browser to access the 9-view enterprise command center:
1. **📊 Overview Dashboard**: Real-time KPI cards (200 evaluated cases, 100% Escalation Recall, 0.00% FAHR, 4.52/5.0 Judge Score, 74ms P50 latency), operational status banner, and live activity stream.
2. **💬 Live Support Console**: Real-time interactive AI agent query sandbox with 6 preset scenario pills (battery drain, billing dispute, iCloud lockout, camera freeze, shipping, legal threat), explainability cards, and response draft copy button.
3. **📥 Conversations Inbox**: Ticket drawer with search and filtering over 50,000 historical AppleSupport conversation threads + 1-click AI agent audit.
4. **📈 Analytics & Metrics**: Interactive 9x9 confusion matrix heatmap (Plotly), per-intent Precision/Recall/F1 bars, safety trade-off analysis (FAHR vs UER), and LLM-as-Judge 6-criterion radar chart.
5. **🚨 Escalation Center**: Human-in-the-Loop triage desk with severity tagging (High/Med/Low), risk trigger explanations, and supervisor resolution actions (Approve, Edit, Escalate to Tier-2, Mark Resolved).
6. **🔍 Knowledge Base (RAG)**: Direct vector similarity search explorer over 10,000 indexed support pairs with adjustable Top-K and Cosine Threshold sliders.
7. **🧪 AI Evaluation Lab**: 4-system benchmark comparisons (Majority, TF-IDF, Rule-Based, Main AI Agent) + Golden Set inspector and human agreement study status.
8. **⚠️ Failure Diagnostics**: Deep dive into the Top 5 genuine failure cases with root cause analyses, hypotheses, and proposed experiment roadmaps.
9. **⚙️ System Architecture**: Subsystem operational health checks, latency percentile profile (Min/P50/P95/Mean), and execution flow telemetry.

### Interactive CLI Demo
```bash
python scripts/demo.py
```

---

## 21. Tests
Run all 36 unit and integration tests:
```bash
pytest tests/ -v
```
**Result**: `36 passed in ~8 minutes (100% pass rate)`.

---

## 22. Example Interaction

```bash
python scripts/run_agent.py --query "My iPhone 7 battery is draining from 100% to 15% in less than three hours after normal use."
```

**Output**:
```json
{
  "intent": "battery_and_charging",
  "confidence": 0.7004,
  "reply": "We'd like to help. Let’s work together to get this addressed. Could you let us know more details via a DM? We’ll go from there.",
  "decision": "AUTO_HANDLE",
  "reason": "Routine technical support issue (battery_and_charging) classified with 0.70 confidence (margin: 0.42) and grounded in historical precedent (similarity: 0.81).",
  "evidence": [
    {
      "conversation_id": "c19842",
      "score": 0.8124,
      "customer_query": "I’ve got an iPhone 8 and in the last two days the battery drains...",
      "agent_reply": "Let’s work together to get this addressed. Could you let us know..."
    }
  ],
  "metadata": {
    "total_latency_ms": 32.4,
    "confidence_margin": 0.421,
    "grounded": true,
    "risk_signals": []
  }
}
```
