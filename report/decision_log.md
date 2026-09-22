# Engineering Decision Log

This log records 15 non-obvious, production-grade architectural and engineering decisions made during the design, implementation, and evaluation of the Customer Support AI Agent.

---

### Decision 1: Selection of `AppleSupport` over `AmazonHelp` and Airline Brands
* **Decision**: Target AppleSupport from among 794k+ multi-turn customer support conversations in the Twitter dataset.
* **Why**: While AmazonHelp has slightly higher raw volume (81k vs 76k threads), >70% of Amazon support queries are generic shipping/delivery tracking inquiries. AppleSupport provides a significantly richer diagnostic support surface spanning hardware, software bug updates (iOS 11 release period), identity/account security (Apple ID/iCloud), in-app billing/subscriptions, and device how-tos.
* **Alternatives Considered**: AmazonHelp (too logistics-heavy), SpotifyCares (lacks hardware dimension), Uber_Support (geolocation-specific).
* **Trade-off**: Requires handling nuanced multi-symptom complaints (e.g., battery drain caused by an iOS software update vs. physical battery degradation), increasing classification complexity.
* **Evidence**: Statistical brand audit in `data/processed/brand_stats.csv` showing 76,639 threads with high diagnostic diversity.

---

### Decision 2: Conversation & Customer-Author Level Splitting (85% Dev / 15% Eval)
* **Decision**: Partition datasets strictly at the unique `conversation_id` boundary rather than random row-level splitting.
* **Why**: Customers frequently send multiple tweets in a single thread or follow up on previous tickets. Random row splitting causes severe data leakage where the model memorizes customer writing quirks or identical thread replies in the test set.
* **Alternatives Considered**: Random train/test row split (causes severe data leakage), temporal time-based split (uneven class distribution due to iOS 11 launch spike).
* **Trade-off**: Slightly reduces the number of independent training instances compared to row-level permutation, but guarantees 100% genuine generalization measurement.
* **Evidence**: Zero conversation ID overlap between `dev_split.parquet` (42,500 pairs) and `eval_pool.parquet` (7,500 pairs).

---

### Decision 3: Compact 9-Class Intent Taxonomy with Clear Inclusion/Exclusion Boundaries
* **Decision**: Constrain the intent taxonomy to 9 well-defined classes (`battery_and_charging`, `ios_software_update_bugs`, `apple_id_and_icloud`, `hardware_audio_and_screen`, `app_store_billing_and_subscriptions`, `order_shipping_and_trade_in`, `general_how_to_and_features`, `general_complaint_and_escalation`, `other`).
* **Why**: Fine-grained taxonomies (>20 classes) suffer from high human labeling ambiguity and low inter-annotator agreement on short Twitter messages (<280 characters). 9 classes capture all primary operational routing paths at Apple Support.
* **Alternatives Considered**: 3-class coarse taxonomy (too vague for actionable routing), 25-class granular taxonomy (severe inter-annotator ambiguity).
* **Trade-off**: Subtle sub-issues (e.g. iPad SIM card error vs. Apple Pencil pairing) are grouped under overarching hardware or how-to buckets rather than distinct micro-intents.
* **Evidence**: High inter-annotator clarity and complete coverage documented in `evaluation/labeling_guidelines.md`.

---

### Decision 4: Decoupled Intent Classification and Historical Retrieval Stages
* **Decision**: Execute Intent Classification first, followed independently by Semantic Historical Retrieval over the indexed knowledge base.
* **Why**: High-level routing (escalation vs. automation) requires knowing *what type of problem* it is (e.g., billing vs. hardware), whereas response generation requires *how Apple historically resolved this exact symptom*. Decoupling allows independent tuning, caching, and fallback logic.
* **Alternatives Considered**: Single end-to-end LLM prompt doing classification, retrieval, and response in one prompt (expensive, slow, unexplainable).
* **Trade-off**: Adds a small sequential processing step (~20ms), but drastically improves system explainability and modularity.
* **Evidence**: Mean execution latency of 95.3 ms with isolated per-stage latency telemetry.

---

### Decision 5: Hybrid Ensemble Classifier with Confidence Margins & Uncertainty Gating
* **Decision**: Combine Sentence-Transformer multi-anchor semantic prototype embeddings (65% weight) with calibrated TF-IDF Logistic Regression posteriors (35% weight), tracking confidence margin between top-1 and top-2 intents.
* **Why**: Embeddings capture semantic paraphrasing and conceptual similarity, while TF-IDF excels at exact lexical domain tokens (e.g., "iOS 11.1", "itunes.com/bill", "iCloud", "2FA"). Tracking margin $\Delta = c_1 - c_2$ allows explicit uncertainty detection.
* **Alternatives Considered**: Pure TF-IDF (poor generalization on paraphrases), pure dense embedding (misses domain keywords like exact error codes).
* **Trade-off**: Requires maintaining both the embedding pipeline and the TF-IDF feature vocabulary in memory (~120MB combined footprint).
* **Evidence**: Main agent intent accuracy reached 63.50% vs 50.00% for pure TF-IDF and 44.50% for rule-based matching.

---

### Decision 6: Dual Escalation Engine (Mandatory Sensitive Intents + Dynamic Risk Signals)
* **Decision**: Implement a two-tiered escalation policy: (1) Deterministic mandatory escalation for high-risk domains (`app_store_billing_and_subscriptions`, locked `apple_id_and_icloud`, supervisor requests), and (2) Multi-signal heuristic risk scoring (low confidence, low retrieval similarity, angry language, legal/manager keywords).
* **Why**: Financial and account credentials require strict compliance and authenticated workflows that an autonomous chatbot should never finalize. Heuristic signals handle edge-case ambiguity.
* **Alternatives Considered**: Pure classifier confidence thresholding (fails on confidently misclassified account takeovers).
* **Trade-off**: Increases the Unnecessary Escalation Rate (UER) on mild billing inquiries, but drives False Auto-Handling Rate (FAHR) to 0.00%.
* **Evidence**: 0.00% False Auto-Handling Rate measured on golden evaluation set.

---

### Decision 7: Strict Anti-Hallucination Grounding via Historical Precedent
* **Decision**: Constrain response drafting to strictly synthesize resolutions from Top-K retrieved historical support pairs, prohibiting the LLM from inventing policies, refund timelines, or hardware repair costs.
* **Why**: Hallucinated commitments (e.g. "We will issue a full refund within 24 hours") create severe customer friction and legal liability.
* **Alternatives Considered**: Unconstrained generation with general LLM weights (produces plausible but fabricated repair policies).
* **Trade-off**: The agent will ask clarifying diagnostic questions when historical precedent is missing rather than guessing an answer.
* **Evidence**: 5.00/5.00 Hallucination Freedom score awarded by LLM Judge.

---

### Decision 8: Top-K=3 Retrieval with Cosine Similarity Thresholding (0.45)
* **Decision**: Set retrieval depth to Top-3 with an absolute cosine similarity threshold of 0.45.
* **Why**: Providing 3 high-quality examples gives the generator enough context without cluttering the prompt with irrelevant noise. If top similarity falls below 0.45, the system flags `low_retrieval_similarity` and triggers escalation.
* **Alternatives Considered**: Top-1 only (brittle to outlier replies), Top-10 (causes prompt dilution and hallucination).
* **Trade-off**: Queries with obscure phrasing may fall below 0.45 and escalate even if a marginal match exists.
* **Evidence**: 91.0% evidence sufficiency rate across evaluated holdout interactions.

---

### Decision 9: Safety-First Metric Prioritization: False Auto-Handling Rate (FAHR)
* **Decision**: Treat False Auto-Handling Rate (FAHR: predicted `AUTO_HANDLE` when gold was `ESCALATE_TO_HUMAN`) as the primary optimization objective over raw intent accuracy.
* **Why**: In customer service operations, an unnecessary human escalation costs ~$3–5 in agent time, but auto-handling a fraudulent billing dispute or locked security account can result in customer churn, compliance fines, or security breaches.
* **Alternatives Considered**: Optimizing exclusively for macro F1 or accuracy (leads to catastrophic safety failures on high-risk disputes).
* **Trade-off**: Slightly depresses automation rate (auto-handling ~60% of volume rather than 85%), prioritizing brand safety.
* **Evidence**: 0.00% FAHR vs 100.00% FAHR for majority class baseline.

---

### Decision 10: Multi-Tiered Generation Backend (Offline Fallback + Gemini/OpenAI API)
* **Decision**: Implement a pluggable generation interface that executes deterministically offline using historical agent synthesis, while seamlessly connecting to Gemini 1.5 Flash or OpenAI when API keys are present.
* **Why**: Guarantees that unit tests, benchmark harnesses, and local interview demos run instantly in <15 minutes with zero cost or internet dependencies, while enabling full GenAI capabilities in production.
* **Alternatives Considered**: Cloud API only (breaks offline grading and CI/CD pipelines), template regex only (lacks GenAI flexibility).
* **Trade-off**: Requires maintaining both the offline grounded synthesizer and LLM prompt templates.
* **Evidence**: Full test suite passes offline in <90 seconds.

---

### Decision 11: Normalized Vector Dot-Product Indexing in NumPy/Pickle
* **Decision**: Use L2-normalized embeddings with cosine dot-product indexing stored in optimized pickle files rather than heavy external vector database services.
* **Why**: For 10,000–50,000 support pairs, in-memory NumPy matrix multiplication computes Top-K retrieval in <5ms with zero external infrastructure overhead (no Docker, no server daemon).
* **Alternatives Considered**: Pinecone / Qdrant cloud services (requires API credentials and network round-trips), FAISS (heavy binary dependencies on Windows).
* **Trade-off**: Does not scale to billions of vectors without migrating to FAISS/Milvus/Qdrant, but ideal for single-brand deployment (<100k conversations).
* **Evidence**: <5ms retrieval latency measured in benchmarks.

---

### Decision 12: Zero Raw Data in Version Control
* **Decision**: Keep raw datasets (`*.parquet`, `*.csv`) in `.gitignore` and automate verified on-demand downloads via dataset mirrors with checksum verification.
* **Why**: Adheres to production repository hygiene, preventing repository bloat from multi-hundred-megabyte binary dumps while maintaining 100% reproducible environments.
* **Alternatives Considered**: Committing 200MB+ raw parquet directly into git repository (slow clones, bad Git practice).
* **Trade-off**: Initial setup requires running `python scripts/prepare_data.py` to download raw data.
* **Evidence**: Clean repository size with clear reproduction scripts.

---

### Decision 13: Domain-Aware Text Preprocessing
* **Decision**: Strip external user handles and URLs but preserve specific diagnostic casing and punctuation patterns (e.g. error codes, uppercase frustration markers).
* **Why**: Removing handles prevents the model from overfitting to user names (`@115858`), but preserving punctuation allows the escalation engine to catch frustration signals (`????`, `!!!!`).
* **Alternatives Considered**: Aggressive lowercasing and punctuation stripping (destroys frustration signals).
* **Trade-off**: Text normalization regexes must be carefully unit-tested to avoid stripping legitimate model numbers (e.g. `iPhone 8 Plus`).
* **Evidence**: `tests/test_preprocessing.py` verifies handle removal, entity unescaping, and punctuation preservation.

---

### Decision 14: Three Meaningful Baselines for Fair Benchmarking
* **Decision**: Benchmark against three diverse baselines: (1) Majority Class Baseline, (2) TF-IDF + Logistic Regression, and (3) Transparent Rule-Based Keyword Matching.
* **Why**: Provides a rigorous progression from trivial heuristics to classical ML to dense semantic RAG, preventing inflated claims of AI capability.
* **Alternatives Considered**: Comparing only against Majority Class (creates a weak strawman comparison).
* **Trade-off**: Requires implementing and evaluating three separate baseline systems.
* **Evidence**: 4-system comparison table in Section 11 of the technical report.

---

### Decision 15: Blind Human Evaluation Workflow Without LLM Score Exposure
* **Decision**: Design `scripts/collect_human_ratings.py` to explicitly conceal the LLM judge rating from human annotators until ratings are recorded.
* **Why**: Exposing the LLM judge score before human rating introduces severe anchoring and confirmation bias, invalidating inter-rater reliability calculations.
* **Alternatives Considered**: Showing LLM score with a checkbox "Do you agree?" (heavily biased toward agreeing with the LLM).
* **Trade-off**: Requires evaluators to formulate independent scores from raw customer messages and retrieved evidence.
* **Evidence**: `evaluation/human_agreement.py` computes true independent correlation and Cohen's $\kappa$.
