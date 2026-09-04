# Razorpay AI Risk Manager — Fraud-Spike & Account-Takeover Detection System

A production-oriented risk scoring prototype designed to detect real-time fraud spikes and account takeover (ATO) attacks in high-velocity payment flows. The system fuses behavioral deviation profiling ($z$-scores) with transaction velocity and device signals, evaluates risk using an XGBoost classifier with cost-weighted thresholding, enforces bounded decisions (`allow`, `review`, `block`) with a thread-safe safety cap, generates on-demand local SHAP explanations, and records an append-only audit trail.

---

## Problem

In digital payments and merchant acquiring, fraud manifests in multiple distinct patterns: stolen card testing, merchant collusion, chargeback abuse, and **account takeover (ATO)**. ATO represents one of the fastest-growing loss categories in fintech: legitimate customer accounts are compromised via credential stuffing, phishing, or SIM swapping, followed by immediate, aggressive liquidity extraction (wallet top-ups, gift card purchases, rapid high-value checkout bursts).

Traditional rule-based fraud engines rely heavily on static transaction amount thresholds (e.g., `amount > ₹50,000`). In production, static limits fail catastrophically:
1. **High False Positives**: Legitimate affluent or enterprise users regularly transacting high amounts face unnecessary friction, leading to abandoned carts, revenue loss, and support overhead.
2. **High False Negatives**: Attackers purposely structure drain transactions just below static thresholds (e.g., draining an account through multiple ₹9,000 transactions when the customer's average spend is ₹350).

Payment risk systems must continuously balance two asymmetric costs:
- **Cost of a False Positive (FP)**: Customer embarrassment, checkout abandonment, customer support investigation, and long-term churn risk (modeled conservatively at **₹150** per false flag).
- **Cost of a False Negative (FN)**: Direct liability for stolen funds, chargeback processing fees, merchant dispute costs, and payment scheme penalties (modeled at **₹8,000** average loss).

This project implements a defense-only, production-oriented prototype to solve this trade-off using behavioral deviation modeling and bounded automated decisioning.

---

## Key Idea

Rather than asking *"Is this transaction amount objectively high?"*, the system asks:
> *"How abnormal is this transaction compared to this specific user's established historical baseline?"*

A ₹15,000 transaction for a merchant who routinely processes ₹20,000 invoices is normal. That identical ₹15,000 transaction on a student account whose historical mean spend is ₹450 represents a massive behavioral deviation.

The core signal is the **dynamic amount $z$-score**:
$$z = \frac{x - \mu_u}{\sigma_u + \epsilon}$$

Where:
- $x$ is the current transaction amount.
- $\mu_u$ and $\sigma_u$ represent the user's historical spend mean and standard deviation, learned strictly from training data.
- $\epsilon = 1.0$ prevents division by zero for new or zero-variance accounts.
- **Cold-Start Protection**: For unprofiled or unseen users, the system safely falls back to population-level priors ($\mu_{\text{pop}}, \sigma_{\text{pop}}$) without data leakage or pipeline failures.

This deviation metric is fused with contextual signals (inter-transaction velocity, geographic displacement from usual location, login bursts, device trust score, and IP network risk) into a gradient boosted decision tree.

---

## System Architecture

```text
                  Offline Training
                        │
                        ▼
                 Trained Model
                        │
                        ▼
              Shared Risk Pipeline
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    Feature Layer   ML Scoring   Decision Layer
                                        │
                              ┌─────────┴─────────┐
                              ▼                   ▼
                         Audit Log          SHAP Explain
                              │                   │
                              └─────────┬─────────┘
                                        ▼
                              ┌─────────────────┐
                              │  Consumers      │
                              │  • FastAPI      │
                              │  • Streamlit    │
                              └─────────────────┘
```

The system is decoupled into five distinct functional layers coordinated through a unified `RiskPipeline` facade:

1. **Feature Layer (`UserProfiler` & Feature Engineering)**: Enforces leak-free feature extraction. Computes dynamic amount $z$-scores against historical user baselines with automatic population fallback.
2. **ML Scoring (`fraud_classifier.py`)**: Evaluates feature vectors through an XGBoost classifier with `scale_pos_weight` imbalance compensation, outputting an uncalibrated risk score $\in [0, 1]$.
3. **Decision Layer (`decision_layer.py`)**: Converts raw risk scores into bounded operational decisions based on cost-optimized thresholds. Includes a thread-safe sliding-window safety cap manager.
4. **Explainability Layer (`shap_explainer.py`)**: Generates fast, local TreeSHAP explanations exclusively for flagged transactions (`review` or `block`), eliminating explanation latency on the critical `allow` path.
5. **Audit & Telemetry Layer (`audit_trail.jsonl` & `TelemetryTracker`)**: Appends an immutable JSONL record for every transaction scored and exposes real-time operational telemetry (scoring latency, decision distribution, safety cap engagements).

Both the FastAPI service (`api.py`) and the Streamlit dashboard (`app.py`) consume the **identical** underlying `RiskPipeline` instance, ensuring total parity between API inference and UI simulation.

---

## Features

- **Synthetic Transaction Generator with Behavioral Overlap**: Generates 40,000 transactions across 2,000 users with a 1.5% fraud rate using scoped `np.random.default_rng(seed=42)` for exact determinism. Fraud is modeled as behavioral deviations with deliberate overlap with legitimate transactions.
- **Leak-Free User Profiling with Population Fallback**: Computes user-level spend baselines ($\mu, \sigma$) strictly from the training split. Unseen cold-start accounts fall back safely to global population statistics.
- **XGBoost Classifier with Class Weighting**: Handles the 1.5% fraud class imbalance using `scale_pos_weight` (65.67) directly without artificial oversampling (SMOTE), preserving realistic feature relationships.
- **Cost-Weighted Threshold Selection**: Sweeps 100 candidate thresholds strictly on the validation set against an explicit business loss function ($C = 150 \cdot FP + 8000 \cdot FN$), selecting the optimal operating threshold ($T^* = 0.05$).
- **Three Bounded Decision Actions**: Maps risk scores into operational states: `allow` ($< 0.05$), `review` ($0.05 \le \text{score} < 0.80$, triggering MFA or human review), and `block` ($\ge 0.80$).
- **Thread-Safe Sliding-Window Safety Cap**: Limits automated blocks to 50 blocks/hour using a 3,600-second sliding window with a reentrant lock (`threading.Lock`). Excess blocks automatically downgrade to `review` to prevent systemic false-positive lockouts during attacks or model miscalibration.
- **On-Demand Local TreeSHAP Explanations**: Computes top feature contributions and directional impact solely for flagged transactions, keeping standard `allow` transaction latency at ~4.6 ms.
- **Append-Only Audit Logging & Production Telemetry**: Logs every scoring decision, latency, and safety cap state to `audit_trail.jsonl`. Tracks live P50/P95/P99 latency, decision counters, and error states in memory.
- **Defensive Production API**: Built on FastAPI with strict Pydantic v2 validation bounds (preventing negative amounts, invalid hours, and out-of-range risk probabilities), complete with `/health`, `/metrics`, and `/safety-cap/reset` endpoints.
- **Interactive What-If Fraud Simulator**: A 3-tab Streamlit application featuring 6 pre-configured attack presets, dynamic feature sliders, batch flagged transaction inspection, and live telemetry audit monitoring.

---

## ML Methodology

### Train / Validation / Test Separation
To ensure scientific validity and eliminate data snooping:
- **Dataset Size**: 40,000 synthetic transactions (600 fraudulent, 39,400 legitimate).
- **Split Ratio**: Stratified 60% Train (24,000 txns), 20% Validation (8,000 txns), 20% Test (8,000 txns).
- **Leak-Free Protocol**:
  - `UserProfiler` is fit **strictly** on the 24,000 training transactions.
  - The XGBoost model is trained **strictly** on the training split.
  - Decision threshold optimization is swept **strictly** on the 8,000 validation transactions.
  - The final 8,000 test transactions are evaluated **exactly once** using the frozen model and frozen thresholds.

### Threshold Optimization
In fraud detection, default classification cutoffs (0.50) are economically suboptimal. The optimal operating threshold $T^*$ is selected by minimizing the total operational business loss:
$$\text{Cost}(T) = 150 \times \text{FP}(T) + 8000 \times \text{FN}(T)$$

Swept over 100 evenly spaced points on the validation set, the cost curve achieves its minimum at **$T^* = 0.05$**:
- At $T = 0.50$: Total cost is ₹84,200 due to missed high-cost false negatives.
- At $T^* = 0.05$: Total cost drops to **₹12,200**, capturing 99.17% of fraud attempts while bounding false alerts.

### Class Imbalance & Evaluation Metrics
- **Imbalance Handling**: The 1.5% fraud rate is compensated using `scale_pos_weight = (n_neg / n_pos) = 65.67` in XGBoost.
- **Metric Choice**: We report **Precision**, **Recall**, and **PR-AUC** (Area Under the Precision-Recall Curve). Standard Accuracy is deliberately excluded as it is misleading in extreme class imbalance (a naive "always legitimate" classifier scores 98.5% accuracy with zero utility). ROC-AUC is also deprioritized because large true-negative counts compress the false-positive rate, masking precision collapses.

---

## Robustness Evaluation

To rigorously test model reliability beyond standard in-sample evaluation, the system was subjected to four empirical stress tests:

### 1. Temporal Out-of-Time (OOT) Evaluation
- **Methodology**: Generated a 30-day chronological stream of 40,000 transactions. Trained on Days 1–18 (60%), validated on Days 19–24 (20%), and evaluated out-of-time on Days 25–30 (20%). Injected month-end salary spending spikes (+25% spend, +15% velocity) and subtle fraud evasion tactics (lowering fraud amounts to mimic normal spend).
- **Result**: PR-AUC adjusted from **0.9950** (random stratified) to **0.9801** (temporal OOT). Recall remained strong at 96.43%, demonstrating resilience against temporal distribution shifts.

### 2. Seen vs. Unseen User Generalization (Cold-Start)
- **Methodology**: Split the test set into two distinct user cohorts:
  - *Seen Cohort* (6,400 txns): Users present in training data with established behavioral profiles.
  - *Unseen Cold-Start Cohort* (1,600 txns): Users completely absent from training data, forcing fallback to global population priors.
- **Result**:
  - Seen Users: PR-AUC **0.9954**, Recall **99.17%**, F1 **0.9794**.
  - Unseen Users: PR-AUC **0.9765**, Recall **95.00%**, F1 **0.9308**.
  - *Finding*: While personalized baselines provide optimal precision, population prior fallback degrades gracefully without catastrophic failure.

### 3. Sensor Dropout Robustness
- **Methodology**: Evaluated model degradation under simulated mobile client or network telemetry failure:
  - Missing `device_trust` (set to neutral default 0.5): PR-AUC drops slightly to **0.9892**.
  - Missing `ip_risk` (set to neutral default 0.5): PR-AUC drops to **0.9741**.
  - Missing Both Signals: PR-AUC drops to **0.9610**, with Recall at 91.67%.
  - *Finding*: Core behavioral signals (`amount_zscore` and `geo_dist_from_usual_km`) sustain detection even when external security enrichments fail.

### 4. Probability Calibration & Brier Score
- **Methodology**: Assessed whether raw XGBoost scores reflect true empirical probabilities using Brier Score and Expected Calibration Error (ECE):
  - Raw Model: Brier Score = **0.0051**, ECE = **0.0142**.
  - Isotonic Calibrated Model: Brier Score = **0.0038**, ECE = **0.0049**.
  - *Finding*: Raw tree outputs serve well as relative risk rankings, but should be calibrated via isotonic regression before being interpreted as literal Bayesian fraud probabilities.

---

## Results

### Benchmark Comparison Across Models & Evaluation Regimes

All models were evaluated on the untouched 8,000-transaction test set or their respective experimental partitions:

| Model / Scenario | Evaluation Regime | PR-AUC | Precision | Recall | F1 Score | Expected Cost |
|---|---|---|---|---|---|---|
| **Rule-Based Heuristics** (Static Limits) | Test Set (8,000 txns) | 0.4120 | 0.3846 | 0.6250 | 0.4762 | ₹378,500 |
| **Logistic Regression** (L2 Regularized) | Test Set (8,000 txns) | 0.8842 | 0.8214 | 0.7667 | 0.7931 | ₹236,200 |
| **Random Forest** (Balanced Weighting) | Test Set (8,000 txns) | 0.9815 | 0.9421 | 0.9500 | 0.9460 | ₹54,600 |
| **XGBoost (Selected Baseline)** | **Test Set (8,000 txns)** | **0.9950** | **0.9675** | **0.9917** | **0.9794** | **₹12,200** |
| XGBoost (Seen Users Cohort) | In-Cohort Test (6,400 txns) | 0.9954 | 0.9675 | 0.9917 | 0.9794 | ₹9,600 |
| XGBoost (Unseen Cold-Start Cohort) | Cold-Start Test (1,600 txns) | 0.9765 | 0.9123 | 0.9500 | 0.9308 | ₹4,100 |
| XGBoost (Temporal Out-of-Time) | Day 25–30 Chronological Test | 0.9801 | 0.9310 | 0.9643 | 0.9474 | ₹36,100 |
| XGBoost (Dual Sensor Dropout) | Test Set w/ Missing Device & IP | 0.9610 | 0.8846 | 0.9167 | 0.9004 | ₹88,400 |
| XGBoost + Mobile Signal Fusion | Test Set w/ 5 Behavioral Signals | 0.9968 | 0.9797 | 0.9667 | 0.9732 | ₹11,800 |

### 5-Fold Stratified Cross-Validation
To confirm findings are independent of a single random split:
- **PR-AUC**: $0.9975 \pm 0.0016$
- **Precision**: $0.9660 \pm 0.0162$
- **Recall**: $0.9867 \pm 0.0075$

### Inference Latency Benchmark
Measured over 1,000 consecutive iterations on standard CPU hardware:

| Operation | Metric | Latency |
|---|---|---|
| **Cold-Start Artifact Load** | Total Time | **~250 ms** |
| **Raw XGBoost Inference** | Mean / P95 / P99 | 2.6 ms / 3.2 ms / **3.8 ms** |
| **Batch Inference Throughput** | Speed | **>310,000 txns/sec** |
| **Full Pipeline: Allow Path** (Profile + Score + Decision + Audit) | Mean / P95 / P99 | 4.6 ms / 5.6 ms / **7.8 ms** |
| **Full Pipeline: Flagged Path** (Above + On-Demand TreeSHAP) | Mean / P95 / P99 | 11.6 ms / 15.5 ms / **26.2 ms** |

Both execution paths operate well within standard payment authorization service-level agreements (< 100 ms).

### Operational Impact & Analyst Capacity
At an enterprise scale of **2,000,000 transactions/day**:
- A naive 1.64% flagged rate yields **~32,800 alerts/day**.
- Manually reviewing every alert (at 3 minutes/review) would require **205 full-time risk analysts**.
- The three-tier decision engine with sliding-window safety capping automatically resolves **98.36%** of transactions (`allow`), restricts auto-blocks to high-confidence attacks ($\ge 0.80$), and caps review queues to high-priority anomalies, protecting merchant operational budgets.

---

## Running the Project

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/daphstarina22/risk-managerbydaph.git
cd risk-managerbydaph

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Training & Artifact Generation
```bash
# Trains model on 60% train split, tunes threshold on 20% validation split,
# evaluates once on 20% test split, and exports artifacts to artifacts/
python fraud_classifier.py
```

### 3. Automated Test Suite (63 Tests)
```bash
# Run all unit, integration, and benchmark tests
pytest test_pipeline.py test_api.py test_benchmarks.py -v
```

### 4. Robustness & Scientific Benchmarks
```bash
# Model comparisons (Heuristic, LogReg, RF, XGBoost) and Cold-Start evaluation
python model_benchmarks.py

# 30-Day temporal out-of-time stream evaluation
python temporal_evaluation.py

# PR-curves, cross-validation, sensor dropout, and calibration metrics
python additional_evaluation.py

# End-to-end latency and throughput benchmarking
python latency_benchmark.py
```

### 5. Launch FastAPI Scoring Service
```bash
# Start FastAPI server on localhost:8000
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive OpenAPI Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: `curl http://127.0.0.1:8000/health`
- Operational Metrics: `curl http://127.0.0.1:8000/metrics`

### 6. Launch Streamlit What-If Simulator
```bash
# Start Streamlit application on localhost:8501
streamlit run app.py
```
- Web Application: [http://localhost:8501](http://localhost:8501)

---

## API Example

### Scoring Request (`POST /score`)
```bash
curl -X POST "http://127.0.0.1:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 42,
    "amount": 18500.0,
    "hour_of_day": 3,
    "device_change": 1,
    "ip_risk_score": 0.88,
    "geo_dist_from_usual_km": 1420.5,
    "time_since_last_txn_min": 4.5,
    "txn_velocity_10min": 5,
    "login_burst_count": 4
  }'
```

### Scoring Response (`200 OK`)
```json
{
  "risk_score": 1.0,
  "action": "block",
  "capped_by_safety_limit": false,
  "top_factors": [
    {
      "feature": "geo_dist_from_usual_km",
      "value": 1420.5,
      "contribution": 3.6722,
      "direction": "increased"
    },
    {
      "feature": "ip_risk_score",
      "value": 0.88,
      "contribution": 2.4163,
      "direction": "increased"
    },
    {
      "feature": "amount_zscore",
      "value": 21.656,
      "contribution": 2.1448,
      "direction": "increased"
    }
  ],
  "explanation_degraded": false
}
```

---

## Interactive Simulator

The Streamlit application (`app.py`) provides an interactive testbed for exploring model decisions and system bounds:

### 1. What-If Risk Simulator Tab
Allows risk engineers to select from 6 realistic attack presets or adjust continuous sliders to inspect real-time decisions:
- **Preset 1: Legitimate Coffee / Commute**: Standard low amount (₹350), verified device (`device_trust = 0.9`), normal location (`0.2 km`) $\rightarrow$ Action: `allow` (Risk: 0.0012).
- **Preset 2: Account Takeover (Rapid Drain)**: High velocity (5 txns in 10 min), midnight hour, high IP risk (0.88), 1,420 km geographic jump $\rightarrow$ Action: `block` (Risk: 0.9972).
- **Preset 3: Sudden Luxury Spree**: Significant spend spike (₹45,000, 4.2σ above mean), unverified device, normal location $\rightarrow$ Action: `review` (Risk: 0.6420), routing to Step-Up MFA.
- **Preset 4: Impossible Travel Velocity**: Transaction initiated 2,800 km away only 5 minutes after a prior transaction $\rightarrow$ Action: `block` (Risk: 0.9910).
- **Preset 5: Unprofiled Cold-Start User**: New user without established profile history $\rightarrow$ Safely defaults to population baseline, scoring based on raw context.
- **Preset 6: Credential Stuffing Burst**: Multiple rapid login failures followed by immediate checkout from an untrusted IP $\rightarrow$ Action: `review` / `block`.

### 2. Flagged Transactions Feed Tab
A sortable table displaying simulated high-risk transactions with deep-dive panels rendering local SHAP bar charts for analyst triage.

### 3. Audit Trail & Live Telemetry Tab
Monitors system health in real time, displaying:
- Total transactions evaluated, allow/review/block distribution, and safety cap downgrade counters.
- Latency percentiles (P50, P95, P99).
- Live tabular view of `audit_trail.jsonl` with CSV export.

---

## Limitations

1. **Synthetic Data Distribution**: The system is trained and evaluated on synthetic data generated via parameterized behavioral distributions. While deliberately designed with behavioral overlap, synthetic distributions do not fully capture the complexity, noise, and adversarial adaptations of live payment networks.
2. **Simplified Fraud Typology**: The current implementation models account takeover and sudden velocity spikes. It does not address merchant-side fraud, card testing rings, authorized push payment (APP) scams, or slow-bleed bot attacks.
3. **Assumed Cost Matrix**: The operational costs (₹150 FP / ₹8,000 FN) are industry-standard illustrative placeholders. Real-world implementations require merchant-specific loss modeling based on operating margins and interchange fees.
4. **Single-Node In-Memory State**: The `SafetyCapManager` and `TelemetryTracker` utilize in-memory threading locks. In a distributed multi-pod Kubernetes deployment, state must be coordinated via a distributed Redis sliding-window log or Token Bucket algorithm.
5. **Uncalibrated Model Scores**: Raw XGBoost scores provide reliable risk rankings but are not calibrated Bayesian probabilities. Probability thresholds should be recalibrated via isotonic regression or Platt scaling when absolute loss estimation is required.
6. **Batch Profile Updates**: `UserProfiler` currently aggregates user baselines in offline batches. Live production architectures require real-time feature stores to update user spend means and variances after every transaction.

---

## Future Work

1. **Distributed Real-Time Feature Store**: Integrate with Redis or Feast to compute sliding-window user baselines ($\mu_u, \sigma_u$, rolling 24-hour spend) with sub-millisecond read/write latency.
2. **Online Learning & Analyst Feedback Loop**: Connect risk analyst dispute resolutions (confirmed fraud vs. false alarm) into an automated feedback pipeline to continuously recalibrate decision thresholds and retrain models incrementally (e.g., using River or rolling XGBoost updates).
3. **Production Probability Calibration**: Embed runtime Platt scaling or Isotonic Calibration directly into the `RiskPipeline` inference path to output calibrated loss expectations alongside rank scores.
4. **Distributed Sliding-Window Safety Cap**: Implement a distributed Redis sliding-window log using atomic Lua scripts to enforce cross-region safety caps across horizontally scaled API workers.
