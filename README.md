# Fraud-spike / account-takeover detector
**Razorpay AI Buildathon — AI Risk Manager track**

## Problem
Merchants lose money to fraud, returns, and chargebacks. This project targets one
specific loss class: **fraud-spike / account-takeover (ATO) detection** — flagging
transactions that show behavioral deviation from a user's normal pattern (sudden
device change, geographic jump, velocity spike, login burst), before the loss occurs.

Account takeover was chosen because it is the fastest-growing fraud category in the
fintech sector specifically, making it the most relevant loss class for a payments
company to defend against.

## Approach
1. **Synthetic data generation** — 40,000 transactions across 2,000 simulated users,
   with a realistic 1.5% fraud rate. Fraud is generated as a *deviation from each
   user's own baseline* (not random noise), with deliberate overlap between fraud and
   legitimate behavior so the classification problem isn't artificially easy.
2. **Feature engineering** — deviation-based features (amount z-score against the
   user's own history) alongside raw signals (geo-distance, transaction velocity,
   device change, login burst count, IP risk score).
3. **Classifier** — XGBoost with class weighting (`scale_pos_weight`) to handle the
   1.5% fraud rate without naive oversampling.
4. **Evaluation** — precision, recall, and PR-AUC only. Accuracy is not reported, since
   it is meaningless on this class distribution (a "never fraud" model would score 98%+).
   Validated with 5-fold cross-validation, not a single lucky split (see Results).
5. **Cost-weighted threshold selection** — thresholds are swept against an explicit
   cost function (₹150 assumed cost per false positive: blocked legitimate customer,
   support load, churn risk; ₹8,000 assumed cost per false negative: average fraud
   loss). The threshold minimizing total expected cost is selected and reported
   explicitly, not just the default 0.5 cutoff.
6. **Explainability (SHAP)** — every flagged transaction includes its top contributing
   features and their direction of effect, so a risk analyst sees *why* a transaction
   was flagged, not just a bare score.
7. **Decision layer with bounded actions** — risk scores are converted into one of
   three fixed actions (`allow` / `review` / `block`) purely by threshold, never by
   free-form model judgment. A hard safety cap (max 50 auto-blocks/hour) forces
   graceful degradation to `review` once hit, so the system cannot silently block an
   unbounded number of transactions. Every decision — including whether it was capped —
   is written to an append-only audit trail (`audit_trail.jsonl`).
8. **Mobile behavioral signal fusion** — five simulated device/interaction signals
   (touch pressure variance, typing rhythm deviation, device motion stability,
   orientation changes, session duration) are fused with transaction features, since
   account takeover shows up in *how* someone uses their phone, not just what they
   transact. Tested as a controlled comparison against the transaction-only baseline,
   not just assumed to help (see Results).
9. **Data drift simulation** — the trained model is evaluated against a simulated
   "one month later" batch where user spending habits and fraud tactics have shifted,
   to test whether performance holds up over time rather than only on same-day data.
10. **Live dashboard** (`app.py`, Streamlit) — summary metrics, a sortable flagged-
    transaction feed, a click-through alert detail panel showing SHAP factors, and a
    live audit trail viewer with a download button.
11. **Scoring API** (`api.py`, FastAPI) — a real callable service: POST a transaction's
    features to `/score`, get back a risk score, the bounded action, and the top SHAP
    factors. Interactive docs at `/docs`. This is what makes the system look like
    something that could sit behind a real payment flow, not just a notebook script.

## Results
**Core classifier** (held-out test set, 10,000 transactions):
| Metric | Value |
|---|---|
| Precision @ selected threshold | 0.967 |
| Recall @ selected threshold | 0.967 |
| PR-AUC | 0.994 |
| Estimated total cost (classifier-only batch) | Rs 18,400 |
| Precision on flagged (review + block) transactions | 90.2% (148/164) |
| Fraudulent transactions missed (`allow`) | 2 |
| Blocks downgraded to review by the safety cap | 97 |

**5-fold cross-validation** (confirms the above isn't a lucky split):
| Metric | Mean | Std dev |
|---|---|---|
| PR-AUC | 0.9975 | ±0.0016 |
| Precision | 0.9660 | ±0.0162 |
| Recall | 0.9867 | ±0.0075 |

**Mobile signal fusion** (transaction-only vs. transaction + behavioral signals):
| Model | PR-AUC | Precision | Recall |
|---|---|---|---|
| Transaction-only (baseline) | 0.9939 | 0.9667 | 0.9667 |
| Transaction + mobile signals | 0.9968 | 0.9797 | 0.9667 |

A modest, honest improvement (+1.3 precision points, recall unchanged) — not an
inflated one. Individually, mobile signals rank low in feature importance versus
`geo_dist_from_usual_km` and `amount_zscore`; they act as a complementary boost on
harder edge cases, not a replacement for the core transaction signal.

**Drift simulation** (same model, evaluated one month later with no retraining):
| Scenario | PR-AUC | Precision | Recall |
|---|---|---|---|
| Today (no drift) | 0.9939 | 0.9667 | 0.9667 |
| One month later (drifted) | 0.9876 | 0.9625 | 0.9400 |

Recall drops 2.7 points under simulated drift — a real, honest degradation.
**Recommendation:** retrain on a rolling 2-4 week window, and monitor precision/recall
on a held-out recent slice weekly, alerting if either drops more than 2 points from
its trained baseline.

**Analyst workload projection** (assuming 2,000,000 transactions/day): the observed
1.64% flagged rate scales to ~32,800 alerts/day — roughly 205 full-time analysts at
3 minutes/review. This is the concrete case for the bounded auto-decision layer: pure
human review does not scale at this volume.

The safety-cap figure above is a genuine result of an actual run, not a hypothetical:
the model's raw confidence would have auto-blocked more than 50 transactions in that
batch, and the hard cap forced 97 of those into human review instead — demonstrating
the bound actually engages under load rather than existing only on paper.

## Repository structure
- `fraud_classifier.py` — synthetic data generation, feature engineering, model
  training, evaluation, and cost-weighted threshold selection.
- `shap_explainer.py` — per-alert and global SHAP explanations built on the trained
  model.
- `decision_layer.py` — converts risk scores into bounded actions (allow/review/block),
  enforces the hourly safety cap, and writes the full audit trail.
- `mobile_signal_fusion.py` — adds simulated mobile behavioral signals and compares
  against the transaction-only baseline.
- `additional_evaluation.py` — precision-recall curve plot, 5-fold cross-validation,
  and the analyst workload projection.
- `drift_simulation.py` — "one month later" drift test and retraining recommendation.
- `app.py` — Streamlit dashboard (metrics, transaction feed, alert detail, audit trail).
- `api.py` — FastAPI scoring service (`/score` endpoint, interactive docs at `/docs`).
- `latency_benchmark.py` — single-transaction and batch scoring speed benchmark.
- `test_pipeline.py` — pytest suite covering decision boundaries, classifier output
  validity, the safety cap, and API input validation.
- `shap_summary.png`, `pr_curve.png`, `fraud_detector_architecture.png` — visuals.
- `audit_trail.jsonl` — sample audit log from an actual run.

## How to run
```
pip install numpy pandas scikit-learn xgboost shap matplotlib streamlit fastapi uvicorn

python fraud_classifier.py          # core classifier
python shap_explainer.py            # explainability
python decision_layer.py            # bounded decisions + audit trail
python mobile_signal_fusion.py      # mobile signal comparison
python additional_evaluation.py     # cross-validation, PR curve, workload
python drift_simulation.py          # drift test

streamlit run app.py                # live dashboard
uvicorn api:app --reload            # scoring API -- visit /docs to test

pip install pytest
pytest test_pipeline.py -v          # 17 tests
python latency_benchmark.py         # scoring speed benchmark
```

## Honest limitations
- All data is synthetic, including the mobile behavioral signals (not pulled from a
  real device SDK). A production version would need actual telemetry via a mobile
  SDK (accelerometer, touch API) — this is explicitly future work, not a hidden gap.
- `geo_dist_from_usual_km` dominates most top-risk alerts, which partly reflects how
  the synthetic fraud cases were generated rather than a universal truth about ATO
  fraud in real data.
- The assumed costs (Rs 150 / Rs 8,000) and workload assumptions (2M txns/day, 3 min/review)
  are illustrative placeholders, not derived from real merchant data. The methodology
  (explicit cost-weighted thresholds, workload projection) is the contribution here,
  not these specific numbers.
- The drift simulation is a controlled synthetic test, not a claim about real-world
  drift magnitude — real fraud tactics may adapt faster or slower than simulated here.
- This is strictly a **defense-only** detection system. It identifies and flags
  suspicious activity; it does not and cannot be used to evade fraud detection.

## What I'd improve with more time
- Real device telemetry via a mobile SDK, replacing the simulated behavioral signals.
- Feedback loop: when a `review` decision is manually confirmed or overturned by an
  analyst, feed that outcome back to recalibrate the threshold over time.
- Automate the drift-monitoring recommendation into a scheduled job that actually
  retrains and alerts, rather than a one-off simulation.

## Latency (can this run in real time, not just as a batch job?)
Single-transaction scoring was benchmarked over 1,000 runs:

| Metric | Value |
|---|---|
| Mean | 1.33 ms |
| P95 | 1.46 ms |
| P99 | 1.62 ms |

Batch scoring throughput: ~322,000 transactions/second.

A P99 of 1.6ms is well within the typical <100-200ms budget for in-line scoring
during a payment authorization flow — this model could run **before** a transaction
completes, not just flag it after the fact. Note: this excludes SHAP explanation
time, which is only computed for flagged (review/block) transactions, not every
transaction — see `decision_layer.py`.

## Testing
17 tests in `test_pipeline.py`, covering:
- Decision boundaries (every threshold edge case for allow/review/block)
- Classifier output validity (probabilities in [0,1], no NaNs, correct shape)
- The safety cap actually holding under a high-fraud-volume batch
- API input validation (rejects out-of-range risk scores and invalid hours)

Run with: `pytest test_pipeline.py -v`

## What broke, and what I did about it
- **Environment setup**: lost time to Python not being on PATH, a PowerShell
  execution-policy block on venv activation, and VS Code's Run button silently
  using a different Python installation than the terminal. Resolved by checking
  `sys.executable` directly to find the actual interpreter in use, and calling the
  venv's `python.exe` by full path rather than relying on `activate` succeeding
  silently.
- **Git merge conflicts**: GitHub auto-created a README when the repo was made via
  the website, which collided with a local commit. Resolved with
  `git pull --allow-unrelated-histories` and a manual merge rather than force-pushing
  over unknown remote content blindly.
- **A suspicious 1.000 PR-AUC**: the first version of the synthetic data generator
  made fraud too easily separable from legitimate transactions (perfectly separable
  data isn't realistic and would have been an inflated, non-credible result). Caught
  this before trusting the number, and rewrote the generator so fraud and legitimate
  behavior genuinely overlap — the real result (0.994 PR-AUC, 0.967 precision/recall)
  is lower, but is the one I can actually defend. The same check was applied again
  when adding mobile signals, for the same reason.

## AI usage
AI assistance (Claude) was used for code scaffolding, debugging environment issues,
and drafting this README. The problem framing (choosing account-takeover as the
loss class), the cost assumptions (₹150/₹8,000), the threshold values (0.05/0.80),
the safety-cap design, the evaluation methodology (cross-validation, drift testing,
the workload projection), and catching and fixing the inflated PR-AUC were my
judgment calls, not generated defaults — and I can walk through and defend any part
of this code.
