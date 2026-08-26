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
5. **Cost-weighted threshold selection** — thresholds are swept against an explicit
   cost function (₹150 assumed cost per false positive: blocked legitimate customer,
   support load, churn risk; ₹8,000 assumed cost per false negative: average fraud
   loss). The threshold minimizing total expected cost is selected and reported
   explicitly, not just the default 0.5 cutoff.
6. **Explainability (SHAP)** — every flagged transaction includes its top contributing
   features and their direction of effect, so a risk analyst sees *why* a transaction
   was flagged, not just a bare score.

## Results (on held-out test set)
| Metric | Value |
|---|---|
| Precision @ selected threshold | 0.967 |
| Recall @ selected threshold | 0.967 |
| PR-AUC | 0.994 |
| Estimated total cost (test batch) | ₹18,400 |

## Repository structure
- `fraud_classifier.py` — synthetic data generation, feature engineering, model
  training, evaluation, and cost-weighted threshold selection.
- `shap_explainer.py` — per-alert and global SHAP explanations built on the trained
  model.
- `shap_summary.png` — global feature-importance visualization.

## How to run
```
pip install numpy pandas scikit-learn xgboost shap matplotlib
python fraud_classifier.py
python shap_explainer.py
```

## Honest limitations
- All data is synthetic. Real transaction data would likely show more varied
  fraud patterns per alert than this dataset does — here, `geo_dist_from_usual_km`
  dominates most top-risk alerts, which partly reflects how the synthetic fraud
  cases were generated rather than a universal truth about ATO fraud.
- The assumed costs (₹150 / ₹8,000) are illustrative placeholders, not derived from
  real merchant data. The methodology (explicit cost-weighted threshold selection)
  is the contribution here, not these specific numbers.
- This is strictly a **defense-only** detection system. It identifies and flags
  suspicious activity; it does not and cannot be used to evade fraud detection.

## What I'd improve with more time
- A rules-based decision/action layer (hold, step-up authentication, escalate) with
  an audit trail logging every decision.
- Fused behavioral signals (device interaction patterns) alongside transaction
  fields, for a stronger account-takeover signal than transaction data alone provides.
