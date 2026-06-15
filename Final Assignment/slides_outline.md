## Slide 1 — Title

**Title:** Delivering AI Technology to Company A: Proactive Churn Prevention Strategy
**Subtitle:** Maximizing Business Revenue through Data Analytics
**Presenter:** [Your Omnicampus Account Name]  
**Date:** June 2026

**Visual:** Professional title slide, AI and Telecom abstract background.

---

## Slide 2 — Introduction & Assumptions

**Bullets:**
- Company A is facing severe competition; acquiring new users is expensive compared to retaining existing ones.
- Our proposal enables **Data-Driven Retention** using predictive machine learning capabilities.
- We analyzed 100,000 historical subscriber records to identify patterns of "Stay" vs. "Churn."
- The goal is to provide a systemic approach to customer retention, saving approximately ~$700K+ in annual revenue through precise targeting.

**Visual:** Bulleted text with a brief workflow graphic.

---

## Slide 3 — Table of Contents

1. **Current State of Customer Retention**
2. **Introduction of Data & Visualizations**
3. **Overview of the New AI Business Model**
4. **Prediction via Machine Learning Methods**
5. **Our Business Proposals & Simulation**

---

## Slide 4 — Current State of Customer Retention

**One-line message:** In a saturated market, proactive retention reduces opportunity loss.

**Bullets:**
- Saturated Market: telecom growth is driven by ARPU lift and retention.
- Huge Opportunity Cost: Company A currently loses ~$2.9M/month from unmitigated churn.
- Limitation of Rule-based Systems: Discovering complex churn signals (like usage drop + device age) manually is difficult.

**Visual:** Bar chart showing Customer Acquisition Cost (CAC) vs. Cost of Retention.

---

## Slide 5 — Introduction of Data

**One-line message:** Over 100+ types of customer information accumulated.

**Data Contents:**
- **Demographics:** Age, occupation, location.
- **Contract & Equipment:** Tenure (months), device age (eqpdays), handset price.
- **Usage Patterns:** Monthly minutes used (mou), changes in usage (change_mou).
- **Billing:** Average revenue, overage charges, credit metrics.

**Target Variable Overview:**
- **49.6% Churn vs. 50.4% Non-Churn.** Balanced dataset enabling reliable AI learning.

**Visual:** Data sources (Client + Record) combined into a unified database icon.

---

## Slide 6 — Data Visualization: Key Churn Drivers

**One-line message:** Significant behavioral shifts define the difference between Stayers and Churners.

**Key Visualizations (From Notebook):**
- **The Contract Cliff:** Churn jumps from 33.8% (months 6–10) to 63.7% (months 11–12).
- **Equipment Lifespan:** Equipment > 270 days sees a 54.9% churn rate.
- **Usage Drop:** Churners display average usage drops (`change_mou` ≈ -23) compared to stayers.

**Visual:** Arrange the notebook charts (Tenure vs Churn, Device Age, Usage Boxplot) into a clear dashboard layout.

---

## Slide 7 — Overview of the New AI Business

**One-line message:** Transition from reactive retention to an AI-driven Profit Maximization System (PMS).

**Two Core Business Pillars:**
1. **At-Risk Customer Prediction:** Automatically extract customers highly likely to cancel their service in the next 31-60 days.
2. **Trend Analysis & Strategic Offers:** Automatically match customers with the optimal retention campaign (upgrades, rate plan adjustments).

**Visual:** Conceptual diagram bridging "Data" -> "AI Brain" -> "Revenue Maximization".

---

## Slide 8 — Prediction Using Machine Learning Models

**One-line message:** Transforming raw customer data into an actionable "Churn Risk Score."

**System Overview:**
- **Input:** Usage patterns, equipment info, demographic data.
- **Machine Learning Method:** **LightGBM Classifier** + **XGBoost Classifier**
- **Output:** Predicted Probability of Churn (Risk Score).

**Q: How do we measure system quality?**
- **A:** We use historical data to test accuracy and "ROC-AUC" (which measures discrimination capability).
- **Results:**
  - **LightGBM Classifier Accuracy:** **63.8%**
  - **LightGBM Classifier ROC-AUC:** **0.693**

**Visual:** Q&A formatting for executive understanding (similar to Sample 2), accompanied by the Confusion Matrix.

---

## Slide 9 — Important Features for Prediction

**One-line message:** AI predictions place distinct weight on dynamic usage behavior over static demographics.

**Top predictive features:**
1. `change_mou` (Usage trend)
2. `mou_Mean` (Average usage)
3. `totmrc_Mean` (Monthly recurring charge)
4. `months` (Tenure)

**Takeaway:** By utilizing both AI anomaly detection and human domain knowledge, highly accurate retention decisions become possible.

**Visual:** Feature Importance horizontal bar chart from the notebook.

---

## Slide 10 — Our Business Proposals

**One-line message:** Leveraging AI to maximize lifetime value through targeted long-term support.

1. **Pre-Expiry Device Subsidies:** 
   - Deploy $100 handset vouchers exactly at month 9–10 to preempt the 63.7% cliff.
2. **Anti-Bill-Shock Alerts:**
   - Migrate users paying >$20 in overages to unlimited plans before they churn.
3. **Usage Decline Intervention:**
   - AI-triggered account health checks for high-value users showing rapid usage drops.

**Visual:** 3-column layout outlining the proposals.

---

## Slide 11 — Revenue Maximization via Optimal Default Thresholds

**One-line message:** Adjusting the "Churn Threshold" allows us to prioritize high-margin interventions.

**Validity of the Churn Risk Score:**
- If we define Churn Score > 0.65 as our trigger, we avoid *"cannibalization"* (spamming safe customers with discounts).
- We can filter out false positives and focus marketing budget only on customers who truly intend to leave but have high ARPU.

**Visual:** Diagram explaining the "Probability Threshold cutoff" mapping to "Intervention vs. No Intervention" (Inspired by Sample 1/2 distribution graphs).

---

## Slide 12 — Revenue Prediction Simulation (PMS)

**One-line message:** AI investment translates directly into gross marginal revenue.

**Conservative Math (Base Run):**
- **Target:** 10,000 high-risk subscribers identified by ML.
- **Campaign Cost:** $1,000,000 ($100 per targeted user).
- **Precision:** Model correctly captures 65% true churners; Offer acceptance is 20%.
- **Customers Saved:** ~1,300 active users.

**Impact on Revenue:**
- Annual Revenue Protected: ~**$916,000** 
- *Targeting only the top 20% high-ARPU subscribers pushes the gross protected value over **$1.5M**.*

**Visual:** Comparison of Net Profit "Before AI Implementation" vs. "After AI Implementation" (Bar chart or Table).

---

## Slide 13 — Future Vision & Long-Term Support

**One-line message:** AI is a continuous growth engine.

**Model Updates & Lifecycle:**
- Retrain the LightGBM model quarterly on fresh metrics to prevent concept drift.
   - Refine thresholds automatically with PMS (Profit Maximization System).
- Expand to geographic A/B testing (e.g., control groups to measure true uplift).

**Visual:** Circular diagram of Data Collection -> Model Retraining -> Campaign Execution.

---

## Slide 14 — References & Academic Integrity

- Company A dataset: `Client.csv`, `Record.csv` (GCI World Final Assignment).
- Telecom churn / CAC industry reports & HomeCredit Sample structures.
- Scikit-learn, XGBoost, LightGBM documentation.
- Generative AI Chat URL: [Insert Chat URL]

---

## Slide 15 — Q&A

**Title:** Thank You / Q&A
**Visual:** Clean conclusion slide, presenter contact details.
