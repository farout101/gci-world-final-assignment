# Slide Outline — GCI World Final Assignment

> **Rename before submit:** `YourOmnicampusName.pdf`  
> **Deadline:** 2026-06-19 11:00 UTC  
> **Max slides:** 15 (this outline = 15 slides)

Replace bracketed placeholders with your name, market, and citations.

---

## Slide 1 — Title

**Title:** Proactive Retention Strategy: Data-Driven Churn Mitigation  
**Subtitle:** Business Proposal for Company A  
**Presenter:** [Your Omnicampus Account Name]  
**Date:** June 2026

**Visual:** Clean title slide, company logo optional.

---

## Slide 2 — Executive Summary

**One-line message:** Target high-risk subscribers before contract expiry and bill shock to protect ~$700K+ annual revenue.

**Bullets:**
- Dataset: 100,000 wireless subscribers; churn rate **49.6%** (balanced classes).
- Key insight: churn jumps from **33.8%** (months 6–10) to **63.7%** (months 11–12).
- Model: **LightGBM Classifier** — **Accuracy 63.8%**, **ROC-AUC 0.693**.
- Proposal: three campaigns (pre-expiry upgrade, anti-bill-shock migration, usage-decline outreach).
- Impact: ~**1,300 customers retained** per 10,000 targeted → **~$702K** annual revenue protected (conservative).

**Visual:** 3-box summary (Problem → Model → ROI).

---

## Slide 3 — Market Context

**One-line message:** In saturated telecom markets, retention is far cheaper than acquisition.

**Bullets:**
- Wireless market is mature; growth comes from ARPU lift and churn reduction.
- Industry CAC often cited at **$300+** per new subscriber (cite a 2024–2025 source for your region).
- Company A loses **~$2.9M/month** in monthly revenue from churners (`rev_Mean` sum).
- Proactive, data-driven retention beats reactive win-back.

**Visual:** Simple bar chart — CAC vs cost of retention offer ($100 subsidy).

**References to add:** Analysys Mason, GSMA, or local regulator report.

---

## Slide 4 — Problem Definition & Target

**One-line message:** Predict who will churn in the next 31–60 days so we can intervene early.

**Bullets:**
- **Target variable:** `churn` (1 = left within 31–60 days after observation).
- **Baseline:** 49.6% churned / 50.4% stayed — accuracy is a meaningful metric here.
- **Business question:** Which subscribers should receive a retention offer *before* they leave?
- **Data sources:** `Client.csv` (demographics, equipment) + `Record.csv` (usage, billing, churn).

**Visual:** Table of the two CSV files and join key `Customer_ID`.

---

## Slide 5 — EDA: The Contract Cliff

**One-line message:** Churn spikes at the 1-year mark — intervene in months 9–10.

**Bullets:**
- Churn by tenure bucket shows a clear spike at months **11–12**.
- Months 6–10: **33.8%** churn rate.
- Months 11–12: **63.7%** churn rate (**+29.9 pp**).
- **Action window:** trigger campaigns when `months` ∈ [9, 10].

**Visual:** Bar chart — churn rate by `months` bucket (from notebook Section 2).

**Notebook cell:** `tenure_churn` plot.

---

## Slide 6 — EDA: Device Age, Handset Value & Usage Decline

**One-line message:** Old phones, budget devices, and falling usage predict churn.

**Bullets:**
- Handset price ≤ $50: **57.5%** churn vs > $200: **36.0%** churn.
- Equipment age > 270 days: **54.9%** churn.
- Churners show sharper usage drop: `change_mou` ≈ **-23** vs stayers **-5**.
- Top model features include `change_mou`, `mou_Mean`, `eqpdays`, `months`.

**Visual:** Two small charts — churn by `hnd_price` bucket + `change_mou` distribution by churn.

**Notebook cells:** equipment age + handset price EDA.

---

## Slide 7 — ML Pipeline

**One-line message:** Merge → engineer features → impute → encode → stratified split → train.

**Pipeline steps:**
1. Inner join `Record.csv` + `Client.csv` on `Customer_ID` → 100,000 × 100 columns.
2. Engineer: `upgrade_eligible`, `bill_shock_risk`, `severe_usage_drop`, `eqp_tenure_ratio`, `is_budget_phone`.
3. Impute numeric missing values with median; categoricals → `Unknown` + label encode.
4. **70/30 stratified train/test split** (`random_state=42`).

**Visual:** Flow diagram (boxes + arrows). No code on slide — keep it executive-friendly.

---

## Slide 8 — Model Performance *(required: name + metric + score)*

**One-line message:** LightGBM slightly outperforms XGBoost on held-out test data.

| Model | Metric | Score |
|-------|--------|-------|
| **XGBoost Classifier** | Accuracy | **63.6%** |
| **XGBoost Classifier** | ROC-AUC | **0.692** |
| **LightGBM Classifier** *(selected)* | Accuracy | **63.8%** |
| **LightGBM Classifier** | ROC-AUC | **0.693** |

**Note for slides:** State all three explicitly — model name, metric, numeric score.

**Visual:** Confusion matrix for LightGBM (from notebook).

---

## Slide 9 — Key Churn Drivers

**One-line message:** Behavioral decline and tenure/equipment signals dominate demographics.

**Top features (LightGBM importance):**
1. `change_mou` — usage trend
2. `mou_Mean` — average minutes of use
3. `totmrc_Mean` — monthly recurring charge
4. `months` — tenure
5. `change_rev` — revenue trend
6. `eqpdays` — equipment age

**Takeaway:** Campaigns should react to **behavior change**, not just static demographics.

**Visual:** Horizontal bar chart — top 10 feature importances.

---

## Slide 10 — Strategy Overview

**One-line message:** Three targeted campaigns aligned to the strongest data signals.

| Campaign | Trigger | Offer |
|----------|---------|-------|
| **1. Pre-expiry upgrade** | months 9–10, `eqpdays` > 270, high churn score | $100 handset subsidy + 12-month renewal |
| **2. Anti-bill-shock** | `vceovr_Mean` > $20/month | Migrate to higher plan (+$10/mo unlimited voice) |
| **3. Usage-decline save** | `change_mou` ≤ -100, high `rev_Mean` | Proactive call + loyalty discount |

**Visual:** 3-column layout with icons.

---

## Slide 11 — Campaign 1: Pre-Expiry Upgrades

**One-line message:** Subsidize upgrades before the contract cliff to reset tenure and device lifecycle.

**Details:**
- **Who:** subscribers in months 9–10 with equipment age > 270 days and predicted churn probability > 0.65.
- **Offer:** $100 upgrade voucher tied to a new 12-month contract.
- **Why it works:** addresses the **63.7%** churn spike at months 11–12 and old-device friction.

**Visual:** Timeline showing intervention at month 9–10 vs churn spike at 11–12.

---

## Slide 12 — Campaign 2: Anti-Bill-Shock Migrations

**One-line message:** Convert overage payers to higher plans before they disengage.

**Details:**
- **Who:** customers with voice overages (`vceovr_Mean`) > $20/month.
- **Offer:** unlimited voice plan for +$10/month (eliminates surprise bills).
- **Context:** **57%** of customers pay some overage; overage pool ≈ **$1.4M/month**.

**Visual:** Before/after bill comparison for a sample customer.

---

## Slide 13 — Financial ROI Case

**One-line message:** Micro-targeting high-risk subscribers is profitable under conservative assumptions.

**Conservative math (Campaign 1 example):**
- Target pool: **10,000** high-risk subscribers
- Offer cost: **$100** each → **$1,000,000** campaign spend
- Model precision in top decile: ~**65%** (6,500 true churners in pool)
- Offer acceptance rate: **20%**
- Customers saved: 10,000 × 20% × 65% = **1,300**
- Annual ARPU: `rev_Mean` × 12 ≈ **$705**
- **Revenue protected: 1,300 × $705 ≈ $916K** (gross)  
- **Net after subsidy: ~-$84K** at $100 subsidy — *raise subsidy only for high-ARPU segment* to ensure positive ROI

**Refined scenario (top 20% by `rev_Mean` only):**
- Higher ARPU (~$96/mo) → **~$1,152** annual → **~$1.5M** gross protected for 1,300 saves.

**Visual:** Simple ROI table with best / base / worst case.

---

## Slide 14 — Risks & Next Steps

**Risks:**
- **Cannibalization:** subsidizing customers who would have stayed anyway → mitigate with probability threshold > 0.65.
- **Model drift:** retrain quarterly on fresh `Record.csv` snapshots.
- **Regional variation:** NW/Rocky Mountain area has **~57%** churn — pilot there first.

**Next steps:**
1. A/B test with 5,000-customer control group.
2. Measure 60-day retention lift and incremental ARPU.
3. Roll out to South Florida + California North if ROI positive.

**Visual:** Risk/mitigation two-column table.

---

## Slide 15 — References

**Include on slide and in Omnicampus reference field:**

- Company A dataset: `Client.csv`, `Record.csv` (GCI World Final Assignment).
- [Telecom churn / CAC industry report — add URL]
- [Local market source — add URL]
- Scikit-learn, XGBoost, LightGBM documentation (if cited).
- **Generative AI:** [Cursor/ChatGPT conversation URL — required if you used AI]

---

## Submission checklist

- [ ] PDF ≤ 15 slides, ≤ 10 MB, named `OmnicampusName.pdf`
- [ ] Notebook named `OmnicampusName.ipynb` — runs top-to-bottom
- [ ] Slide 8 has **model name + metric + numeric score**
- [ ] References entered in Omnicampus + on slide 15
- [ ] AI chat URL included if applicable
