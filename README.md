# 🛒 Retail Customer Segmentation using RFM Analysis | Python

**Author:** Hoang Duc Kien
**Tools Used:** Python

---

## 📑 Table of Contents

- 📌 [Background & Overview](#-background--overview)
- 📂 [Dataset Description & Data Structure](#-dataset-description--data-structure)
- 🧹 [Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
- 🔍 [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- 🧮 [Apply RFM Model](#-apply-rfm-model)
- 📊 [Visualization & Analysis](#-visualization--analysis)
- 💡 [Insight & Recommendation](#-insight--recommendation)

---

## 📌 Background & Overview

### 🌍 Business Context

This project focuses on customer segmentation for a retail transaction dataset using the RFM framework.

In a transactional retail business, customer value is not determined only by how much a customer spends once. It is also shaped by how recently the customer purchased and how often the customer returns. Because of that, RFM is a practical model for translating raw order data into segment-based business actions.

The notebook builds a complete workflow that starts from raw transaction records, cleans data quality issues, creates customer-level metrics, scores customers, maps them into business segments, and then visualizes customer behavior from multiple perspectives.

### 🎯 Project Objective

The project aims to:

- clean and validate retail transaction data before analysis
- calculate customer-level Recency, Frequency, and Monetary values
- assign quintile-based RFM scores
- map customers into business-friendly segments
- support marketing and sales decisions with segment-level visual analysis

### 🧠 RFM Framework

RFM is a data-mining technique used to quantitatively rank and group customer behavior.

**Core assumption:** past customer behavior is one of the strongest predictors of future behavior.

RFM provides a practical snapshot of current customer engagement and value, so teams can move from broad campaigns to segment-based actions.

#### 🧮 How RFM Metrics Are Calculated

- **Recency (R)**
    - Definition: time (days) since the last completed transaction.
    - Business question: "How recently did they buy?"

- **Frequency (F)**
    - Definition: total number of completed transactions (repeat purchases).
    - Business question: "How often do they buy?"

- **Monetary (M)**
    - Definition: customer spending value.
    - Common options: total spend, average order value (AOV), or period-based spend.
    - In this notebook: Monetary is measured as total revenue per customer.
    - Business question: "How much value do they generate?"


#### 🚀 Why Use RFM

RFM is widely used because it helps businesses:

- identify and prioritize high-value customers
- optimize marketing spend and improve ROI
- proactively reduce churn with early risk signals
- improve retention through targeted re-engagement flows
- personalize communication, promotions, and offer timing
- align sales and CRM effort toward the most valuable segments

#### ✅ Business Effectiveness of RFM

When implemented correctly, RFM can create measurable impact:

- better campaign precision than mass marketing
- lower budget waste on low-value audiences
- stronger retention in high-potential segments
- higher repeat purchase rate and customer lifetime value
- clearer decision-making through segment-specific KPIs

In short, not all customers provide the same value, and RFM gives a structured way to act on that difference.

### ❓ Key Business Questions

- Which customer groups are the most valuable?
- Which segments appear to be losing momentum?
- Which metric should receive the highest operational priority?
- How should segment-level findings be translated into marketing actions?

---

## 📂 Dataset Description & Data Structure

### 📌 Data Source

The notebook loads a retail transaction dataset from an Excel file and works with transaction-level order data.

### 📋 Main Columns Used

| Column Name | Description |
|-------------|-------------|
| InvoiceNo | Invoice identifier. Values starting with `C` indicate cancellations. |
| StockCode | Product identifier. |
| Description | Product description. |
| Quantity | Number of units purchased. |
| InvoiceDate | Transaction date and time. |
| UnitPrice | Unit selling price. |
| CustomerID | Customer identifier used for RFM grouping. |
| Country | Customer country. |

### 🗺️ Segment Mapping

The segment rules are defined directly in the notebook through a hard-coded `segment_map` dictionary.

<details>
<summary>RFM Score to Segment Mapping</summary>

<br>

| Segment | RFM Scores |
|---------|------------|
| Champions | 555, 554, 544, 545, 454, 455, 445 |
| Loyal | 543, 444, 435, 355, 354, 345, 344, 335 |
| Potential Loyalist | 553, 551, 552, 541, 542, 533, 532, 531, 452, 451, 442, 441, 431, 453, 433, 432, 423, 353, 352, 351, 342, 341, 333, 323 |
| New Customers | 512, 511, 422, 421, 412, 411, 311 |
| Promising | 525, 524, 523, 522, 521, 515, 514, 513, 425, 424, 413, 414, 415, 315, 314, 313 |
| Need Attention | 535, 534, 443, 434, 343, 334, 325, 324 |
| About To Sleep | 331, 321, 312, 221, 213, 231, 241, 251 |
| At Risk | 255, 254, 245, 244, 253, 252, 243, 242, 235, 234, 225, 224, 153, 152, 145, 143, 142, 135, 134, 133, 125, 124 |
| Cannot Lose Them | 155, 154, 144, 214, 215, 115, 114, 113 |
| Hibernating customers | 332, 322, 233, 232, 223, 222, 132, 123, 122, 212, 211 |
| Lost customers | 111, 112, 121, 131, 141, 151 |

</details>

---

## 🧹 Data Cleaning & EDA

### 1. 🛠️ Standardize Data Types

The notebook converts all identifier-style columns to string and parses the invoice date as datetime.

```python
column_list = ["InvoiceNo", "StockCode", "Description", "CustomerID", "Country"]

for col in column_list:
    ecommerce_retail[col] = ecommerce_retail[col].astype(str)

ecommerce_retail["InvoiceDate"] = pd.to_datetime(
    ecommerce_retail["InvoiceDate"],
    errors="coerce"
)
```

This ensures that identifiers are treated as labels rather than numeric values and that the date column is ready for recency calculation.

### 3. 🚫 Remove Invalid Transactions


```python
rows_before_rules = len(ecommerce_retail)

mask_cancel = ecommerce_retail["InvoiceNo"].str.startswith("C", na=False)
mask_negative_qty = ecommerce_retail["Quantity"] < 0
mask_negative_price = ecommerce_retail["UnitPrice"] < 0

desc_norm = ecommerce_retail["Description"].str.strip().str.lower()
mask_invalid_desc = desc_norm.isin(["nan", "", "none", "?", "??", "???"])

ecommerce_retail = ecommerce_retail.loc[~mask_cancel].copy()
ecommerce_retail = ecommerce_retail.loc[ecommerce_retail["Quantity"] > 0].copy()
ecommerce_retail = ecommerce_retail.loc[ecommerce_retail["UnitPrice"] > 0].copy()
ecommerce_retail = ecommerce_retail.loc[
    ~mask_invalid_desc.reindex(ecommerce_retail.index, fill_value=False)
].copy()
```

The logic behind these rules is:

- invoices beginning with `C` are cancellations
- negative quantity values are not valid sales transactions in this workflow
- negative unit prices are invalid for revenue analysis
- placeholder or broken descriptions are excluded as incorrect product records


### 4. 🧾 Investigate and Handle Missing CustomerID

Customer-level segmentation requires a valid customer identifier, so the notebook treats missing `CustomerID` as a critical issue.

First, missing values are visualized with `missingno`:

```python
msno.matrix(ecommerce_retail)
```
![Missingno Matrix](https://github.com/user-attachments/assets/b136880e-c60d-456e-a670-6401a16290ef)

Then rows without a valid customer identifier are removed:

```python
ecommerce_retail = ecommerce_retail.dropna(subset=["CustomerID"]).copy()
```

### 5. ♻️ Duplicate Handling

When duplicate records are detected, they are resolved under two rule-based scenarios to ensure data integrity and preserve revenue accuracy.

**Case 1: Identical Duplicates**

- Definition: same `InvoiceNo`, `StockCode`, `CustomerID`, `InvoiceDate`, and `Quantity`.
- Interpretation: likely generated by system-level duplication.
- Action: keep one record and remove duplicate rows.

**Case 2: Same Transaction with Different Quantities**

- Definition: same `InvoiceNo` and `StockCode`, but recorded in separate rows with different `Quantity` values.
- Interpretation: likely caused by transaction split at order-line level.
- Action: aggregate `Quantity` for each transaction-product combination to maintain accurate revenue calculation.

---

## 🧮 Apply RFM Model

### 1. 📦 Build Customer-Level RFM Metrics

After cleaning, the notebook creates a revenue column and aggregates transactions by customer.

```python
ecommerce_retail["Revenue"] = ecommerce_retail["Quantity"] * ecommerce_retail["UnitPrice"]

reference_date = pd.to_datetime("2011-12-31")

rfm = ecommerce_retail.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,
    "InvoiceNo": "nunique",
    "Revenue": "sum",
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
```

Metric definitions in the notebook are:

- **Recency:** number of days since the customer's most recent purchase, measured against `2011-12-31`
- **Frequency:** number of unique invoices per customer
- **Monetary:** total revenue contributed by the customer

### 2. 🎯 Assign Quintile-Based Scores

The notebook converts each RFM metric into a 1 to 5 score.

```python
rfm["R"] = pd.qcut(
    rfm["Recency"],
    q=5,
    labels=[5, 4, 3, 2, 1],
    duplicates="drop",
)

rfm["F"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    q=5,
    labels=[1, 2, 3, 4, 5],
    duplicates="drop",
)

rfm["M"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    q=5,
    labels=[1, 2, 3, 4, 5],
    duplicates="drop",
)

rfm["RFM_scores"] = (
    rfm["R"].astype(str) +
    rfm["F"].astype(str) +
    rfm["M"].astype(str)
)
```

Scoring logic:

- lower Recency is better, so more recent customers receive higher `R` scores
- higher Frequency is better, so more frequent customers receive higher `F` scores
- higher Monetary is better, so higher-spending customers receive higher `M` scores

The use of ranked values for `F` and `M` helps `qcut` work more reliably when many customers share the same raw values.

### 3. 🏷️ Map RFM Scores to Segments

The notebook uses a predefined `segment_map` and reverses it into a score-to-segment dictionary.

```python
rfm_to_segment = {
    score: seg
    for seg, scores in segment_map.items()
    for score in scores
}

rfm["Segment"] = rfm["RFM_scores"].map(rfm_to_segment).fillna("Others")
```

This step turns technical score combinations into business-friendly customer groups such as `Champions`, `Loyal`, `At Risk`, and `Lost customers`.

### 4. 🔗 Push Segments Back to Transaction Level

The segment label is also attached back to the transaction table so that order-level records can be analyzed by customer segment.

```python
ecommerce_retail["Segment"] = ecommerce_retail["CustomerID"].map(
    rfm.set_index("CustomerID")["Segment"]
)
```

This makes it possible to compare both customer-level and transaction-level views in later analysis.

---

## 📊 Visualization & Analysis

The notebook contains a segment visualization section followed by an extended RFM visualization pack. Before plotting, it prepares a helper dataframe named `rfm_viz`.

```python
rfm_viz = rfm.copy()

for metric in ["R", "F", "M"]:
    rfm_viz[metric] = rfm_viz[metric].astype(int)

rfm_viz["FM_Score"] = ((rfm_viz["F"] + rfm_viz["M"]) / 2).round().astype(int)
rfm_viz["Monetary_log1p"] = np.log1p(rfm_viz["Monetary"])
```

This section analyzes customer behavior from multiple angles.

### 1. 👥 Customer Count by Segment

This bar chart shows how the customer base is distributed across RFM segments.
![Customer Count by Segment](https://github.com/user-attachments/assets/5c075c18-0f01-454d-bdde-4695af2aac02)
**Key insight:**
- `Champions` is the largest segment at about 830 customers (~19%), followed by `Hibernating customers` at about 700 (~16%) and `Lost customers` at about 480 (~11%).
- Inactive groups (`Hibernating customers` + `Lost customers`) account for about 27% of the customer base, indicating a large reactivation pool.
- Small segments such as `Cannot Lose Them` (about 90 customers, ~2%) should still be prioritized with high-touch retention due to strategic value.

### 2. 💰 Revenue by Segment

This chart compares total monetary contribution by segment.
![Revenue by Segment](https://github.com/user-attachments/assets/06078f22-f122-4e02-bbbd-4f9672939121)

**Key insight:**
- `Champions` contributes about 5.6M in revenue, far above `Loyal` (~1.0M) and `At Risk` (~0.75M).
- The top 3 segments (`Champions`, `Loyal`, `At Risk`) contribute roughly 7.35M, or about 84% of total segment revenue.
- Revenue concentration is very high, so protecting top segments has a much larger impact than broad untargeted campaigns.

### 3. 📈 Distribution of Core RFM Metrics

The notebook uses histograms and boxplots for `Recency`, `Frequency`, and `Monetary`.
![Distribution of RFM Metrics](https://github.com/user-attachments/assets/f4a43e26-9f25-4a5c-9cfe-feab7d9aae39)

**Key insight:**
- `Recency` spans roughly 20 to 390 days and is right-skewed, with many customers clustered in the most recent bins and a long inactive tail.
- `Frequency` is highly concentrated at low values (mostly 1-3 invoices) with a long tail extending above 200 invoices.
- `Monetary` (log1p) is centered around 6-7 but has a long upper tail, confirming that a small customer group drives disproportionate value.

### 4. 🔢 RFM Score Distribution

This view checks how customers are distributed across score bands for `R`, `F`, and `M`.
![Distribution of RFM Scores](https://github.com/user-attachments/assets/75161af8-678d-4f8a-a27e-1e79616e000c)

**Key insight:**
- Each score band (1-5) contains roughly 860-880 customers for `R`, `F`, and `M`.
- This near-uniform distribution confirms quintile scoring is balanced, with each bucket close to ~20% of customers.
- Balanced score bins make segment rules more stable for campaign targeting and performance tracking.

### 5. 🔥 Correlation Heatmap

The correlation matrix compares the relationships between `Recency`, `Frequency`, and `Monetary`.
![Correlation Heatmap](https://github.com/user-attachments/assets/abe186ad-f335-4237-9a28-1ca369998dde)

**Key insight:**
- Correlation values are: `Frequency-Monetary = 0.55`, `Recency-Frequency = -0.26`, and `Recency-Monetary = -0.12`.
- `Frequency` has the strongest positive relationship with value, making it the clearest growth signal.
- As recency worsens, repeat behavior declines, so recency should be monitored as an early churn-risk indicator.

### 6. 🧩 Score Matrix Heatmaps

The notebook generates:

- a customer-density matrix by `R` and `F`
- an average monetary matrix by `R` and `M`
![Score Matrix Heatmaps](https://github.com/user-attachments/assets/221c7f6c-2d45-4079-97f1-48c4ce0acdd7)

**Key insight:**
- In the `R-F` density map, the largest active-value cell is `R=5, F=5` with 439 customers, while low-engagement pockets remain at `R=1, F=1` with 362 customers.
- In the `R-M` value map, the `M=5` band is extremely high at both recency extremes (`R=1: 10,924` and `R=5: 10,686`).
- This indicates two important value pools: active high-value customers to protect and stale high-value customers to win back quickly.

### 7. 🗺️ RFM Segmentation Dimension Map

This chart shows the dominant segment across combinations of Recency score and FM band.
![Segmentation Dimension Map](https://github.com/user-attachments/assets/0b697ac2-7839-4552-a841-6ca8966661d9)

**Key insight:**
- High FM and high R zones are dominated by `Champions` (for example: `FM5-R5 = 348`, `FM5-R4 = 175`, `FM4-R5 = 209`).
- `At Risk` dominates weaker-recency but medium/high-FM areas (`FM4-R2 = 242`).
- Churn-heavy zones are clear in low R and low FM, especially `Lost` (`FM2-R1 = 304`, `FM1-R1 = 182`).

### 8. ⚖️ Segment Share by Customers and Revenue

This two-panel comparison shows customer share and revenue share by segment.
![Segment Share by Customers and Revenue](https://github.com/user-attachments/assets/02e85a52-f57a-4047-93f4-120d4e360f27)

**Key insight:**
- `Champions` represents about 19% of customers but contributes about 63% of revenue.
- `Hibernating customers` holds about 16% of customers but only about 3% of revenue; `Lost customers` is about 11% of customers but around 1% of revenue.
- Customer volume and revenue value are structurally different, so budget allocation should prioritize revenue share over customer share.

### 9. 📉 Pareto Revenue by Segment

The Pareto chart ranks segments by revenue and overlays cumulative contribution.
![Pareto Revenue by Segment](https://github.com/user-attachments/assets/dd8e4520-71a9-4795-b52c-85e8e707cfa2)

**Key insight:**
- `Champions` alone contributes about 63% of revenue in the Pareto curve.
- The 80% cumulative threshold is crossed within the top 3 segments (`Champions + Loyal + At Risk`), reaching about 84%.
- Top 4 segments contribute roughly 89%, confirming strong concentration and clear prioritization logic.

### 10. 🫧 Behavior Bubble Chart

This chart plots `Recency` against `Monetary`, with bubble size representing `Frequency`.
![Behavior Bubble Chart](https://github.com/user-attachments/assets/91e390d0-37cf-44b6-99ab-f940b7d12542)

**Key insight:**
- `Champions` points are concentrated at low recency (~20-60 days) with high monetary values (many above 1,000 and several above 10,000).
- `At Risk` spreads across recency ~90-350 days with meaningful monetary values (~500-3,000), indicating recoverable high-value customers.
- `Lost customers` clusters at high recency (~200-390) and low monetary (mostly below 400), fitting low-cost automated reactivation approaches.

### 11. 🌡️ Segment Profile Heatmap

The normalized heatmap compares segment-level averages for Recency, Frequency, and Monetary.
![Segment Profile Heatmap](https://github.com/user-attachments/assets/fc395cbd-69fe-4409-8680-4828e178ac51)

**Key insight:**
- `Champions` has the strongest profile (`AvgRecency = 31.8`, `AvgFrequency = 12.1`, `AvgMonetary = 6,711.8`).
- High-risk segments still carry large value: `At Risk` (`164.4`, `3.8`, `1,781.5`) and `Cannot Lose Them` (`256.2`, `2.3`, `2,238.1`).
- `Lost customers` is weakest (`296.6`, `1.1`, `200.4`), showing very low current activity and value.

### 12. 📆 Monthly Revenue Trend by Top Segments

The notebook tracks monthly revenue for the top revenue-generating segments.
![Monthly Revenue Trend by Top Segments](https://github.com/user-attachments/assets/4bc48808-687e-43f6-ab12-38e88ef440e8)

**Key insight:**
- `Champions` revenue ranges from about 250k (Apr) to about 820k (Nov), with sustained strength above 600k in Sep-Oct.
- `Loyal` peaks around 215k (Oct) then declines sharply (~105k in Nov, near zero in Dec), while `Need Attention` spikes to around 180k in Dec.
- Segment volatility is different by month, so campaign timing and intervention triggers should be segment-specific.

### 🧠 Overall Analytical Takeaway

Core chart evidence points to one clear pattern: revenue is concentrated, and repeat behavior is the strongest value driver.

- `Champions` is about **19% of customers** but contributes about **63% of revenue**.
- Pareto confirms concentration: top 3 segments (`Champions + Loyal + At Risk`) already contribute about **84%** of total revenue.
- Correlation supports operating logic: `Frequency-Monetary = 0.55` (strongest positive link), while `Recency-Frequency = -0.26` (engagement drops as inactivity rises).
- Value-at-risk is meaningful: `At Risk AvgMonetary = 1,781.5` and `Cannot Lose Them AvgMonetary = 2,238.1`.

---

## 💡 Insight & Recommendation

### 🎯 Strategic Insight

- Growth should focus on **high-value active segments**, not broad campaigns.
- Retention and recovery should be separated: `Champions/Loyal` need protection, while `At Risk/Cannot Lose Them` need fast reactivation.
- Dormant mass segments (`Hibernating` and `Lost`) should be managed with low-cost automation due to low revenue yield.

### 🚀 Why Frequency Is the Top Priority
- customers who buy more often also tend to generate more revenue
- strong segments such as Champions and Loyal are consistently supported by repeat-purchase behavior
- improving purchase frequency creates more stable growth than depending only on occasional large orders

### 📌 Recommended Priority Order

1. **Frequency (F):** primary growth KPI
2. **Recency (R):** early-warning trigger for churn prevention
3. **Monetary (M):** outcome KPI to validate value uplift

### 🛠️ Recommendation (Actionable and Concise)

1. **Protect Core Revenue (`Champions`, `Loyal`)**  
Use VIP offers, loyalty benefits, and service-priority programs to defend the revenue base.

2. **Recover High-Value Risk (`At Risk`, `Cannot Lose Them`)**  
Trigger win-back flows when recency worsens (time-based reminders, personalized bundles, short-term offers).

3. **Automate Dormant Segments (`Hibernating`, `Lost`)**  
Use low-cost nurture campaigns and strict discount control to avoid margin erosion.


### ✅ Final Conclusion

The most practical operating model from this analysis is: **increase purchase frequency, intervene quickly when recency declines, and measure monetary uplift by segment**. This keeps strategy data-driven, focused, and scalable.
