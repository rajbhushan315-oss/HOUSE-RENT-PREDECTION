# 🏠 House Rent Prediction (Chennai): Linear Regression Built From Scratch

Predicting monthly rent for Chennai flats using **linear regression written from scratch with NumPy gradient descent**, then improved step by step through feature engineering, a log-transformed target, and hand-grouped locality data. The from-scratch model is cross-checked against scikit-learn to prove the maths is right.

> **Result:** test R² went from **0.54 → 0.73 (log-rent) / 0.78 (rupee scale)**, and the average error dropped from about ₹9,200 to about ₹3,900.

---

## 📌 Why this project
I learned gradient descent in a Coursera ML course and wanted to apply it to real, messy data instead of a clean tutorial dataset. The goal was not only to get a good score but to understand *why* each change helped, so every step is a separate notebook and the whole journey is documented.

## 📂 Dataset
- Kaggle **House Rent Prediction Dataset**: 4,746 listings across 6 Indian cities (May–July 2022).
- Columns: `BHK, Rent, Size, Floor, Area Type, Area Locality, City, Furnishing Status, Tenant Preferred, Bathroom, Point of Contact`.
- Download `House_Rent_Dataset.csv` from Kaggle and put it in a `data/` folder (not included here).

**Why only Chennai?** I compared rent per sq ft across cities (see chart below). Chennai (891 listings) had a much tighter spread than Delhi, Mumbai, Hyderabad or Bangalore, whose standard deviations were 2–4× larger, so its data is more uniform and has fewer extreme outliers. Mixing cities would have forced the model to learn city effects before learning anything about the flats themselves.

| City | Listings | Mean ₹/sq ft | Std |
|---|---|---|---|
| Chennai | 891 | 19.8 | 23.9 |
| Kolkata | 524 | 16.5 | 19.6 |
| Bangalore | 886 | 21.8 | 47.9 |
| Hyderabad | 868 | 23.6 | 70.6 |
| Delhi | 605 | 67.6 | 90.5 |
| Mumbai | 972 | 81.7 | 41.5 |

![City-wise boxplots](images/03_citywise_rent_per_sqft_boxplots.png)

---

## 🗺️ The journey (one notebook per stage)

| # | Notebook | What changed | Test R² | Test MAE |
|---|---|---|---|---|
| 1 | `01_univariate_size_vs_rent` | Size only, raw rent | **0.535** | ₹9,156 |
| 2 | `02_multivariate_raw_rent` | 6 numeric features, raw rent | 0.494 | ₹6,047 |
| 3 | `03_multivariate_log_rent` | Same features, **log(rent)** target | 0.576 (log) | ₹5,418 |
| 4a | `04_categorical_features_and_pruning` | + Area Type, Tenant Preferred, Point of Contact | 0.670 (log) / 0.702 (₹) | ₹4,370 |
| 4b | same notebook | Dropped Total Floors and Area Type | 0.672 (log) / 0.707 (₹) | ₹4,357 |
| 5 | `05_location_groups_final` | + grouped **Area Locality** (19 groups) | **0.732 (log) / 0.775 (₹)** | **₹3,914** |

*"(log)" means R² measured on log(rent); "(₹)" means measured after converting predictions back to rupees. Rows 1 and 2 use different row sets (see notes), so compare trends rather than exact decimals.*

### Stage 1: EDA and one feature (Size vs Rent)
- Checked the correlation matrix: **Size is the strongest single predictor of Rent (r = 0.74)**. BHK (0.55) and Bathroom (0.56) follow, while Floor number and Total floors are weak (about 0.17–0.18).
- Removed one extreme listing (₹6,00,000 rent for a 950 sq ft flat, ₹632/sq ft) found via boxplot.
- Wrote gradient descent from scratch (z-score scaling using training statistics only, 80/20 split).
- **Result: R² = 0.543 (train), 0.535 (test).** This agrees with the correlation: r² = 0.74² ≈ 0.55, which means the size-only line explains about half of the variance.

![Correlation heatmap](images/06_corr_heatmap_univariate_stage.png)
![Size vs rent fit](images/09_uni_fit_line_train.png)

### Stage 2: Multiple features (raw rent): R² didn't improve
Adding BHK, Furnishing, Floor number, Total floors and Bathroom (and capping rent below ₹1,00,000) gave test R² of only **0.494**. The residual plot showed errors fanning out as rent grew, a classic sign that the target is skewed.

### Stage 3: Log-transforming the target
**Why log(rent)?**
1. Rent is heavily right-skewed (mean ₹20,964 vs median ₹14,000, max ₹3,30,000), so a few expensive flats dominate the squared-error cost.
2. Rent behaves *multiplicatively* (an extra bedroom adds a percentage, not a fixed amount), and log turns that into a straight-line relationship.
3. It stabilises the variance of the errors and guarantees predictions stay positive after `exp()`.
4. Gradient descent converges more smoothly (cost dropped from about 4×10⁷ to about 0.065).

Result: test R² **0.576** on log scale (train 0.636), and rupee-scale MAE improved from ₹6,047 to ₹5,418. The coefficients also became interpretable: each extra BHK raises rent by roughly **24%** (e^0.22).

### Stage 4: Adding categorical features (and removing useless ones)
One-hot encoded **Area Type, Tenant Preferred, Point of Contact**. I expected little gain, but R² jumped to **0.67 (log) / 0.70 (₹)** and RMSE fell from ₹8,827 to ₹6,812.
- **Point of Contact** was the biggest surprise: listings posted by the owner were about **36% cheaper** than agent listings (coefficient −0.445 on log-rent, other features equal).
- Features with near-zero learned **weights (coefficients)**, namely **Total Floors** and **Area Type**, were dropped. Removing them did not hurt and slightly improved R² (0.7065 vs 0.7016), which is a good sign of a leaner model.
- I also tried a `Size × Owner` interaction term. It gave no meaningful gain (0.7046), so I did not keep it.

### Stage 5: Area Locality (the hard one)
`Area Locality` has dozens of unique values, many appearing only once or twice, so one-hot encoding it directly would create too many sparse columns. I **grouped localities into 18 Chennai zones + "Other"** (e.g. OMR, ECR, Velachery, T-Nagar–Mambalam, Mylapore–Teynampet, Tambaram–Perungalathur…) using a keyword-matching function written with AI assistance and checked against Chennai geography.

**Result: test R² 0.732 (log) / 0.775 (₹), MAE ₹3,914.** The learned location effects match real-world intuition:

| Zone | Rent effect vs. average (approx.) |
|---|---|
| Mylapore–Teynampet | +58% |
| Adyar–Besant Nagar | +50% |
| T-Nagar–Mambalam | +47% |
| Ambattur–Avadi | −21% |
| Outer West–South | −21% |
| Tambaram–Perungalathur | −21% |

![Final actual vs predicted](images/25_final_actual_vs_pred.png)
![Final residuals](images/26_final_residuals.png)

---

## ✅ Validation: was my gradient descent correct?
I'm new to scikit-learn, so I used it purely as a **referee**. I fitted `LinearRegression` on the same scaled data and target and compared:

| Check | Result |
|---|---|
| Train / test R² (log scale) | **Identical to 4 decimals** (0.7564 / 0.7317) |
| Max difference in predictions | ≈ 0.003 in log units (≈ 0.3% in rent) |
| Intercept | Matches to 10 decimals in scaled space |

Individual coefficients differ by about 10⁻³: gradient descent stops after 2,500 epochs at lr = 0.01 instead of solving the equations exactly, and the 19 location dummies are perfectly collinear with each other (see improvements).

## 🔍 Is the model underfitting?
Evidence says **not badly**:
- Train and test R² are close (0.756 vs 0.732 on log scale), so there's no large gap either way.
- **5-fold cross-validation** gives log-R² of 0.680 without location and **0.737 with location**.
- A much more flexible **Gradient Boosting Regressor** on the same features reaches only **0.739** (MAE ₹4,155 vs ₹4,363 for linear regression under CV). If the model were badly underfitting, a tree model should beat it by a wide margin; it doesn't. The remaining error mostly comes from information the dataset doesn't contain (flat condition, amenities, exact street, building age).

## ⚠️ Limitations (honest notes)
- The model **under-predicts expensive flats** (above roughly ₹40k): see the residual fan-out in the final chart. Rent was capped at ₹1,00,000 for modelling, so it is not meant for luxury listings.
- The test set is small (~174 rows), so a single split's R² is noisy (the rupee-scale test R² is even higher than train). Cross-validated numbers are the more reliable ones.
- Stage 1 used all Chennai rows except one outlier, while later stages cap rent at ₹1,00,000, so stage 1 and 2 numbers are not strictly like-for-like.
- Only about 890 listings from a 3-month window in 2022 were used.

## 🔧 Next Project
- [ ] Extend to other cities and their locality.

## 🛠️ Tech stack
Python · NumPy · Pandas · Matplotlib · Seaborn · scikit-learn (validation only) · Jupyter

## ▶️ How to run

Prerequisites

* Python 3.9 or later
* Git

1. Clone the Repository
```bash
git clone https://github.com/rajbhushan315-oss/HOUSE-RENT-PREDECTION.git
```
2. Navigate to the Project Directory
```bash
cd HOUSE-RENT-PREDECTION
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/
```
The Jupyter Notebook interface will open in your browser, allowing you to explore the notebooks and run the code.

## 📚 What I learned
* Correlation vs. R²: A correlation of 0.74 translates to an R² of approximately 0.55 in simple linear regression.
* Data Leakage: Always calculate scaling parameters using training data only.
* Feature Engineering: Log transformations and meaningful categorical features can improve predictions more than increasing model complexity.
* Experimentation Over Assumptions: Features like contact type and tenant preference proved more valuable than I initially expected.
* Gradient Descent: Feature scaling and learning rate significantly affect convergence and training stability.
* Model Evaluation: R², MAE, and RMSE reveal different aspects of performance; no single metric tells the whole story.
* Visual Diagnostics: Cost curves, actual-vs-predicted plots, and residual plots help identify issues that metrics alone cannot.
* Outlier Analysis: Investigate unusual observations before removing them, as they may represent genuine market variations.
* Validation: Comparing my implementation with Scikit-learn and using cross-validation helped verify correctness and assess generalization.
* Iterative Learning: Every experiment should test a hypothesis and help explain why a model performs better or worse.
