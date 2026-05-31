# Machine Learning Foundations: Python, Linear Algebra & Statistics

This repository hosts a comprehensive foundational suite of tasks covering core data engineering and mathematical principles required for deep machine learning implementations. Every section avoids opaque black-box framework logic to focus on underlying structural code mechanics.


## 🛠️ Detailed Implementations

### 1. Python Fundamentals & Defensive Programming
* **Control Flow Automation:** Formulated numerical evaluation algorithms using clean, boundary-validated conditional pathways.
* **Algorithmic Optimization:** Processed dynamic string frequencies manually to avoid standard library structures, while utilizing advanced list comprehensions and sets for optimized memory operations.
* **Exception Layer Design:** Constructed robust, runtime-safe division handlers built to safely intercept arithmetic anomalies (`ZeroDivisionError`) and raise explicit feedback alerts (`TypeError`) when catching non-numeric parameters.

### 2. Scientific Computing with NumPy
* **Multi-Dimensional Transformations:** Conducted dimensional mutations across structural representations from initial 1D vectors up to highly targeted 3D multi-layered matrices (`2x2x3`).
* **Matrix Operations & Invariance Checking:** Evaluated element-wise array operations alongside traditional inner dot-products, while explicitly demonstrating non-commutative spatial behavior ($P \times Q \neq Q \times P$).

### 3. Data Wrangling & Feature Profiling (Pandas)
* **Relational Query Slicing:** Isolated key target parameters using absolute indexing approaches (`.loc` vs `.iloc`) based on composite masking operations.
* **Production Preprocessing Pipelines:** Formulated complete missing value recovery tracks utilizing dynamic localized statistics (median value column imputations for target outliers, and mean-shift standard approximations for age metrics).

### 4. Linear Algebra & Dimensionality Reduction
* **Euclidean Norm Calculations:** Computed matrix $L_2$ properties and plotted continuous directional coordinates utilizing spatial quiver visualizations.
* **Spectral Analysis (Eigenpair Extraction):** Decomposed linear configurations to yield core transformation properties (Eigenvalues & Eigenvectors). Formulated assertion layers verifying the fundamental identity vector relation:
  $$A\vec{v} = \lambda\vec{v}$$
* **Data Matrix Compression (SVD):** Implemented Singular Value Decomposition to break arbitrary observation profiles into principal coordinates ($U, \Sigma, V^T$). Executed low-rank approximations using leading dominant singular elements to replicate structural PCA patterns.

### 5. Descriptive Statistics & Exploration
* **Distribution Characterization:** Computed complete numerical profiles (Mean, Median, Standard Deviation, and Interquartile Ranges) on continuous fields.
* **Density Visualizations:** Plotted kernel density estimates (KDE) over frequency histograms to map production variations against standard normal profiles.

* # **Week2 Task**
# Tesla Monthly Global Deliveries Forecasting Pipeline
## 📌 Project Overview
This project designs and implements an end-to-end Machine Learning and Statistical Time Series pipeline to predict **Tesla's Monthly Global Vehicle Deliveries**. Utilizing real-world operational tracking datasets spanning from 2015 through 2025, the workflow evaluates predictive capabilities under strict out-of-sample chronological forecasting boundaries.
The primary engineering focus of this project was the structural refactoring of an initial high-performing regression model to eliminate a severe case of **Target Leakage / Look-Ahead Bias**, ensuring true predictive utility in real-world deployment scenarios.
## 🛠️ Tech Stack & Dependencies
 * **Data Analysis & Profiling:** Pandas, NumPy
 * **Data Visualization:** Matplotlib, Seaborn
 * **Machine Learning Models:** Scikit-Learn (Gradient Boosting Regressor, Random Forest Regressor)
 * **Statistical Modeling:** Statsmodels (Holt-Winters Exponential Smoothing)
 * **Data Ingestion:** KaggleHub
## 🚀 Pipeline Architecture
### 1. Ingestion & Preprocessing
 * Automated raw file downloads directly through kagglehub.
 * Cleaned whitespace inconsistencies in structural column headers.
 * Converted fractional time metrics (Year, Month) into a standardized DatetimeIndex.
 * Aggregated granular, regional, and model-specific entries to compute continuous **Unified Global Monthly Totals**.
### 2. Exploratory Data Analysis (EDA)
 * Profiled selling price distributions using Kernel Density Estimate (KDE) histograms.
 * Analyzed total production volume distribution trends categorized by specific vehicle classes.
 * Generated a numeric metric correlation matrix to examine multi-collinear interactions between production output and consumer deliveries.
### 3. Feature Engineering & Target Transformation
To construct a valid forecasting environment, the feature matrix was updated using two major techniques:
 * **Operational Feature Lagging:** Key predictors (Production_Units, Avg_Price_USD) were lagged by 1 month (\text{Lag}_1), ensuring future target points are calculated exclusively using verified historical context.
 * **Stationary Target Differencing:** To overcome the mathematical limitation where tree-based algorithms struggle to extrapolate outside their historical absolute training scale limits, the target metric was transformed into a **Month-over-Month Change** series:
   
### 4. Validation Framework
 * **Chronological Splitting:** Avoided standard randomized shuffling (train_test_split) which breaks sequential dependencies.
 * **Holdout Test Set:** Segmented the final 12 unique months (**2025-01 to 2025-12**) as an unseen holdout window.
 * **Cross-Validation:** Applied TimeSeriesSplit cross-validation during the optimization loops to secure parameters without look-ahead contamination.
## 🚨 Resolving the Target Leakage Trap
During initial baseline evaluations, standard regression setups yielded an artificial **R^2 Score of 0.86 to 0.95**. Technical audit revealed that this high performance was a structural flaw caused by target leakage:
 1. Concurrent production units for the same target month were available in the feature matrix.
 2. Because Tesla manufactures vehicles to fulfill active order queues, current production has a **0.99 correlation** with current deliveries.
### The Solution:
By explicitly dropping all current-month indicators from the input matrix, shifting operational metrics back by 1 month, and training a **Tuned Gradient Boosting Regressor** exclusively on differenced variations, the look-ahead bias was entirely eliminated. The predicted changes were reconstructed back into true absolute figures during post-processing using the preceding month's actual baseline:

## 📊 Final Performance Scoreboard (2025 Validation Horizon)
| Model Strategy | RMSE | MAE | R^2 Score |
|---|---|---|---|
| **Tuned Gradient Boosting (Stationary Pipeline)** | 14,865.62 | 13,039.42 | **-0.34** |
| **Holt-Winters Baseline (Time Series)** | 11,741.93 | 9,483.22 | **0.17** |
### Key Strategic Insights:
 * **Statistical Resilience:** The classical Holt-Winters model achieved a positive R^2 score of **0.17**, outperforming machine learning models by cutting through month-over-month noise to model Tesla's long-term 12-month cyclical corporate delivery loops.
 * **Volatility Limits:** Pure history-based autoregressive lags cannot anticipate sudden real-world global market adjustments (such as the sharp V-shaped plunge observed in May 2025) without external leading indicators.
## 🔮 Future Enhancements (Phase 2 Roadmap)
 1. **Exogenous Feature Integration:** Incorporate global macroeconomic signals, including consumer price indices (CPI), electric vehicle battery raw material costs, and regional interest rate updates.
 2. **Hybrid Model Architectures:** Implement deep learning sequence models (LSTM, GRU) paired with linear trend-residual structural decompositions.

---

## 💻 Technical Setup & Stack
* **Language Environment:** Python 3 (Google Colab Environment)
* **Core Packages:** `numpy`, `pandas`, `scipy.stats`, `matplotlib`, `seaborn`
