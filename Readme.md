# Predictive Paradox — Next Hour Power Demand Forecasting

---

## Overview

This repository presents a complete machine learning pipeline for **one-step-ahead (next-hour) electricity demand forecasting** on national power grid. The target variable is `demand_mw` at time *t+1*, predicted using information available only up to time *t* — a strict model with no data leakage.

The pipeline integrates three data sources (hourly power demand, hourly weather, annual macro-economic indicators from the World Bank) and uses carefully processed temporal, weather, and economic features to train classical machine learning models.The model is trained on all data through end of 2023 and evaluated on the full year 2024.

---

## Repository Structure

- `predictive_paradox.ipynb` :- Main notebook (complete end-to-end     pipeline with proper markdown and comments).
- `README.md` :- Serves as the summary report.
- `PGCB_date_power_demand.csv` :- Hourly grid demand & generation (primary source).
- `weather_data.csv` :- Hourly weather (temperature, humidity, etc.).
- `economic_full_1.csv` :- Annual economic macro-indicators (World Bank).

## Summary Report

### A. Power Demand Data Preparation: What Was Done and Why :-

- `Handling Half-Hourly (:30) Timestamps :-` The original PGCB dataset contains timestamps ending in `:30` (e.g., 2:30, 3:30). Since our target is an hourly time series, each `:30` reading must be merged with its parent hour.

    The approach is a **50/50 weighted average**

    CASE-1:- For each floor hour that has both a `:00` reading and a `:30` reading, the merged hourly value is computed as `0.5 × (value at :00) + 0.5 × (value at :30)`. 

    CASE-2:- Only `:30` reading exists but no matching `:00` was recorded, the `:30` value is used directly as the hourly representative.

- `Handling Duplicate Hourly Timestamps :-` Some timestamps appear more than once in  **the PGCB_date_power_demand dataset**. These are resolved by taking the **mean** across all duplicate entries.

- `Handling Missing Timestamps (Gaps in the Series) :-` There are some hours where no measurement was recorded at all. So we apply a **forward-fill**: fill the missing timestamps with previous available values.

- `Anomaly Detection & Removal :-`The original data contains severe undocumented spikes — values as low as 6 MW and as high as 156,050 MW.

    We use **IQR Method** to handle these outliers.It applies a fence at **1.5xIQR** which catches extreme values.For **load_shedding** different ratio is used as it contains a lot of zeros.

    3-columns **wind**,**india_adani**,**nepal** contain a lot of NaN values(more than 80%)... these values can negatively impact model performance, so these three columns are dropped.

- `Columns dropped :-` Since generation variables are highly correlated with demand, including them may introduce leakage, so we have to drop those columns which contain any type of power generation data.Even The model must not have access to future demand values as we have to predict it. Failing to remove these columns will lead to data leakage.**generation_mw**,**demand_mw**,**gas**,**liquid_fuel**,**coal**,**hydro**,**solar**,**india_bheramara_hvdc**,**india_tripura** - these columns contain generation data, so we drop them.

### B. Feature Engineering: Encoding Time for Non-Sequential Models :-

Non-sequential models treat each row as an independent observation with no notion of temporal order. However, electricity demand is inherently sequential.The model should know about past demand values too.

- `Calendar features :-` capture the repeating periodic patterns that dominate electricity demand.

    `hour` :- captures hourly patterns.  
    `day` :- capture daily patterns.  
    `month` :- capture monthly patterns.  
    `weekend` :- capture weekend patterns(important because of holidays).

- `Lag features :-` It give the model direct access to recent past demand values. 

    `lag_1h` :- model get to know about previous 1 hour demand values  
    `lag_24h` :- model get to know about previous 24 hour demand values(same hour yesterday).  
    `lag_168h` :- model get to know about previous week demand values(same hour last week).

    All lags use `shift(n)` with n ≥ 1, so the feature at time *t* only ever looks at data from *t−1* or earlier — zero leakage.

- `cyclical encoding :-` The sin/cos pair maps the circular structure of time correctly so the model experiences no artificial discontinuity at period boundaries. Without this, a raw integer encoding would treat hour 23 and hour 0 as numerically distant when they are physically adjacent.

- `Rolling features :-` summarise recent demand behaviour beyond single past points. Rolling means over 3h, 6h, 12h, 24h, and 168h windows capture the prevailing demand level at different time scales.

    `rolling_mean_24` :- Taking mean of past 24 hours demand data.  
    `rolling_std_24` :- Taking standard deviation of past 24 hours data. 

### C. Economical Indicator selection :- 

Some indicators from economic dataset are selected based on direct causal or correlational links to electricity.A few of the selected indicators are listed below :- 

1. **GDP (current US$)** :-GDP and GDP growth capture overall economic activity — more industrial and commercial output means more power consumed.
2. **GDP per capita (current US$)** :- reflects rising household incomes, which drive appliance adoption (fans, refrigerators, air conditioners)
3. **Industry (including construction), value added (current US$)** :- captures the heavy manufacturing sector (garments, steel, cement) which accounts for a major share of peak demand
4. **Population, total** :- more population means more demand.
5. **Urban population (% of total population)** :-urban residents consume 3–5× more electricity than rural ones.

### D. Train / Test Strategy :- 

- All data through 31 December 2023 forms the training set. 
- All of 2024 is the strict chronological hold-out test set.

The model never sees any 2024 data during training. This ensures zero data leakage.

### E. Model Selection :- 

Five models are compared: Linear Regression, Decision Tree, Random Forest, XGBoost and LightGBM

1. `LightGBM`: It uses leaf-wise tree growth and histogram binning, making it typically the fastest and most accurate on large tabular time-series problems. 
2. `XGBoost`:It uses level-wise growth, is slightly more conservative, and often complements LightGBM well. 
3. `Random Forest`: It is an ensemble of independently trained trees — slower but robust and highly interpretable.

### G. Evaluation Metric :- 

MAPE (Mean Absolute Percentage Error) is used as the primary metric, as specified. 

### H. Feature Importances :- 

The feature importance analysis consistently shows that **temporal lag features** — specifically `lag_1h`, `lag_24h`,`lag_168h`,**rolling features** , **calendar features**,**weather features**, specially temperature,heat index etc dominates over other features.  

These are expected and well-supported by the electricity demand forecasting literature: and the best predictor of next hour's demand is what demand was recently doing.
