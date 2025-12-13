I sincerely apologize for the frustration. Here is the \*\*entire\*\* content of the `README.md` file, including all the expanded details, enclosed in a single raw code block for easy copying.



````markdown

\# 🏙️ Urban Intelligence: Forecasting PM2.5 in New Delhi



!\[Python](https://img.shields.io/badge/Python-3.10%2B-blue)

!\[Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)

!\[Status](https://img.shields.io/badge/Status-Completed-success)



\## 📖 Project Overview

This project addresses the critical issue of air pollution in New Delhi by developing a robust machine learning pipeline to forecast daily \*\*PM2.5 concentrations\*\*. 



Using a \*\*Hybrid Ensemble Voting Model\*\*, the system integrates historical pollution data (2019-2025), meteorological conditions, and solar radiation data to predict air quality levels. The project culminates in an interactive \*\*Streamlit Web Dashboard\*\* that not only predicts pollution levels but also provides actionable health insights for the public.



\## 📂 Project Structure

The codebase is organized as follows:



```text

├── app/                 # Source code for the Streamlit Web Dashboard (PoC)

├── data/                # Data storage

│   ├── raw/             # Original immutable data

│   │   ├── pm25/        # OpenAQ PM2.5 data (2019-2025)

│   │   ├── meteorological/ # Visual Crossing weather data

│   │   └── solar/       # Copernicus solar radiation \& boundary layer data (NetCDF)

│   ├── interim/         # Intermediate transformed data

│   └── processed/       # Final datasets merged and ready for modeling

├── models/              # Serialized trained models (CatBoost, XGBoost, etc.)

├── notebooks/           # Jupyter notebooks for EDA and experimentation

├── src/                 # Modular source code for the pipeline

│   ├── data/            # Scripts to fetch, clean, and impute data

│   ├── features/        # Feature engineering logic (Lags, Rolling stats, Cyclical)

│   └── models/          # Model training, grid search, and evaluation scripts

└── requirements.txt     # Python dependencies

````



\## 🛠️ Tech Stack \& Dependencies



The project relies on a comprehensive stack of data science and engineering libraries:



| Category | Libraries |

| :--- | :--- |

| \*\*Core\*\* | `numpy`, `pandas`, `scipy` |

| \*\*Machine Learning\*\* | `scikit-learn`, `catboost`, `xgboost`, `lightgbm` |

| \*\*Visualization\*\* | `plotly`, `matplotlib`, `seaborn`, `pydeck`, `altair` |

| \*\*Web Application\*\* | `streamlit`, `flask` |

| \*\*Data Handling\*\* | `xarray`, `netCDF4` (for .nc files), `pyarrow` |



\## 🚀 Installation \& Setup



1\.  \*\*Clone the repository:\*\*



&nbsp;   ```bash

&nbsp;   git clone \[https://github.com/YOUR\_USERNAME/YOUR\_REPO\_NAME.git](https://github.com/YOUR\_USERNAME/YOUR\_REPO\_NAME.git)

&nbsp;   cd ml\_proj

&nbsp;   ```



2\.  \*\*Set up the environment:\*\*

&nbsp;   It is recommended to use a virtual environment to avoid conflicts.



&nbsp;   ```bash

&nbsp;   # Create virtual environment

&nbsp;   python -m venv venv



&nbsp;   # Activate (Windows)

&nbsp;   .\\venv\\Scripts\\activate



&nbsp;   # Activate (Mac/Linux)

&nbsp;   source venv/bin/activate

&nbsp;   ```



3\.  \*\*Install dependencies:\*\*



&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```



\## 📊 Data Pipeline



\### 1\\. Data Acquisition



The dataset spans \*\*2019 to 2025\*\* and aggregates data from three distinct sources:



&nbsp; \* \*\*OpenAQ:\*\* Daily average PM2.5 levels.

&nbsp; \* \*\*Visual Crossing:\*\* Meteorological data (Temperature, Humidity, Windspeed, Precipitation).

&nbsp; \* \*\*Copernicus Climate Data Store:\*\* Solar Radiation and Boundary Layer Height.



\### 2\\. Preprocessing \& Cleaning



&nbsp; \* \*\*Outlier Removal:\*\* Removed negative and impossible PM2.5 values.

&nbsp; \* \*\*Gap Imputation:\*\* Addressed a significant \*\*109-day gap\*\* in the PM2.5 data by imputing values based on historical yearly averages, leveraging the strong seasonal patterns.

&nbsp; \* \*\*One-Hot Encoding:\*\* Applied to categorical `Conditions` (e.g., Rain, Clear).



\### 3\\. Feature Engineering



To capture temporal dependencies, we engineered the following features:



&nbsp; \* \*\*Lag Features:\*\* `lag\_1` through `lag\_7` (past 7 days of pollution).

&nbsp; \* \*\*Rolling Statistics:\*\* 7-day `moving\_mean` and `moving\_std`.

&nbsp; \* \*\*Cyclical Encoding:\*\* Sine and Cosine transformations for `Day` and `Month`.

&nbsp; \* \*\*Calendar Features:\*\* `day\_of\_week`, `is\_weekend`.



\## 📈 Model Performance



We evaluated multiple models on the \*\*2025 Test Set\*\*. The \*\*Hybrid Ensemble\*\* (Voting Regressor) outperformed individual models by combining the temporal strengths of CatBoost with the physical modeling capabilities of XGBoost.



\### Hybrid Formula



The final prediction is calculated as:

$$ Pred\_{ensemble} = (0.72 \\times Pred\_{CatBoost}) + (0.28 \\times Pred\_{XGBoost}) $$



\### Evaluation Metrics (Test Set 2025)



| Model | MAE | RMSE | MAPE |

| :--- | :--- | :--- | :--- |

| \*\*Baseline\*\* | 57.25 | 68.28 | 1.36 |

| Linear Regression | 19.71 | 30.07 | 0.37 |

| SVR | 18.38 | 30.04 | 0.38 |

| Elastic Net | 19.41 | 29.67 | 0.36 |

| XGBoost | 16.33 | 26.40 | 0.22 |

| CatBoost | 15.95 | 25.73 | 0.22 |

| \*\*Hybrid Ensemble\*\* | \*\*15.71\*\* | \*\*25.33\*\* | \*\*0.21\*\* |



\## 💻 Usage



\### Running the Dashboard (PoC)



To launch the interactive web application which demonstrates the model:



```bash

streamlit run app/main.py

```



\### Dashboard Features



The Proof of Concept (PoC) includes two main sections:



1\.  \*\*Validation:\*\* Visualize model performance against ground truth for 2025.

2\.  \*\*Prediction \& Health:\*\* Input custom weather/lag conditions to get a forecast and specific health advice:

&nbsp;     \* 🏋️ \*\*Workout:\*\* Recommendations for indoor vs. outdoor exercise.

&nbsp;     \* 🏫 \*\*Schools:\*\* Advisory on whether schools should remain open.

&nbsp;     \* 🚬 \*\*Cigarette Equivalence:\*\* Visualizing pollution in terms of cigarettes smoked.

&nbsp;     \* 😷 \*\*Protection:\*\* Mask mandates and warnings for asthma sensitivity.



\## 👥 Authors



&nbsp; \* \*\*Syed Mohammad Abdullah Rashid\*\*

&nbsp; \* \*\*Saad Saghir Minhas\*\*



\## 📜 License



This project is licensed under the MIT License - see the \[LICENSE](https://www.google.com/search?q=LICENSE) file for details.



```

```

