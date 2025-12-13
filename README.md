# 🏙️ Urban Intelligence: Forecasting PM2.5 in New Delhi

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Project Overview
This project addresses the critical issue of air pollution in New Delhi by developing a robust machine learning pipeline to forecast daily **PM2.5 concentrations**. 

Using a **Hybrid Ensemble Voting Model (CatBoost + XGBoost)**, the system integrates historical pollution data, meteorological conditions, and solar radiation data to predict air quality levels. The project culminates in an interactive **Streamlit Web Dashboard** that provides actionable health insights based on these predictions.

## 📂 Project Structure
The codebase is organized as follows:

```text
├── app/                 # Source code for the Streamlit Web Dashboard (PoC)
├── data/                # Data storage 
│   ├── raw/             # Original immutable data
│   │   ├── pm25/        # Historical PM2.5 data (2019-2025)
│   │   ├── meteorological/ # Visual Crossing weather data
│   │   └── solar/       # Copernicus solar radiation & boundary layer data
│   ├── interim/         # Intermediate transformed data
│   └── processed/       # Final datasets ready for modeling
├── models/              # Serialized trained models
├── notebooks/           # Jupyter notebooks for EDA and experimentation
├── src/                 # Modular source code for the pipeline
│   ├── data/            # Scripts to fetch and clean data
│   ├── features/        # Feature engineering logic (Lags, Rolling stats)
│   └── models/          # Model training and evaluation scripts
└── requirements.txt     # Python dependencies
````

## 🛠️ Tech Stack & Dependencies

The project relies on the following key libraries (found in `venv`):

  * **Core:** `numpy`, `pandas`, `scikit-learn`
  * **Modeling:** `catboost`, `xgboost`, `lightgbm`
  * **Visualization:** `matplotlib`, `seaborn`
  * **Dashboarding:** `streamlit`
  * **Data Handling:** `xarray`, `netCDF4` (for solar .nc files)

## 🚀 Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone git clone https://github.com/abdullahhrashid/air-quality-forecasting-using-classical-machine-learning.git project
    cd project
    ```

2.  **Set up the environment:**
    It is recommended to use a virtual environment.

    ```bash
    # Create virtual environment
    python -m venv venv

    # Activate (Windows)
    .\venv\Scripts\activate

    # Activate (Mac/Linux)
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

### Running the Dashboard (PoC)

To launch the interactive web application:

```bash
streamlit run app/app.py
```

### Reproducing the Model

To re-run the training pipeline or feature engineering:

1.  Navigate to the `notebooks/` directory to view the step-by-step EDA.
2.  Run the scripts in `src/models`.

## 📊 Data Pipeline

The dataset spans from **2019 to 2025** and aggregates data from three sources:

1.  **OpenAQ:** Daily average PM2.5 levels (Imputed using historical yearly averages for gaps).
2.  **Visual Crossing:** Temperature, Humidity, Windspeed, Precipitation.
3.  **Copernicus Climate Data Store:** Solar Radiation and Boundary Layer Height.

**Feature Engineering highlights:**

  * Lag features (1-7 days).
  * Rolling Mean & Standard Deviation (7-day window).
  * Cyclical encoding for Day and Month.

## 📈 Model Performance

The final **Hybrid Ensemble Model** achieved the following on the 2025 Test Set:

| Metric | Score |
| :--- | :--- |
| **MAE** | **15.71** |
| **RMSE** | **25.33** |
| **MAPE** | **0.21** |

*The model effectively captures seasonal trends and demonstrates low bias, though it remains conservative regarding extreme outlier events.*

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
