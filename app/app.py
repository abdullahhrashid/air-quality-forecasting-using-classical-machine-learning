import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

st.set_page_config(
    page_title="Delhi Air Quality Forecaster",
    page_icon="🍃",
    layout="wide"
)

EXPECTED_COLS = [
    'solar_radiation', 'boundary_layer_height', 'temp', 'humidity', 
    'precip', 'precipprob', 'precipcover', 'windspeed', 'pressure', 
    'day_of_week', 'is_weekend', 
    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 
    'moving_avg_7', 'moving_std_7', 
    'month_cos', 'month_sin', 'day_cos', 'day_sin', 
    'conditions_Clear', 'conditions_Overcast', 'conditions_Partially cloudy', 
    'conditions_Rain', 'conditions_Rain, Overcast', 'conditions_Rain, Partially cloudy'
]

@st.cache_resource
def load_models():
    """Loads the pre-trained XGBoost and CatBoost models."""
    xgb_path = os.path.join(os.path.dirname(__file__), '../models/xgb.joblib')
    cat_path = os.path.join(os.path.dirname(__file__), '../models/cat.joblib')
    xgb = joblib.load(xgb_path)
    cat = joblib.load(cat_path)
    return xgb, cat

@st.cache_data
def load_data():
    """Loads the 2025 test dataset for comparison."""
    data_path = os.path.join(os.path.dirname(__file__), '../data/test_set.csv')
    df = pd.read_csv(data_path) 
    return df

def get_cigarette_equivalence(pm25):
    cigs = pm25 / 22
    return round(cigs, 2)

def get_health_advice(pm25, is_asthmatic, is_active):
    advice = []
    if pm25 > 35:
        if is_asthmatic:
            advice.append("⚠️ **ASTHMA WARNING:** Carry your inhaler.")
    
    if pm25 > 150:
        if is_active:
            advice.append("🚫 **CANCEL JOGGING:** Exercise indoors today.")
        else:
            advice.append("😷 **MASK REQUIRED:** Wear a mask if going outside.")
    
    if pm25 > 250:
        advice.append("☠️ **HAZARDOUS:** Avoid all outdoor exertion.")

    if not advice:
        return ["✅ Air quality is good! Enjoy your day."]
    return advice

def school_closure_recommendation(pm25_value):
    if pm25_value > 200:
        return "🔴 RECOMMENDATION: CLOSE SCHOOLS"
    return "🟢 Schools can remain open"

st.title("🍃 Delhi Air Quality AI Dashboard")
st.markdown("### Hybrid Ensemble Model (XGBoost + CatBoost)")

try:
    xgb_model, cat_model = load_models()
    df_test = load_data()
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}. Please ensure 'models/' folder and csv exist.")
    st.stop()

tab1, tab2 = st.tabs(["📊 2025 Forecast Analysis", "🔮 Real-Time Predictor"])

with tab1:
    st.header("Compare Actual vs. Hybrid Prediction (2025 Data)")
    
    max_idx = len(df_test) - 1
    selected_index = st.slider("Select a record index from the 2025 test set:", 0, max_idx, 0)
    
    row = df_test.iloc[[selected_index]].copy()
    actual_pm25 = row['pm25_level'].values[0] if 'pm25_level' in row.columns else 0
    
    X_input = row.drop(columns=['pm25_level', 'datetime', 'month', 'day'], errors='ignore')
    
    try:
        X_input = X_input[EXPECTED_COLS]
    except KeyError as e:
        st.error(f"Data mismatch! The CSV is missing columns: {e}")
        st.stop()
    
    # Prediction
    pred_xgb = xgb_model.predict(X_input)[0]
    pred_cat = cat_model.predict(X_input)[0]
    ensemble_pred = (0.72 * pred_cat) + (0.28 * pred_xgb)
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Actual PM2.5", f"{actual_pm25:.2f}")
    with col2: st.metric("Hybrid Prediction", f"{ensemble_pred:.2f}", delta=f"{ensemble_pred - actual_pm25:.2f}")
    with col3: st.metric("Absolute Error", f"{abs(ensemble_pred - actual_pm25):.2f}")
        
    st.dataframe(row)
    st.caption(f"Model Contribution: CatBoost ({0.72*pred_cat:.1f}) + XGBoost ({0.28*pred_xgb:.1f})")

with tab2:
    st.header("Predict PM2.5 for Custom Conditions")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("1. Atmospheric Conditions")
        
        date_input = st.date_input("Date")
        month = date_input.month
        day = date_input.day
        day_of_week = date_input.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        day_sin = np.sin(2 * np.pi * day / 31)
        day_cos = np.cos(2 * np.pi * day / 31)
        
        condition_options = ['Clear', 'Overcast', 'Partially cloudy', 'Rain', 'Rain, Overcast', 'Rain, Partially cloudy', 'Other']
        selected_condition = st.selectbox("Weather Condition", condition_options)
        
        temp = st.slider("Temperature (°C)", -5.0, 50.0, 25.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
        windspeed = st.slider("Wind Speed (km/h)", 0.0, 100.0, 10.0)
        pressure = st.slider("Pressure (hPa)", 900.0, 1100.0, 1013.0)
        solar_rad = st.slider("Solar Radiation", 0.0, 1000.0, 150.0)
        boundary_layer = st.slider("Boundary Layer Height", 0.0, 2000.0, 800.0)
        precip = st.slider("Precipitation (mm)", 0.0, 100.0, 0.0)
        precipprob = st.slider("Precip Probability (%)", 0.0, 100.0, 0.0)
        precipcover = st.slider("Precip Cover (%)", 0.0, 100.0, 0.0)

        with st.expander("Advanced Lag Features (Past 7 Days)"):
            st.info("Set past values manually. Rolling stats will be calculated automatically.")
            lag_1 = st.number_input("PM2.5 (1 day ago)", value=50.0)
            lag_2 = st.number_input("PM2.5 (2 days ago)", value=50.0)
            lag_3 = st.number_input("PM2.5 (3 days ago)", value=50.0)
            lag_4 = st.number_input("PM2.5 (4 days ago)", value=50.0)
            lag_5 = st.number_input("PM2.5 (5 days ago)", value=50.0)
            lag_6 = st.number_input("PM2.5 (6 days ago)", value=50.0)
            lag_7 = st.number_input("PM2.5 (7 days ago)", value=50.0)
            
            lag_values = [lag_1, lag_2, lag_3, lag_4, lag_5, lag_6, lag_7]
            moving_avg_7 = np.mean(lag_values)
            moving_std_7 = np.std(lag_values, ddof=1)
            
            st.caption(f"**Auto-Calculated:** Rolling Avg: {moving_avg_7:.2f} | Rolling Std: {moving_std_7:.2f}")

    cond_clear = 1 if selected_condition == 'Clear' else 0
    cond_overcast = 1 if selected_condition == 'Overcast' else 0
    cond_p_cloudy = 1 if selected_condition == 'Partially cloudy' else 0
    cond_rain = 1 if selected_condition == 'Rain' else 0
    cond_rain_overcast = 1 if selected_condition == 'Rain, Overcast' else 0
    cond_rain_p_cloudy = 1 if selected_condition == 'Rain, Partially cloudy' else 0
    
    input_data = {
        'solar_radiation': solar_rad, 'boundary_layer_height': boundary_layer, 'temp': temp, 
        'humidity': humidity, 'precip': precip, 'precipprob': precipprob, 'precipcover': precipcover, 
        'windspeed': windspeed, 'pressure': pressure, 'day_of_week': day_of_week, 'is_weekend': is_weekend,
        'lag_1': lag_1, 'lag_2': lag_2, 'lag_3': lag_3, 'lag_4': lag_4, 'lag_5': lag_5, 'lag_6': lag_6, 'lag_7': lag_7,
        'moving_avg_7': moving_avg_7, 'moving_std_7': moving_std_7,
        'month_cos': month_cos, 'month_sin': month_sin, 'day_cos': day_cos, 'day_sin': day_sin,
        'conditions_Clear': cond_clear, 'conditions_Overcast': cond_overcast, 
        'conditions_Partially cloudy': cond_p_cloudy, 'conditions_Rain': cond_rain, 
        'conditions_Rain, Overcast': cond_rain_overcast, 'conditions_Rain, Partially cloudy': cond_rain_p_cloudy
    }
    
    input_df = pd.DataFrame([input_data])
    input_df = input_df[EXPECTED_COLS]

    with col_result:
        st.subheader("2. Personal Context")
        
        col_check1, col_check2 = st.columns(2)
        with col_check1:
            is_asthma = st.checkbox("I have asthma", value=True)
        with col_check2:
            is_jogger = st.checkbox("I plan to exercise outdoors", value=True)
            
        st.divider()
        st.subheader("3. Prediction")

        if st.button("Generate Prediction", type="primary"):
            
            r_xgb = xgb_model.predict(input_df)[0]
            r_cat = cat_model.predict(input_df)[0]
            final_pred = (0.72 * r_cat) + (0.28 * r_xgb)
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                <h2 style="color: #333; margin:0;">Predicted PM2.5 Level</h2>
                <h1 style="font-size: 80px; margin:0; color: #ff4b4b;">{int(final_pred)}</h1>
                <p>µg/m³</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()

            cigs = get_cigarette_equivalence(final_pred)
            st.warning(f"🚬 **Health Impact:** Breathing this air for 24 hours is equivalent to smoking **{cigs} cigarettes**.")
            
            school_msg = school_closure_recommendation(final_pred)
            if "CLOSE" in school_msg:
                st.error(school_msg)
            else:
                st.success(school_msg)
                
            st.divider()
            
            st.subheader("Advice for You")
            advice_list = get_health_advice(final_pred, is_asthma, is_jogger)
            
            for item in advice_list:
                if "Warning" in item or "CANCEL" in item or "Mask" in item:
                    st.error(item)
                else:
                    st.success(item)