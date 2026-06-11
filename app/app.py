import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="BikeDemand AI", page_icon="🚲", layout="wide")

# --- CSS DARK PREMIUM ---
st.markdown("""
<style>
    .stApp { background-color: #050505; }
    .metric-card {
        background: rgba(30, 30, 30, 0.6);
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        text-align: center;
    }
    /* Quitar puntos de radio buttons */
    [data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display: none; }
    [data-testid="stSidebar"] [role="radiogroup"] > label {
        padding: 12px 15px; border-radius: 8px; background: #111;
        margin-bottom: 5px; border: 1px solid #222; transition: 0.3s;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:hover { background: #222; }
    /* Botones */
    div.stButton > button {
        background: linear-gradient(90deg, #3a1c71, #d76d77, #ffaf7b);
        color: white; border: none; border-radius: 8px; font-weight: bold; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- CARGA DE DATOS ---
@st.cache_resource
def load_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    model = joblib.load(BASE_DIR / "models" / "xgboost_model.pkl")
    df = pd.read_csv(BASE_DIR / "data" / "bike_clean.csv")
    return model, df

model, df = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🚲 BikeDemand AI")
    page = st.radio("NAVEGACIÓN", ["🏠 Inicio", "🔮 Predicción", "📊 Dashboard"])

# --- LÓGICA ---
if page == "🏠 Inicio":
    st.title("Optimización de Bicicletas 🚲")
    st.write("Bienvenido al sistema de predicción basado en IA. Usa el menú izquierdo.")

elif page == "🔮 Predicción":
    st.title("🔮 Predicción con XGBoost")
    
    weathersit_map = {
        1: "☀️ Despejado",
        2: "⛅ Niebla",
        3: "🌧️ Lluvia ligera",
        4: "⛈️ Condiciones extremas"
    }

    col1, col2 = st.columns(2)
    with col1:
        season_map = {1: "☀️ Invierno", 2: "🌱 Primavera", 3: "🔥 Verano", 4: "🍂 Otoño"}
        season = st.selectbox("Estación", [1,2,3,4], format_func=lambda x: season_map[x])
        hr = st.slider("Hora", 0, 23, 12)
        workingday = st.selectbox("¿Día laboral?", [0, 1], format_func=lambda x: "Sí" if x else "No")
        # Aquí hemos reinsertado el clima que faltaba
        weathersit = st.selectbox("Clima", list(weathersit_map.keys()), format_func=lambda x: weathersit_map[x])
    
    with col2:
        temp_c = st.slider("Temperatura (°C)", -8.0, 39.0, 20.0)
        hum_pct = st.slider("Humedad (%)", 0, 100, 50)
        wind_kmh = st.slider("Velocidad Viento (km/h)", 0, 67, 15)
        
        temp_norm = temp_c / 41.0
        hum_norm = hum_pct / 100.0
        wind_norm = wind_kmh / 67.0

    if st.button("🚀 Ejecutar Predicción"):
        input_data = pd.DataFrame([[season, 0, 1, hr, 0, 0, workingday, weathersit, temp_norm, temp_norm, hum_norm, wind_norm]],
                                  columns=["season","yr","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed"])
        pred = int(model.predict(input_data)[0])
        st.markdown(f'<div class="metric-card"><h3>Demanda estimada ({temp_c}°C)</h3><h1 style="font-size: 50px; color: #ffaf7b;">{pred}</h1><p>bicicletas requeridas</p></div>', unsafe_allow_html=True)

elif page == "📊 Dashboard":
    st.title("📊 Análisis de Datos")
    k1, k2, k3 = st.columns(3)
    k1.markdown(f'<div class="metric-card"><h3>Total Viajes</h3><h2>{df["cnt"].sum():,.0f}</h2></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="metric-card"><h3>Media/Hora</h3><h2>{df["cnt"].mean():.1f}</h2></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="metric-card"><h3>Hora Pico</h3><h2>{df.groupby("hr")["cnt"].mean().idxmax()}:00</h2></div>', unsafe_allow_html=True)
    
    st.write("##")
    df_plot = df.groupby("hr")["cnt"].mean().reset_index()
    df_plot.columns = ["hr", "Bicicletas"]
    fig = px.bar(df_plot, x="hr", y="Bicicletas", template="plotly_dark")
    fig.update_traces(marker_color='#ffaf7b')
    st.plotly_chart(fig, use_container_width=True)