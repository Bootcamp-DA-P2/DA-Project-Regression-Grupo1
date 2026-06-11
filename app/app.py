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
    st.title("🚲 Bike Demand AI")
    page = st.radio("NAVEGACIÓN", ["🏠 Inicio", "🔮 Predicción", "📊 Dashboard"])

# --- LÓGICA ---
if page == "🏠 Inicio":
    st.title("🚲 Bike Demand AI")

    st.markdown("""
    Bienvenido al sistema de predicción de demanda de bicicletas mediante Machine Learning.

    Esta aplicación estima cuántas bicicletas se alquilarán en un momento concreto
    en función de variables temporales y meteorológicas.
    """)
    col1, col2, col3 = st.columns(3)

    col1.metric("Modelo seleccionado", "XGBoost")
    col2.metric("R² del modelo", "0.928")
    col3.metric("MAE", "≈ 31 bicicletas")

    results = pd.DataFrame({
    "Modelo": ["Linear", "Ridge", "Lasso", "Decision Tree", "XGBoost"],
    "MSE": [10131.41, 10128.22, 10124.64, 8915.71, 2220.47],
    "MAE": [74.42, 74.41, 74.31, 69.57, 31.44],
    "R2": [0.67, 0.67, 0.67, 0.71, 0.93],
    "Overfitting": [0.019, 0.019, 0.018, 0.039, 0.026]
    })

    st.subheader("📊 Comparación de modelos")

    st.dataframe(results, use_container_width=True)

    best_model = results.loc[results["R2"].idxmax(), "Modelo"]

    st.success(f"""
    🏆 Mejor modelo: {best_model}

    El modelo XGBoost es el que mejor generaliza los datos,
    con el menor error y el mayor R².
    """)

    st.info("""
    📌 Insight clave del modelo:

    La variable más influyente en la predicción es la **hora del día**,
    lo que indica un patrón claro de demanda ligado a horarios laborales y movilidad urbana.
    """)

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
        st.markdown(f'<div class="metric-card"><h3>Demanda estimada</h3><h1 style="font-size: 50px; color: #ffaf7b;">{pred}</h1><p>bicicletas requeridas</p></div>', unsafe_allow_html=True)

elif page == "📊 Dashboard":
    st.title("📊 Análisis de Datos")

    f1, f2 = st.columns(2)

    with f1:
        year_filter = st.selectbox(
            "📅 Año",
            ["Todos", "2011", "2012"]
        )

    with f2:
        day_filter = st.selectbox(
            "📆 Tipo de día",
            ["Todos", "Laborable", "No laborable"]
        )
    
    # =========================
    # FILTROS
    # =========================
    df_filtrado = df.copy()

    if year_filter == "2011":
        df_filtrado = df_filtrado[df_filtrado["yr"] == 0]

    elif year_filter == "2012":
        df_filtrado = df_filtrado[df_filtrado["yr"] == 1]

    if day_filter == "Laborable":
        df_filtrado = df_filtrado[df_filtrado["workingday"] == 1]

    elif day_filter == "No laborable":
        df_filtrado = df_filtrado[df_filtrado["workingday"] == 0]

    # =========================
    # KPIs
    # =========================
    k1, k2, k3 = st.columns(3)

    k1.markdown(
        f"""
        <div class="metric-card">
            <h4>🚲 Total de bicicletas alquiladas</h4>
            <h2>{df_filtrado["cnt"].sum():,.0f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    k2.markdown(
        f"""
        <div class="metric-card">
            <h4>📊 Demanda media</h4>
            <h2>{df_filtrado["cnt"].mean():.1f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    k3.markdown(
        f"""
        <div class="metric-card">
            <h4>⏰ Hora pico</h4>
            <h2>{df_filtrado.groupby("hr")["cnt"].mean().idxmax()}:00</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("##")

    # =========================
    # GRÁFICAS
    # =========================
    col1, col2 = st.columns(2)

    # ---- Demanda por hora ----
    with col1:

        df_hour = (
            df_filtrado.groupby("hr")["cnt"]
            .mean()
            .reset_index()
        )

        df_hour.columns = ["Hora", "Bicicletas medias"]

        fig_hour = px.bar(
            df_hour,
            x="Hora",
            y="Bicicletas medias",
            title="🚲 Demanda media por hora",
            template="plotly_dark"
        )

        fig_hour.update_traces(marker_color="#4cc9f0")

        fig_hour.update_layout(
            title_x=0.5,
            xaxis_title="Hora del día",
            yaxis_title="Bicicletas",
            xaxis=dict(dtick=1)
        )

        st.plotly_chart(fig_hour, use_container_width=True)

    # ---- Demanda por clima ----
    with col2:

        weather_names = {
            1: "Despejado",
            2: "Niebla/Nubes",
            3: "Lluvia ligera",
            4: "Extremo"
        }

        df_weather = (
            df_filtrado.groupby("weathersit")["cnt"]
            .mean()
            .reset_index()
        )

        df_weather["weathersit"] = df_weather["weathersit"].map(weather_names)

        fig_weather = px.bar(
            df_weather,
            x="weathersit",
            y="cnt",
            title="🌦️ Demanda media según clima",
            template="plotly_dark"
        )

        fig_weather.update_traces(marker_color="#ffaf7b")

        fig_weather.update_layout(
            title_x=0.5,
            xaxis_title="Condición meteorológica",
            yaxis_title="Bicicletas"
        )

        st.plotly_chart(fig_weather, use_container_width=True)

    # =========================
    # INSIGHT
    # =========================
    st.info(
        f"""
        📌 **Insight principal**

        La mayor demanda se registra alrededor de las **{df.groupby("hr")["cnt"].mean().idxmax()}:00 horas**.
        Además, las condiciones meteorológicas favorables incrementan significativamente el uso del servicio.
        """
    )