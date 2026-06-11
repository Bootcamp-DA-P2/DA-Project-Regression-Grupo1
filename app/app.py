import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Bike Demand App", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
DATA_PATH = BASE_DIR / "data" / "bike_clean.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# navegación estable
if "page" not in st.session_state:
    st.session_state.page = "🔮 Predicción"

page = st.sidebar.radio(
    "🚲 Navegación",
    ["🔮 Predicción", "📊 Dashboard"]
)

st.session_state.page = page

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    font-weight: 600;
}

div[data-testid="stMetric"] {
    background-color: #f6f8fa;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

season_map = {1:"Invierno", 2:"Primavera", 3:"Verano", 4:"Otoño"}

season_month_map = {
    1: [12,1,2],
    2: [3,4,5],
    3: [6,7,8],
    4: [9,10,11]
}

weathersit_map = {
    1:"☀️ Despejado",
    2:"⛅ Niebla",
    3:"🌧️ Lluvia ligera",
    4:"⛈️ Condiciones extremas"
}

if page == "🔮 Predicción":

    st.title("🚲 Predicción de demanda")

    st.markdown("Introduce las condiciones para estimar la demanda de bicicletas.")

    col1, col2, col3 = st.columns(3)

    with col1:
        season = st.selectbox("Estación", list(season_map.keys()),
                              format_func=lambda x: season_map[x])

        yr = st.selectbox("Año", [0,1], format_func=lambda x: "2011" if x==0 else "2012")

        mnth = st.selectbox("Mes", season_month_map[season])

    with col2:
        hr = st.slider("Hora", 0, 23)
        weekday = st.slider("Día semana", 0, 6)

        workingday = st.selectbox("Día laboral", [0,1], format_func=lambda x: "Sí" if x else "No")

    with col3:
        holiday = st.selectbox("Festivo", [0,1], format_func=lambda x: "Sí" if x else "No")

        weathersit = st.selectbox(
            "Clima",
            list(weathersit_map.keys()),
            format_func=lambda x: weathersit_map[x]
        )

    st.divider()

    st.subheader("🌡️ Condiciones meteorológicas")

    col4, col5, col6 = st.columns(3)

    with col4:
        temp = st.slider("Temperatura (normalizada)", 0.0, 1.0, 0.5)

    with col5:
        atemp = st.slider("Sensación térmica", 0.0, 1.0, 0.5)

    with col6:
        hum = st.slider("Humedad", 0.0, 1.0, 0.5)

    windspeed = st.slider("Viento", 0.0, 1.0, 0.5)

    input_data = pd.DataFrame([[season, yr, mnth, hr,
                                holiday, weekday, workingday,
                                weathersit, temp, atemp,
                                hum, windspeed]],
                              columns=[
        "season","yr","mnth","hr",
        "holiday","weekday","workingday",
        "weathersit","temp","atemp",
        "hum","windspeed"
    ])

    if st.button("🚀 Predecir", use_container_width=True):
        prediction = model.predict(input_data)[0]

        colA, colB = st.columns(2)

        with colA:
            st.metric("🚲 Bicicletas estimadas", int(prediction))

        with colB:
            st.success("Predicción realizada correctamente")

elif page == "📊 Dashboard":

    st.title("📊 Análisis de datos")

    st.markdown("Resumen general del dataset de alquiler de bicicletas.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Distribución de demanda")
        st.bar_chart(df["cnt"].value_counts().sort_index())

    with col2:
        st.subheader("⏰ Demanda media por hora")
        st.bar_chart(df.groupby("hr")["cnt"].mean())

    st.divider()

    st.subheader("📋 Estadísticas generales")
    st.dataframe(df.describe(), use_container_width=True)