# 🚲 BikeDemand AI: Sistema Inteligente de Predicción de Demanda

Este proyecto es una solución integral de Machine Learning diseñada para predecir la demanda de bicicletas compartidas.

[Dataset de bicicletas](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

## 🚀 Descripción del Proyecto
El objetivo es estimar el flujo de usuarios basándose en variables meteorológicas, temporales y de calendario, optimizando la logística de las estaciones.

## 🛠️ Tecnologías Utilizadas
* **Lenguaje**: Python
* **Ciencia de Datos**: Pandas, NumPy, Scikit-Learn
* **Modelado**: XGBoost, Regresión Lineal, Ridge, Lasso
* **Despliegue**: Streamlit

## 📊 Metodología
1. **Análisis Exploratorio (EDA)**: Identificación de patrones y tratamiento de outliers.
2. **Entrenamiento**: Comparación de algoritmos usando *pipelines* para garantizar reproducibilidad.
3. **App de Predicción**: Interfaz en tiempo real integrada con el modelo.

## ⚙️ Cómo ejecutar la aplicación
# 1. Descargar dataset
    Es necesario crear la carpeta data en local 
    En el enlace de arriba es necesario descargar los archivos csv 
    El archivo hours.csv es necesario renombrarlo a bike_sharing_hour y introducirlo en la carpeta data 
# 2. Crear el entorno vitual
    python -m venv venv
# En Windows:
    venv\Scripts\activate
# En macOS/Linux:
    source venv/bin/activate
# 3. Instalar dependecias
    pip install -r requirements.txt
# 4. Ejecutar
    Sera necesario ejecutar el archivo de 01_limpieza_eda
        al ejecutarlo se generara el archivo bike_clean 
    Luego ejecutaras el archivo 02_modelo_regresion
        al ejecutarlo se genera la carpeta de models con los distintos modelos necesarios

# 5. Ejecutar la aplicacion
    streamlit run app.py



## 📁 Estructura del Proyecto
```text
.
├── data/               # Dataset original y procesado (bike_clean.csv)
├── models/             # Modelos entrenados (Ridge, Lasso, XGBoost, etc.)
├── notebooks/          # Notebooks de Jupyter con el flujo completo
│   ├── 01_limpieza_eda.ipynb
│   └── 02_modelo_regresion.ipynb
├── app.py              # Aplicación web interactiva con Streamlit
└── README.md           # Documentación del proyecto
