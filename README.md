# 🚲 BikeDemand AI: Sistema Inteligente de Predicción de Demanda

Proyecto de Machine Learning orientado a la predicción de la demanda de bicicletas compartidas mediante variables temporales, meteorológicas y de calendario.

📊 **Dataset utilizado:** [Bike Sharing Dataset (UCI)](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

---

# 🚀 Descripción del Proyecto

BikeDemand AI es una solución de análisis predictivo desarrollada para estimar la demanda de bicicletas compartidas en función de las condiciones ambientales y temporales.

La predicción se realiza utilizando información como:

* 🌡️ Temperatura
* 💧 Humedad
* 💨 Velocidad del viento
* 🌦️ Condiciones meteorológicas
* ⏰ Hora del día
* 📅 Mes y estación del año
* 🎉 Días festivos
* 💼 Días laborables

El objetivo es anticipar la demanda de bicicletas y comprender los patrones de movilidad urbana mediante técnicas de Machine Learning.

---

# 🌍 Contexto del Dataset

El conjunto de datos utilizado procede del sistema público de bicicletas compartidas **Capital Bikeshare** de Washington D.C. (Estados Unidos).

Los registros abarcan los años:

* 📅 2011
* 📅 2012

La información fue agregada a nivel horario e incluye variables meteorológicas y estacionales que permiten analizar cómo influyen las condiciones del entorno en el uso de bicicletas.

Los sistemas de bike sharing constituyen una importante fuente de información para estudiar patrones de movilidad urbana, ya que registran de forma detallada cada alquiler realizado.

## Volumen de datos

| Archivo  | Descripción              | Registros |
| -------- | ------------------------ | --------- |
| hour.csv | Datos agregados por hora | 17.379    |
| day.csv  | Datos agregados por día  | 731       |

Para este proyecto se ha utilizado la versión horaria (**hour.csv**) debido a su mayor nivel de detalle.

---

# 🎯 Objetivos

* Analizar los factores que afectan a la demanda de bicicletas.
* Construir modelos predictivos de regresión.
* Comparar distintos algoritmos de Machine Learning.
* Seleccionar el modelo con mejor capacidad predictiva.
* Desarrollar una aplicación interactiva para realizar predicciones en tiempo real.
* Crear un dashboard para explorar los principales patrones de demanda.

---

# 🛠️ Tecnologías Utilizadas

| Categoría         | Tecnologías                                             |
| ----------------- | ------------------------------------------------------- |
| Lenguaje          | Python                                                  |
| Análisis de Datos | Pandas, NumPy                                           |
| Visualización     | Matplotlib, Seaborn, Plotly                             |
| Machine Learning  | Scikit-Learn, XGBoost                                   |
| Modelos Evaluados | Linear Regression, Ridge, Lasso, Decision Tree, XGBoost |
| Despliegue        | Streamlit                                               |

---

# 📊 Variables Utilizadas

| Variable   | Descripción                           |
| ---------- | ------------------------------------- |
| season     | Estación del año                      |
| yr         | Año                                   |
| mnth       | Mes                                   |
| hr         | Hora                                  |
| holiday    | Día festivo                           |
| weekday    | Día de la semana                      |
| workingday | Día laborable                         |
| weathersit | Situación meteorológica               |
| temp       | Temperatura normalizada               |
| atemp      | Sensación térmica normalizada         |
| hum        | Humedad normalizada                   |
| windspeed  | Velocidad del viento normalizada      |
| cnt        | Número total de bicicletas alquiladas |

### Variable objetivo

```text
cnt = número total de bicicletas alquiladas
```

---

# 📈 Metodología

## 1. Limpieza y Preparación de Datos

Se realizaron tareas de:

* Revisión de calidad de datos.
* Eliminación de variables no utilizadas.
* Tratamiento y transformación de variables.
* Preparación para modelado.

Resultado generado:

```text
data/bike_clean.csv
```

---

## 2. Análisis Exploratorio de Datos (EDA)

Durante el análisis se estudiaron:

* Distribución de la demanda.
* Patrones horarios.
* Influencia de estaciones y meses.
* Impacto de las condiciones meteorológicas.
* Correlaciones entre variables.
* Evolución temporal de la demanda.

---

## 3. Entrenamiento de Modelos

Se entrenaron distintos modelos de regresión utilizando pipelines para garantizar la reproducibilidad del proceso.

Modelos evaluados:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Tree Regressor
* XGBoost Regressor

---

## 4. Evaluación

Métricas utilizadas:

* MSE (Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score
* Overfitting (Train R² − Test R²)

### Resultados

| Modelo            | MSE      | MAE   | R²    | Overfitting |
| ----------------- | -------- | ----- | ----- | ----------- |
| Linear Regression | 10131.41 | 74.42 | 0.670 | 0.019       |
| Ridge Regression  | 10128.22 | 74.41 | 0.670 | 0.019       |
| Lasso Regression  | 10124.65 | 74.31 | 0.670 | 0.019       |
| Decision Tree     | 8915.72  | 69.57 | 0.710 | 0.040       |
| XGBoost           | 2220.47  | 31.44 | 0.928 | 0.027       |

---

# 🏆 Modelo Seleccionado

## XGBoost Regressor

Fue seleccionado por ofrecer el mejor equilibrio entre precisión y capacidad de generalización.

### Resultados finales

| Métrica | Valor            |
| ------- | ---------------- |
| R²      | 0.928            |
| MAE     | 31.44 bicicletas |

### Conclusiones

El análisis de importancia de variables mostró que:

🚲 **La hora del día es el factor más influyente en la demanda de bicicletas**, reflejando claramente los patrones de movilidad asociados a horarios laborales y desplazamientos urbanos.

---

# 📱 Aplicación Streamlit

La aplicación web permite interactuar con el modelo de forma sencilla mediante tres secciones principales:

## 🏠 Home

* Descripción del proyecto.
* Información sobre el modelo seleccionado.
* Métricas principales de rendimiento.
* Resumen funcional de la aplicación.

## 🔮 Predicción

Permite estimar la demanda introduciendo:

* Estación.
* Hora.
* Día laborable o festivo.
* Condiciones meteorológicas.
* Temperatura.
* Humedad.
* Velocidad del viento.

## 📊 Dashboard

Incluye:

* KPIs principales.
* Total de bicicletas alquiladas.
* Demanda media.
* Hora pico.
* Demanda media por hora.
* Demanda por condición meteorológica.
* Filtros interactivos.

---

# ⚙️ Cómo ejecutar el proyecto

## 1. Descargar el Dataset

Crear la carpeta:

```text
data/
```

Descargar el dataset desde:

[Bike Sharing Dataset (UCI)](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

Utilizar el archivo:

```text
hour.csv
```

Renombrarlo como:

```text
bike_sharing_hour.csv
```

Y colocarlo dentro de:

```text
data/
```

---

## 2. Crear un entorno virtual

```bash
python -m venv venv
```

### Activación

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 4. Generar el dataset procesado

Ejecutar:

```text
01_limpieza_eda.ipynb
```

Se generará:

```text
data/bike_clean.csv
```

---

## 5. Entrenar los modelos

Ejecutar:

```text
02_modelo_regresion.ipynb
```

Se generará automáticamente:

```text
models/
```

Con los modelos serializados:

```text
linear_model.pkl
ridge_model.pkl
lasso_model.pkl
decision_tree_model.pkl
xgboost_model.pkl
```

---

## 6. Ejecutar la aplicación

```bash
streamlit run app/app.py
```

---

# 📁 Estructura del Proyecto

```text
.
├── app/
│   └── app.py
│
├── data/
│   ├── bike_sharing_hour.csv
│   └── bike_clean.csv
│
├── models/
│   ├── linear_model.pkl
│   ├── ridge_model.pkl
│   ├── lasso_model.pkl
│   ├── decision_tree_model.pkl
│   └── xgboost_model.pkl
│
├── notebooks/
│   ├── 01_limpieza_eda.ipynb
│   └── 02_modelo_regresion.ipynb
│
├── requirements.txt
└── README.md
```

---

# 👥 Autores

* Alejandra Duque García
* José Carlos de Santiago Sánchez
* María Zorayda Bejarano Jilon
* Elena Suárez Serrano



