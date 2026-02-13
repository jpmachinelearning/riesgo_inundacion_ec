# Riesgo de Inundacion por Parroquia (Guayas)

Proyecto de clasificacion supervisada para estimar riesgo de inundacion por parroquia, con datos reales oficiales de Ecuador e integracion en una aplicacion web Flask con visualizacion geoespacial.

## Objetivo del proyecto

Construir y evaluar modelos de clasificacion supervisada que asignen categoria de riesgo de inundacion por parroquia, cumpliendo:

- uso exclusivo de datos reales oficiales,
- variable objetivo construida tecnicamente (sin etiquetas predefinidas),
- comparacion de modelos base vs optimizados,
- exportacion de resultados para visualizacion geoespacial.

## Fuentes oficiales utilizadas

- INEC DPA parroquial (ArcGIS):
  - https://services7.arcgis.com/iFGeGXTAJXnjq0YN/ArcGIS/rest/services/Parroquias_del_Ecuador/FeatureServer/0
- INEC Censo 2022 (MANLOC CSV):
  - https://www.ecuadorencifras.gob.ec/documentos/web-inec/bd-censo/manzana/BDD_CPV2022_MANLOC_CSV.zip
- Base historica para construccion de etiqueta supervisada:
  - `data/raw/dataset_proyecto.csv` (columna `inundacion`)

## Estructura del repositorio

```text
.
├── app/
│   ├── app.py
│   ├── wsgi.py
│   ├── data/
│   │   ├── parroquias_riesgo.geojson
│   │   └── predicciones_parroquias.csv
│   ├── static/
│   │   ├── css/styles.css
│   │   └── js/map-ui.js
│   └── templates/
│       └── index.html
├── data/
│   └── raw/
│       ├── dataset_proyecto.csv
│       └── official/
│           └── .gitkeep
├── ml/
│   └── train_and_prepare.py
├── notebooks/
│   └── Proyecto_Riesgo_Inundacion.ipynb
├── outputs/
│   ├── dataset_guayas_oficial_completo.csv
│   ├── fuentes_y_metodologia_oficial.json
│   ├── modelo_entrenado.joblib
│   ├── parroquias_guayas_sin_historial_predichas.csv
│   ├── predicciones_parroquias.csv
│   └── resumen_metricas_modelos.csv
├── scripts/
│   └── preflight_check.py
├── Procfile
├── requirements.txt
├── runtime.txt
├── wsgi.py
└── README.md
```

## Flujo de ejecucion

### 1) Instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Entrenar y generar artefactos

```bash
python3 ml/train_and_prepare.py
```

Salidas principales:

- `outputs/resumen_metricas_modelos.csv`
- `outputs/predicciones_parroquias.csv`
- `outputs/modelo_entrenado.joblib`
- `outputs/dataset_guayas_oficial_completo.csv`
- `outputs/parroquias_guayas_sin_historial_predichas.csv`
- `outputs/fuentes_y_metodologia_oficial.json`
- `app/data/predicciones_parroquias.csv`
- `app/data/parroquias_riesgo.geojson`

### 3) Ejecutar aplicacion web local

```bash
flask --app app.app run --host=0.0.0.0 --port=5000
```

URL local:
- http://127.0.0.1:5000

## Modelos implementados

- Regresion Logistica (base)
- Arbol de Decision
- Ensamble (RL + DT + RF)
- Arbol de Decision optimizado con `GridSearchCV`

Metricas reportadas:

- Precision
- Recall (prioritaria en gestion de riesgo)
- F1-score
- ROC-AUC

## Criterio de etiqueta objetivo

La etiqueta se construye por parroquia desde historial observado:

- `tasa_inundacion_historica = eventos_inundacion / total_periodos`
- umbral de alto riesgo: percentil 66 de la tasa historica

Metodologia y trazabilidad:

- `outputs/fuentes_y_metodologia_oficial.json`

## Notebook tecnico

Archivo:

- `notebooks/Proyecto_Riesgo_Inundacion.ipynb`

Incluye:

- limpieza y control de calidad de datos,
- construccion de variable objetivo y variable derivada,
- entrenamiento y optimizacion de modelos,
- comparacion de metricas y curvas ROC,
- verificacion de cobertura completa en Guayas,
- conclusiones tecnicas.

## Aplicacion web: funcionalidades

- mapa interactivo con Leaflet + Esri Leaflet,
- visualizacion de parroquias de Guayas por categoria de riesgo,
- hover con parroquia/canton/provincia,
- popup con riesgo y probabilidad,
- leyenda de simbologia,
- contexto visual fuera de Guayas en gris oscuro.

## Validacion rapida del proyecto

Ejecutar:

```bash
python3 scripts/preflight_check.py
```

Valida:

- integridad de archivos clave,
- consistencia de predicciones y GeoJSON,
- ausencia de probabilidades/riesgos faltantes en Guayas,
- coherencia basica del notebook.
