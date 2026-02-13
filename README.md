# Riesgo de Inundacion por Parroquia (Guayas)

Proyecto de clasificacion supervisada + aplicacion web Flask para estimar riesgo de inundacion por parroquia usando solo datos reales oficiales.

## 1) Estructura final del repositorio

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
├── deploy/
│   └── pythonanywhere_wsgi.py
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

## 2) Fuentes oficiales (sin sinteticos)

- INEC DPA parroquial (ArcGIS):
  - https://services7.arcgis.com/iFGeGXTAJXnjq0YN/ArcGIS/rest/services/Parroquias_del_Ecuador/FeatureServer/0
- INEC Censo 2022 (MANLOC CSV):
  - https://www.ecuadorencifras.gob.ec/documentos/web-inec/bd-censo/manzana/BDD_CPV2022_MANLOC_CSV.zip
- Historico base para etiqueta supervisada (eventos):
  - `data/raw/dataset_proyecto.csv` (columna `inundacion`)

Notas:
- El ZIP oficial pesado no se versiona en GitHub (`.gitignore`), se descarga automaticamente cuando falta.
- La metodologia y trazabilidad quedan en `outputs/fuentes_y_metodologia_oficial.json`.

## 3) Flujo correcto del proyecto

### 3.1 Crear entorno e instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2 Entrenar y regenerar artefactos

```bash
python3 ml/train_and_prepare.py
```

Genera:
- `outputs/resumen_metricas_modelos.csv`
- `outputs/predicciones_parroquias.csv`
- `outputs/modelo_entrenado.joblib`
- `outputs/dataset_guayas_oficial_completo.csv`
- `outputs/parroquias_guayas_sin_historial_predichas.csv`
- `outputs/fuentes_y_metodologia_oficial.json`
- `app/data/predicciones_parroquias.csv`
- `app/data/parroquias_riesgo.geojson`

### 3.3 Ejecutar app local

```bash
flask --app app.app run --host=0.0.0.0 --port=5000
```

Abrir:
- http://127.0.0.1:5000

## 4) Validacion antes de subir a GitHub

Ejecutar:

```bash
python3 scripts/preflight_check.py
```

Este chequeo valida:
- no haya archivos >95 MB (riesgo de rechazo en GitHub),
- dependencias criticas en `requirements.txt`,
- predicciones sin faltantes de riesgo/probabilidad,
- GeoJSON consistente para Guayas,
- notebook JSON valido.

## 5) Modelos implementados

- Regresion Logistica (base)
- Arbol de Decision
- Ensamble (RL + DT + RF)
- Arbol de Decision optimizado con `GridSearchCV`

Metricas reportadas:
- Precision
- Recall (prioritaria por gestion de riesgo)
- F1-score
- ROC-AUC

## 6) Notebook tecnico (Colab/Jupyter)

Archivo:
- `notebooks/Proyecto_Riesgo_Inundacion.ipynb`

Cobertura del notebook:
- limpieza de datos,
- construccion de variable derivada,
- definicion y justificacion de etiqueta objetivo,
- entrenamiento de RL/DT/ensamble,
- optimizacion con GridSearchCV,
- comparacion de metricas y conclusiones.

## 7) Deploy en PythonAnywhere (desde GitHub)

1. Crear web app en PythonAnywhere (Flask, Python 3.11).
2. Clonar repositorio en `/home/<usuario>/<repo>`.
3. Crear venv e instalar:
   ```bash
   mkvirtualenv --python=/usr/bin/python3.11 riesgo-inundacion
   pip install -r /home/<usuario>/<repo>/requirements.txt
   ```
4. En consola, generar artefactos si hace falta:
   ```bash
   cd /home/<usuario>/<repo>
   python ml/train_and_prepare.py
   ```
5. Editar archivo WSGI de PythonAnywhere usando `deploy/pythonanywhere_wsgi.py` como plantilla (o importando `wsgi.py` de raiz).
6. Recargar la app en el panel de PythonAnywhere.

Si no quieres entrenar en PythonAnywhere, sube el repo ya con `app/data/predicciones_parroquias.csv` y `app/data/parroquias_riesgo.geojson` actualizados.

## 8) Web app (funcionalidad requerida)

- Mapa interactivo con Leaflet + Esri Leaflet.
- Hover: parroquia, canton, provincia.
- Popup: riesgo y probabilidad.
- Colores por riesgo para Guayas.
- Parroquias fuera de Guayas en gris oscuro de contexto.
- Leyenda e indicadores de resumen.
