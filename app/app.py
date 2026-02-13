from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

GEOJSON_PATH = DATA_DIR / "parroquias_riesgo.geojson"
PRED_PATH = DATA_DIR / "predicciones_parroquias.csv"
METRICS_PATH = OUTPUT_DIR / "resumen_metricas_modelos.csv"
ARCGIS_PARROQUIAS_URL = (
    "https://services7.arcgis.com/iFGeGXTAJXnjq0YN/ArcGIS/rest/services/"
    "Parroquias_del_Ecuador/FeatureServer/0"
)


def load_geojson() -> dict:
    if not GEOJSON_PATH.exists():
        return {"type": "FeatureCollection", "features": []}
    return json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))


def load_predictions() -> pd.DataFrame:
    if not PRED_PATH.exists():
        return pd.DataFrame(
            columns=[
                "codigo",
                "provincia",
                "canton",
                "parroquia",
                "latitud",
                "longitud",
                "superficie_km2",
                "anio",
                "mes",
                "probabilidad_inundacion",
                "prediccion_inundacion",
                "riesgo_categoria",
            ]
        )
    return pd.read_csv(PRED_PATH)


def load_metrics() -> list[dict]:
    if not METRICS_PATH.exists():
        return []

    df_metrics = pd.read_csv(METRICS_PATH).copy()
    for col in ["precision", "recall", "f1", "roc_auc"]:
        if col in df_metrics.columns:
            df_metrics[col] = df_metrics[col].astype(float).round(4)
    return df_metrics.to_dict(orient="records")


def summary_stats(df_pred: pd.DataFrame) -> dict:
    if df_pred.empty:
        return {
            "total_parroquias": 0,
            "alto": 0,
            "medio": 0,
            "bajo": 0,
            "periodo": "Sin datos",
            "latitud_centro": -2.17,
            "longitud_centro": -79.9,
            "probabilidad_promedio": 0.0,
        }

    riesgo = df_pred["riesgo_categoria"].fillna("Bajo")
    periodo_df = df_pred[["anio", "mes"]].dropna()
    if periodo_df.empty:
        periodo_text = "Sin datos"
    else:
        latest_period = (
            periodo_df.assign(period_key=periodo_df["anio"].astype(int) * 100 + periodo_df["mes"].astype(int))
            .sort_values("period_key")
            .iloc[-1]
        )
        anio = int(latest_period["anio"])
        mes = int(latest_period["mes"])
        periodo_text = f"{anio}-{mes:02d}"

    return {
        "total_parroquias": int(len(df_pred)),
        "alto": int((riesgo == "Alto").sum()),
        "medio": int((riesgo == "Medio").sum()),
        "bajo": int((riesgo == "Bajo").sum()),
        "periodo": periodo_text,
        "latitud_centro": float(df_pred["latitud"].mean()),
        "longitud_centro": float(df_pred["longitud"].mean()),
        "probabilidad_promedio": float(df_pred["probabilidad_inundacion"].mean()),
    }


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.route("/")
    def index():
        pred_df = load_predictions()
        metrics = load_metrics()
        summary = summary_stats(pred_df)
        predictions = pred_df.to_dict(orient="records")

        return render_template(
            "index.html",
            predictions=predictions,
            arcgis_parroquias_url=ARCGIS_PARROQUIAS_URL,
            metrics=metrics,
            summary=summary,
        )

    @app.route("/api/parroquias")
    def api_parroquias():
        return jsonify(load_geojson())

    @app.route("/api/metricas")
    def api_metricas():
        return jsonify(load_metrics())

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
