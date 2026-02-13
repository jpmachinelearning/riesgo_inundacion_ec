from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def check_exists(path: Path) -> None:
    if not path.exists():
        fail(f"No existe: {path}")


def check_large_files(limit_mb: int = 95) -> None:
    limit_bytes = limit_mb * 1024 * 1024
    excluded = {".git", ".venv", "venv", "__pycache__"}

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in excluded for part in path.parts):
            continue
        if path.stat().st_size > limit_bytes:
            fail(
                f"Archivo demasiado grande para GitHub (> {limit_mb} MB): {path} "
                f"({path.stat().st_size / (1024 * 1024):.1f} MB)"
            )


def check_predictions() -> None:
    pred_path = ROOT / "outputs" / "predicciones_parroquias.csv"
    check_exists(pred_path)

    pred_df = pd.read_csv(pred_path, dtype={"codigo": str})
    if pred_df.empty:
        fail("predicciones_parroquias.csv esta vacio.")

    pred_df["codigo"] = pred_df["codigo"].str.zfill(6)
    required_cols = {"codigo", "probabilidad_inundacion", "riesgo_categoria"}
    missing_cols = required_cols - set(pred_df.columns)
    if missing_cols:
        fail(f"Columnas faltantes en predicciones: {sorted(missing_cols)}")

    if pred_df["codigo"].nunique() != len(pred_df):
        fail("Hay codigos de parroquia duplicados en predicciones.")

    if pred_df["probabilidad_inundacion"].isna().any():
        fail("Existen probabilidades faltantes en predicciones.")

    if pred_df["riesgo_categoria"].isna().any():
        fail("Existen riesgos faltantes en predicciones.")

    print(
        "[OK] Predicciones:",
        f"{len(pred_df)} filas,",
        f"{pred_df['codigo'].nunique()} codigos unicos,",
        f"rango_prob=[{pred_df['probabilidad_inundacion'].min():.4f}, "
        f"{pred_df['probabilidad_inundacion'].max():.4f}]",
    )


def check_geojson() -> None:
    geo_path = ROOT / "app" / "data" / "parroquias_riesgo.geojson"
    check_exists(geo_path)

    payload = json.loads(geo_path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if not features:
        fail("GeoJSON sin features.")

    guayas = [
        f
        for f in features
        if str(f.get("properties", {}).get("provincia", "")).strip().upper() == "GUAYAS"
    ]
    if not guayas:
        fail("GeoJSON no contiene parroquias de Guayas.")

    missing_risk = sum(1 for f in guayas if not f.get("properties", {}).get("riesgo"))
    missing_prob = sum(
        1 for f in guayas if f.get("properties", {}).get("probabilidad") in (None, "", "NaN")
    )
    if missing_risk or missing_prob:
        fail(
            "GeoJSON de Guayas tiene faltantes: "
            f"riesgo={missing_risk}, probabilidad={missing_prob}"
        )

    print(f"[OK] GeoJSON: {len(features)} features totales, {len(guayas)} de Guayas.")


def check_requirements() -> None:
    req_path = ROOT / "requirements.txt"
    check_exists(req_path)
    requirements = req_path.read_text(encoding="utf-8").lower()

    needed = ["flask", "pandas", "numpy", "scikit-learn", "joblib", "requests", "gunicorn"]
    missing = [name for name in needed if name not in requirements]
    if missing:
        fail(f"requirements.txt incompleto. Faltan: {missing}")

    print("[OK] requirements.txt contiene dependencias criticas.")


def check_notebook() -> None:
    nb_path = ROOT / "notebooks" / "Proyecto_Riesgo_Inundacion.ipynb"
    check_exists(nb_path)

    try:
        payload = json.loads(nb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"Notebook invalido: {exc}")

    cells = payload.get("cells", [])
    if len(cells) < 10:
        warn("Notebook muy corto; revisa detalle metodologico.")

    print(f"[OK] Notebook valido con {len(cells)} celdas.")


def main() -> None:
    print(f"Ejecutando preflight en: {ROOT}")
    check_large_files()
    check_requirements()
    check_predictions()
    check_geojson()
    check_notebook()
    print("[OK] Preflight completado. Proyecto listo para GitHub + despliegue.")


if __name__ == "__main__":
    main()
