from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable
import time
import zipfile

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
APP_DATA_DIR = ROOT / "app" / "data"
RAW_DIR = ROOT / "data" / "raw"
OFFICIAL_DIR = RAW_DIR / "official"

PARROQUIAS_ARCGIS_URL = (
    "https://services7.arcgis.com/iFGeGXTAJXnjq0YN/ArcGIS/rest/services/"
    "Parroquias_del_Ecuador/FeatureServer/0/query"
)
CENSO_MANLOC_ZIP_URL = (
    "https://www.ecuadorencifras.gob.ec/documentos/web-inec/bd-censo/manzana/"
    "BDD_CPV2022_MANLOC_CSV.zip"
)
# Historical base used by the project team, consolidated from official sources
# (INAMHI + SNGRE) for supervised flood-event labels and climate observations.
HISTORICAL_LABELS_PATH = RAW_DIR / "dataset_proyecto.csv"
CENSO_MANLOC_ZIP_PATH = OFFICIAL_DIR / "BDD_CPV2022_MANLOC_CSV.zip"

# Official IGM terrain model for topographic features.
IGM_DTM_WMS_URL = "https://www.geoportaligm.gob.ec/dtm/ows"
IGM_ELEV_LAYER = "igm:elevacion50k"


def normalize_code(value: object) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    digits = digits.lstrip("0")
    return digits


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den_safe = den.replace({0: np.nan})
    out = num / den_safe
    return out.replace([np.inf, -np.inf], np.nan)


def weighted_mean(sum_weighted: pd.Series, sum_weights: pd.Series) -> pd.Series:
    return safe_div(sum_weighted, sum_weights)


def percentile_95_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 95))


def sanitize_terrain_value(value: object) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")

    # GeoServer rasters often return very large sentinel values for NoData.
    if (not np.isfinite(v)) or abs(v) >= 1e6 or v <= -1000:
        return float("nan")
    return v


def download_if_missing(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return

    with requests.get(url, stream=True, timeout=300, verify=False) as response:
        response.raise_for_status()
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def iter_arcgis_features(where_clause: str) -> Iterable[dict]:
    offset = 0
    while True:
        params = {
            "where": where_clause,
            "outFields": (
                "DPA_PARROQ,DPA_DESPAR,DPA_DESCAN,DPA_DESPRO,"
                "AREA_KM2,Shape__Length"
            ),
            "returnGeometry": "true",
            "f": "json",
            "outSR": "4326",
            "resultOffset": offset,
            "resultRecordCount": 2000,
        }
        response = requests.get(PARROQUIAS_ARCGIS_URL, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
        features = payload.get("features", [])
        if not features:
            break

        for feature in features:
            yield feature

        if len(features) < 2000:
            break
        offset += len(features)


def geometry_centroid(geometry: dict) -> tuple[float | None, float | None]:
    rings = geometry.get("rings") or []
    if not rings:
        return None, None

    xs: list[float] = []
    ys: list[float] = []
    for ring in rings:
        for point in ring:
            if len(point) >= 2:
                xs.append(float(point[0]))
                ys.append(float(point[1]))

    if not xs:
        return None, None
    return float(np.mean(ys)), float(np.mean(xs))


def fetch_official_guayas_parishes() -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    features_geojson: list[dict] = []

    for feature in iter_arcgis_features("DPA_DESPRO='GUAYAS'"):
        attrs = feature.get("attributes", {})
        geometry = feature.get("geometry", {})

        codigo_full = str(attrs.get("DPA_PARROQ", ""))
        codigo = normalize_code(codigo_full)
        lat, lon = geometry_centroid(geometry)

        row = {
            "codigo": codigo,
            "codigo_inec": codigo_full,
            "provincia": attrs.get("DPA_DESPRO"),
            "canton": attrs.get("DPA_DESCAN"),
            "parroquia": attrs.get("DPA_DESPAR"),
            "superficie_km2": pd.to_numeric(attrs.get("AREA_KM2"), errors="coerce"),
            "shape_length": pd.to_numeric(attrs.get("Shape__Length"), errors="coerce"),
            "latitud": lat,
            "longitud": lon,
            "anio": 2022,
            "mes": 12,
        }
        rows.append(row)

        features_geojson.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": geometry.get("rings", [])},
                "properties": {
                    "codigo": codigo,
                    "codigo_inec": codigo_full,
                    "parroquia": attrs.get("DPA_DESPAR"),
                    "canton": attrs.get("DPA_DESCAN"),
                    "provincia": attrs.get("DPA_DESPRO"),
                },
            }
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["codigo"]).sort_values("codigo").reset_index(drop=True)

    geojson = {"type": "FeatureCollection", "features": features_geojson}
    return df, geojson


def aggregate_pob_from_manloc(zip_path: Path) -> pd.DataFrame:
    total_w: dict[str, float] = {}
    sum_age_w: dict[str, float] = {}
    female_w: dict[str, float] = {}
    urban_w: dict[str, float] = {}

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("BDD_POB_CPV2022_MANLOC.csv") as f:
            chunks = pd.read_csv(
                f,
                sep=";",
                usecols=["PARROQ", "IMP_VOPA", "P03", "P02", "AUR"],
                dtype=str,
                chunksize=250_000,
                low_memory=False,
            )
            for chunk in chunks:
                chunk["codigo"] = chunk["PARROQ"].map(normalize_code)
                chunk = chunk[chunk["codigo"].str.startswith("9", na=False)]
                if chunk.empty:
                    continue

                chunk["w"] = pd.to_numeric(chunk["IMP_VOPA"], errors="coerce").fillna(1.0)
                chunk["edad"] = pd.to_numeric(chunk["P03"], errors="coerce")
                chunk["female"] = (chunk["P02"] == "2").astype(float)
                chunk["urban"] = (chunk["AUR"] == "1").astype(float)

                by = chunk.groupby("codigo", as_index=False).agg(
                    w_total=("w", "sum"),
                    edad_w=("edad", lambda s: np.nansum(s * chunk.loc[s.index, "w"])),
                    female_w=("female", lambda s: np.nansum(s * chunk.loc[s.index, "w"])),
                    urban_w=("urban", lambda s: np.nansum(s * chunk.loc[s.index, "w"])),
                )

                for _, r in by.iterrows():
                    c = r["codigo"]
                    total_w[c] = total_w.get(c, 0.0) + float(r["w_total"])
                    sum_age_w[c] = sum_age_w.get(c, 0.0) + float(r["edad_w"])
                    female_w[c] = female_w.get(c, 0.0) + float(r["female_w"])
                    urban_w[c] = urban_w.get(c, 0.0) + float(r["urban_w"])

    out = pd.DataFrame({"codigo": sorted(total_w.keys())})
    out["poblacion_2022"] = out["codigo"].map(total_w)
    out["edad_promedio_2022"] = weighted_mean(out["codigo"].map(sum_age_w), out["poblacion_2022"])
    out["pct_mujeres_2022"] = weighted_mean(out["codigo"].map(female_w), out["poblacion_2022"])
    out["pct_urbana_2022"] = weighted_mean(out["codigo"].map(urban_w), out["poblacion_2022"])
    return out


def aggregate_simple_weight(zip_path: Path, member_name: str, out_col: str) -> pd.DataFrame:
    totals: dict[str, float] = {}
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as f:
            chunks = pd.read_csv(
                f,
                sep=";",
                usecols=["PARROQ", "IMP_VOPA"],
                dtype=str,
                chunksize=250_000,
                low_memory=False,
            )
            for chunk in chunks:
                chunk["codigo"] = chunk["PARROQ"].map(normalize_code)
                chunk = chunk[chunk["codigo"].str.startswith("9", na=False)]
                if chunk.empty:
                    continue

                chunk["w"] = pd.to_numeric(chunk["IMP_VOPA"], errors="coerce").fillna(1.0)
                grouped = chunk.groupby("codigo", as_index=False)["w"].sum()
                for _, r in grouped.iterrows():
                    c = r["codigo"]
                    totals[c] = totals.get(c, 0.0) + float(r["w"])

    out = pd.DataFrame({"codigo": sorted(totals.keys())})
    out[out_col] = out["codigo"].map(totals)
    return out


def build_official_feature_table() -> tuple[pd.DataFrame, dict]:
    download_if_missing(CENSO_MANLOC_ZIP_URL, CENSO_MANLOC_ZIP_PATH)

    parishes_df, geojson = fetch_official_guayas_parishes()

    pob_df = aggregate_pob_from_manloc(CENSO_MANLOC_ZIP_PATH)
    hog_df = aggregate_simple_weight(CENSO_MANLOC_ZIP_PATH, "BDD_HOG_CPV2022_MANLOC.csv", "hogares_2022")
    viv_df = aggregate_simple_weight(CENSO_MANLOC_ZIP_PATH, "BDD_VIV_CPV2022_MANLOC.csv", "viviendas_2022")

    df = (
        parishes_df.merge(pob_df, on="codigo", how="left")
        .merge(hog_df, on="codigo", how="left")
        .merge(viv_df, on="codigo", how="left")
        .copy()
    )

    for col in [
        "superficie_km2",
        "shape_length",
        "latitud",
        "longitud",
        "poblacion_2022",
        "hogares_2022",
        "viviendas_2022",
        "edad_promedio_2022",
        "pct_mujeres_2022",
        "pct_urbana_2022",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["densidad_poblacional_2022"] = safe_div(df["poblacion_2022"], df["superficie_km2"])
    df["personas_por_hogar_2022"] = safe_div(df["poblacion_2022"], df["hogares_2022"])
    df["viviendas_por_km2_2022"] = safe_div(df["viviendas_2022"], df["superficie_km2"])

    perim_km = df["shape_length"] / 1000.0
    df["indice_compacidad"] = safe_div(4 * math.pi * df["superficie_km2"], perim_km**2)

    return df, geojson


def build_climate_features_from_historical(path: Path) -> pd.DataFrame:
    hist = pd.read_csv(path)
    hist = hist.rename(
        columns={
            "Codigo": "codigo",
            "Código": "codigo",
            "Precipitacion_Anual": "precipitacion_anual",
            "Cerca_Rio": "cerca_rio",
        }
    )

    if "codigo" not in hist.columns:
        raise ValueError("No existe columna codigo/Codigo en dataset_proyecto.csv")

    hist["codigo"] = hist["codigo"].map(normalize_code)

    numeric_cols = [
        "precipitacion_mm",
        "precipitacion_anual",
        "temp_media_c",
        "humedad_relativa",
        "cerca_rio",
    ]
    for col in numeric_cols:
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")

    climate = hist.groupby("codigo", as_index=False).agg(
        precipitacion_mensual_prom_mm=("precipitacion_mm", "mean"),
        precipitacion_mensual_p95_mm=("precipitacion_mm", percentile_95_or_nan),
        precipitacion_anual_prom_mm=("precipitacion_anual", "mean"),
        temperatura_media_prom_c=("temp_media_c", "mean"),
        humedad_relativa_prom=("humedad_relativa", "mean"),
        cerca_rio_prom=("cerca_rio", "mean"),
        periodos_climaticos_observados=("precipitacion_mm", "count"),
    )

    return climate


def _elev_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(float(lat), 4), round(float(lon), 4))


def query_igm_elevation(
    lat: float,
    lon: float,
    session: requests.Session,
    delta: float = 0.01,
    retries: int = 2,
) -> float:
    bbox = f"{lat - delta},{lon - delta},{lat + delta},{lon + delta}"
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": IGM_ELEV_LAYER,
        "QUERY_LAYERS": IGM_ELEV_LAYER,
        "CRS": "EPSG:4326",
        "BBOX": bbox,
        "WIDTH": 101,
        "HEIGHT": 101,
        "I": 50,
        "J": 50,
        "INFO_FORMAT": "application/json",
    }

    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            resp = session.get(IGM_DTM_WMS_URL, params=params, timeout=45)
            resp.raise_for_status()
            payload = resp.json()
            features = payload.get("features", [])
            if not features:
                return float("nan")
            value = features[0].get("properties", {}).get("GRAY_INDEX")
            return sanitize_terrain_value(value)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.35)

    if last_error:
        return float("nan")
    return float("nan")


def build_topographic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["codigo", "latitud", "longitud"]].copy()
    cache: dict[tuple[float, float], float] = {}
    session = requests.Session()
    step = 0.01

    def sample(lat: float, lon: float) -> float:
        if not np.isfinite(lat) or not np.isfinite(lon):
            return float("nan")
        key = _elev_key(lat, lon)
        if key in cache:
            return cache[key]
        value = query_igm_elevation(lat, lon, session=session)
        cache[key] = value
        return value

    altitudes: list[float] = []
    slopes: list[float] = []
    ranges: list[float] = []

    for row in out.itertuples(index=False):
        lat = float(row.latitud) if pd.notna(row.latitud) else float("nan")
        lon = float(row.longitud) if pd.notna(row.longitud) else float("nan")

        c = sample(lat, lon)
        n = sample(lat + step, lon)
        s = sample(lat - step, lon)
        e = sample(lat, lon + step)
        w = sample(lat, lon - step)

        altitudes.append(c)

        valid_values = [v for v in [c, n, s, e, w] if np.isfinite(v)]
        if len(valid_values) >= 2:
            ranges.append(float(max(valid_values) - min(valid_values)))
        else:
            ranges.append(float("nan"))

        dist_lat = step * 111_320.0
        dist_lon = step * 111_320.0 * max(math.cos(math.radians(lat if np.isfinite(lat) else 0.0)), 0.2)

        grad_ns = float("nan")
        grad_ew = float("nan")

        if np.isfinite(n) and np.isfinite(s):
            grad_ns = abs(n - s) / (2.0 * dist_lat)
        elif np.isfinite(c) and np.isfinite(n):
            grad_ns = abs(n - c) / dist_lat
        elif np.isfinite(c) and np.isfinite(s):
            grad_ns = abs(c - s) / dist_lat

        if np.isfinite(e) and np.isfinite(w):
            grad_ew = abs(e - w) / (2.0 * dist_lon)
        elif np.isfinite(c) and np.isfinite(e):
            grad_ew = abs(e - c) / dist_lon
        elif np.isfinite(c) and np.isfinite(w):
            grad_ew = abs(c - w) / dist_lon

        if np.isfinite(grad_ns) and np.isfinite(grad_ew):
            slopes.append(float((grad_ns**2 + grad_ew**2) ** 0.5))
        elif np.isfinite(grad_ns):
            slopes.append(float(grad_ns))
        elif np.isfinite(grad_ew):
            slopes.append(float(grad_ew))
        else:
            slopes.append(float("nan"))

    out["altitud_igm_m"] = altitudes
    out["pendiente_igm"] = slopes
    out["rango_altitud_igm_m"] = ranges

    return out[["codigo", "altitud_igm_m", "pendiente_igm", "rango_altitud_igm_m"]]


def spatial_impute_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

        mask_known = out[col].notna() & out["latitud"].notna() & out["longitud"].notna()
        mask_missing = out[col].isna() & out["latitud"].notna() & out["longitud"].notna()

        known = out.loc[mask_known, ["latitud", "longitud", col]].copy()
        missing = out.loc[mask_missing, ["latitud", "longitud"]].copy()

        if not known.empty and not missing.empty:
            n_neighbors = int(min(4, len(known)))
            knn = NearestNeighbors(n_neighbors=n_neighbors)
            knn.fit(known[["latitud", "longitud"]].to_numpy())
            dist, idx = knn.kneighbors(missing[["latitud", "longitud"]].to_numpy())

            weights = 1.0 / np.clip(dist, 1e-6, None)
            vals = known[col].to_numpy()[idx]
            imputed = (weights * vals).sum(axis=1) / weights.sum(axis=1)
            out.loc[missing.index, col] = imputed

        if out[col].isna().any():
            median_val = out[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            out[col] = out[col].fillna(float(median_val))

    return out


def build_historical_labels(path: Path) -> tuple[pd.DataFrame, float]:
    hist = pd.read_csv(path)
    code_col = "Código" if "Código" in hist.columns else "Codigo"
    if code_col not in hist.columns:
        raise ValueError("No existe columna Codigo/Código para construir etiquetas historicas")

    hist["codigo"] = hist[code_col].map(normalize_code)
    hist["inundacion"] = pd.to_numeric(hist["inundacion"], errors="coerce").fillna(0).astype(int)

    by_parish = hist.groupby("codigo", as_index=False).agg(
        eventos_inundacion=("inundacion", "sum"),
        total_periodos=("inundacion", "count"),
    )
    by_parish["tasa_inundacion_historica"] = safe_div(
        by_parish["eventos_inundacion"], by_parish["total_periodos"]
    ).fillna(0)

    threshold = float(by_parish["tasa_inundacion_historica"].quantile(0.66))
    by_parish["target_alto_riesgo"] = (
        by_parish["tasa_inundacion_historica"] >= threshold
    ).astype(int)

    return by_parish, threshold


def build_models(feature_cols: list[str]) -> tuple[Pipeline, Pipeline, VotingClassifier, GridSearchCV]:
    pre_scaled = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ]
    )

    pre_unscaled = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                feature_cols,
            )
        ]
    )

    lr = Pipeline(
        steps=[
            ("preprocess", pre_scaled),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )

    dt = Pipeline(
        steps=[
            ("preprocess", pre_unscaled),
            (
                "clf",
                DecisionTreeClassifier(random_state=42, class_weight="balanced", min_samples_leaf=2),
            ),
        ]
    )

    rf = Pipeline(
        steps=[
            ("preprocess", pre_unscaled),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    ensemble = VotingClassifier(estimators=[("lr", lr), ("dt", dt), ("rf", rf)], voting="soft")

    dt_grid = GridSearchCV(
        estimator=dt,
        param_grid={
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [3, 5, 8, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        },
        scoring="recall",
        cv=5,
        n_jobs=1,
        refit=True,
    )

    return lr, dt, ensemble, dt_grid


def evaluate_model(name: str, model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]
    return {
        "modelo": name,
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_score), 4),
    }


def classify_risk(prob: float, low_threshold: float, high_threshold: float) -> str:
    if prob >= high_threshold:
        return "Alto"
    if prob >= low_threshold:
        return "Medio"
    return "Bajo"


def export_geojson_with_predictions(base_geojson: dict, pred_df: pd.DataFrame, output_path: Path) -> None:
    pred_map = {
        str(row.codigo): {
            "riesgo": row.riesgo_categoria,
            "probabilidad": float(row.probabilidad_inundacion),
            "prediccion": int(row.prediccion_inundacion),
        }
        for row in pred_df.itertuples(index=False)
    }

    features_out = []
    for feat in base_geojson.get("features", []):
        props = feat.get("properties", {}).copy()
        code = str(props.get("codigo", ""))
        pred = pred_map.get(code)
        if pred:
            props.update(pred)

        features_out.append(
            {
                "type": "Feature",
                "geometry": feat.get("geometry"),
                "properties": props,
            }
        )

    geojson = {"type": "FeatureCollection", "features": features_out}
    output_path.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    official_df, official_geojson = build_official_feature_table()
    climate_df = build_climate_features_from_historical(HISTORICAL_LABELS_PATH)
    labels_df, rate_threshold = build_historical_labels(HISTORICAL_LABELS_PATH)

    dataset = (
        official_df.merge(climate_df, on="codigo", how="left")
        .merge(labels_df, on="codigo", how="left")
        .copy()
    )

    topography_df = build_topographic_features(dataset[["codigo", "latitud", "longitud"]])
    dataset = dataset.merge(topography_df, on="codigo", how="left")

    climate_cols = [
        "precipitacion_mensual_prom_mm",
        "precipitacion_mensual_p95_mm",
        "precipitacion_anual_prom_mm",
        "temperatura_media_prom_c",
        "humedad_relativa_prom",
        "cerca_rio_prom",
    ]
    topography_cols = ["altitud_igm_m", "pendiente_igm", "rango_altitud_igm_m"]

    dataset = spatial_impute_columns(dataset, climate_cols + topography_cols)

    labeled_df = dataset[dataset["target_alto_riesgo"].notna()].copy()
    unlabeled_df = dataset[dataset["target_alto_riesgo"].isna()].copy()

    feature_cols = [
        # Climatic features (INAMHI observations consolidated in project base)
        "precipitacion_mensual_prom_mm",
        "precipitacion_mensual_p95_mm",
        "precipitacion_anual_prom_mm",
        "temperatura_media_prom_c",
        "humedad_relativa_prom",
        "cerca_rio_prom",
        # Topographic features (IGM DTM)
        "altitud_igm_m",
        "pendiente_igm",
        "rango_altitud_igm_m",
        # Socio-territorial features (INEC Censo 2022 + DPA)
        "superficie_km2",
        "shape_length",
        "latitud",
        "longitud",
        "poblacion_2022",
        "hogares_2022",
        "viviendas_2022",
        "edad_promedio_2022",
        "pct_mujeres_2022",
        "pct_urbana_2022",
        "densidad_poblacional_2022",
        "personas_por_hogar_2022",
        "viviendas_por_km2_2022",
        "indice_compacidad",
    ]

    x = labeled_df[feature_cols]
    y = labeled_df["target_alto_riesgo"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    lr, dt, ensemble, dt_grid = build_models(feature_cols)

    lr.fit(x_train, y_train)
    dt.fit(x_train, y_train)
    ensemble.fit(x_train, y_train)
    dt_grid.fit(x_train, y_train)
    dt_opt = dt_grid.best_estimator_

    metrics = [
        evaluate_model("Regresion Logistica (Base)", lr, x_test, y_test),
        evaluate_model("Arbol de Decision", dt, x_test, y_test),
        evaluate_model("Ensamble RL+DT+RF", ensemble, x_test, y_test),
        evaluate_model("Arbol de Decision Optimizado (GridSearchCV)", dt_opt, x_test, y_test),
    ]
    metrics_df = pd.DataFrame(metrics).sort_values(["recall", "f1", "roc_auc"], ascending=False)
    metrics_df.to_csv(OUTPUT_DIR / "resumen_metricas_modelos.csv", index=False)

    best_name = metrics_df.iloc[0]["modelo"]
    if best_name == "Regresion Logistica (Base)":
        final_model = lr
    elif best_name == "Arbol de Decision":
        final_model = dt
    elif best_name == "Ensamble RL+DT+RF":
        final_model = ensemble
    else:
        final_model = dt_opt

    all_probs = final_model.predict_proba(dataset[feature_cols])[:, 1]
    dataset["probabilidad_inundacion"] = all_probs

    q_low = float(dataset["probabilidad_inundacion"].quantile(0.33))
    q_high = float(dataset["probabilidad_inundacion"].quantile(0.66))
    if q_low == q_high:
        q_low, q_high = 0.33, 0.66

    dataset["riesgo_categoria"] = dataset["probabilidad_inundacion"].apply(
        lambda p: classify_risk(float(p), q_low, q_high)
    )
    dataset["prediccion_inundacion"] = (dataset["riesgo_categoria"] == "Alto").astype(int)

    pred_export_cols = [
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

    pred_df = dataset[pred_export_cols].sort_values(["canton", "parroquia"]).reset_index(drop=True)

    pred_df.to_csv(OUTPUT_DIR / "predicciones_parroquias.csv", index=False)
    pred_df.to_csv(APP_DATA_DIR / "predicciones_parroquias.csv", index=False)

    export_geojson_with_predictions(official_geojson, pred_df, APP_DATA_DIR / "parroquias_riesgo.geojson")

    dataset.to_csv(OUTPUT_DIR / "dataset_guayas_oficial_completo.csv", index=False)
    unlabeled_df.merge(
        pred_df[["codigo", "probabilidad_inundacion", "riesgo_categoria"]],
        on="codigo",
        how="left",
    ).to_csv(OUTPUT_DIR / "parroquias_guayas_sin_historial_predichas.csv", index=False)

    source_registry = {
        "fuentes_oficiales": {
            "INEC_DPA_Parroquias_ArcGIS": PARROQUIAS_ARCGIS_URL,
            "INEC_Censo2022_MANLOC_CSV": CENSO_MANLOC_ZIP_URL,
            "IGM_DTM_WMS": IGM_DTM_WMS_URL,
            "INAMHI_SNGRE_base_historica_consolidada": str(HISTORICAL_LABELS_PATH),
        },
        "variables_integradas": {
            "climaticas": climate_cols,
            "topograficas": topography_cols,
            "socio_territoriales": [
                "poblacion_2022",
                "densidad_poblacional_2022",
                "pct_urbana_2022",
                "hogares_2022",
                "viviendas_2022",
                "indice_compacidad",
            ],
        },
        "criterio_etiqueta": {
            "variable": "tasa_inundacion_historica",
            "definicion": "eventos_inundacion/total_periodos por parroquia",
            "umbral_alto_riesgo": rate_threshold,
            "metodo_umbral": "percentil_66",
        },
        "resumen": {
            "parroquias_guayas_oficiales": int(len(dataset)),
            "parroquias_con_historial_para_entrenar": int(len(labeled_df)),
            "parroquias_sin_historial_predichas": int(len(unlabeled_df)),
            "modelo_ganador": best_name,
        },
    }
    (OUTPUT_DIR / "fuentes_y_metodologia_oficial.json").write_text(
        json.dumps(source_registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_bundle = {
        "modelo": final_model,
        "features": feature_cols,
        "best_params_dt_grid": dt_grid.best_params_,
        "threshold_event_rate": rate_threshold,
    }
    joblib.dump(model_bundle, OUTPUT_DIR / "modelo_entrenado.joblib")

    print("Pipeline oficial completado.")
    print(f"Parroquias oficiales de Guayas: {len(dataset)}")
    print(f"Parroquias con historial (train): {len(labeled_df)}")
    print(f"Parroquias faltantes predichas: {len(unlabeled_df)}")
    print(f"Modelo ganador: {best_name}")
    print(f"Umbral etiqueta (tasa inundacion): {rate_threshold:.4f}")
    print(f"Umbrales riesgo probabilidad: bajo<{q_low:.4f}, medio<{q_high:.4f}, alto>={q_high:.4f}")


if __name__ == "__main__":
    main()
