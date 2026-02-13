function parseJSONScript(scriptId) {
  const node = document.getElementById(scriptId);
  if (!node) return null;
  try {
    return JSON.parse(node.textContent);
  } catch (error) {
    console.error(`No se pudo parsear ${scriptId}:`, error);
    return null;
  }
}

function animateCounters() {
  const counters = document.querySelectorAll('.counter[data-target]');
  counters.forEach((counter) => {
    const target = Number(counter.getAttribute('data-target'));
    if (Number.isNaN(target)) return;

    const duration = 850;
    const start = performance.now();

    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      counter.textContent = Math.round(eased * target).toLocaleString('es-EC');
      if (progress < 1) requestAnimationFrame(tick);
    }

    requestAnimationFrame(tick);
  });
}

function firstDefined(values) {
  for (const value of values) {
    if (value === undefined || value === null) continue;
    const text = String(value).trim();
    if (text !== '') return value;
  }
  return null;
}

function normalizeText(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .toUpperCase();
}

function normalizeCode(value) {
  const digits = String(value || '').replace(/\D/g, '');
  if (!digits) return '';
  const parsed = Number.parseInt(digits, 10);
  return Number.isNaN(parsed) ? digits : String(parsed);
}

function toNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function buildPredictionIndexes(predictions) {
  const byCode = new Map();
  const byName = new Map();

  (predictions || []).forEach((row) => {
    const code = normalizeCode(row.codigo);
    const name = normalizeText(row.parroquia);

    if (code) byCode.set(code, row);
    if (name) byName.set(name, row);
  });

  return { byCode, byName };
}

function extractBaseProps(rawProps) {
  return {
    codigo: firstDefined([
      rawProps.DPA_PARROQ,
      rawProps.dpa_parroq,
      rawProps.codigo,
      rawProps.CODIGO,
    ]),
    parroquia: firstDefined([
      rawProps.DPA_DESPAR,
      rawProps.dpa_despar,
      rawProps.parroquia,
      rawProps.PARROQUIA,
      'Sin nombre',
    ]),
    canton: firstDefined([
      rawProps.DPA_DESCAN,
      rawProps.dpa_descan,
      rawProps.canton,
      rawProps.CANTON,
      'No disponible',
    ]),
    provincia: firstDefined([
      rawProps.DPA_DESPRO,
      rawProps.dpa_despro,
      rawProps.provincia,
      rawProps.PROVINCIA,
      'No disponible',
    ]),
  };
}

function findPrediction(baseProps, indexes) {
  const code = normalizeCode(baseProps.codigo);
  if (code && indexes.byCode.has(code)) return indexes.byCode.get(code);

  const parishName = normalizeText(baseProps.parroquia);
  if (parishName && indexes.byName.has(parishName)) return indexes.byName.get(parishName);

  return null;
}

function enrichFeature(rawProps, indexes) {
  const base = extractBaseProps(rawProps);
  const provinceNorm = normalizeText(base.provincia);
  const prediction = findPrediction(base, indexes);
  const isGuayas = provinceNorm === 'GUAYAS' || Boolean(prediction);

  const risk = firstDefined([
    prediction?.riesgo_categoria,
    rawProps.riesgo,
    rawProps.riesgo_categoria,
    isGuayas ? 'Sin dato' : 'Fuera de Guayas',
  ]);

  const probabilityRaw = firstDefined([
    prediction?.probabilidad_inundacion,
    rawProps.probabilidad,
    rawProps.probabilidad_inundacion,
  ]);

  const probability = probabilityRaw === null ? null : Number(probabilityRaw);

  return {
    ...base,
    isGuayas,
    risk,
    probability: Number.isFinite(probability) ? probability : null,
  };
}

function getFeatureStyle(enriched) {
  if (!enriched.isGuayas) {
    return {
      fillColor: '#2b2d33',
      color: '#3a3d45',
      weight: 0.4,
      opacity: 0.95,
      fillOpacity: 0.78,
    };
  }

  const riskColors = {
    Alto: '#e2655b',
    Medio: '#f1ba3b',
    Bajo: '#3db480',
    'Sin dato': '#8293a4',
  };

  return {
    fillColor: riskColors[enriched.risk] || '#8293a4',
    color: '#f6fffa',
    weight: 1.05,
    opacity: 0.95,
    fillOpacity: 0.58,
  };
}

function getHoverStyle(enriched) {
  if (!enriched.isGuayas) {
    return {
      weight: 0.65,
      color: '#111418',
      fillOpacity: 0.86,
    };
  }

  return {
    weight: 1.4,
    color: '#ffffff',
    fillOpacity: 0.72,
  };
}

function popupHTML(enriched) {
  if (!enriched.isGuayas) {
    return `
      <strong>${enriched.parroquia}</strong><br>
      Canton: ${enriched.canton}<br>
      Provincia: ${enriched.provincia}<br>
      Riesgo: <strong>Fuera de alcance del modelo</strong><br>
      <small>Esta parroquia se muestra como contexto nacional.</small>
    `;
  }

  const probabilityText = enriched.probability === null
    ? 'No disponible'
    : `${(enriched.probability * 100).toFixed(2)}%`;

  return `
    <strong>${enriched.parroquia}</strong><br>
    Canton: ${enriched.canton}<br>
    Provincia: ${enriched.provincia}<br>
    Riesgo: <strong>${enriched.risk}</strong><br>
    Probabilidad: ${probabilityText}
  `;
}

function markerPopupHTML(row) {
  const probability = toNumber(row.probabilidad_inundacion);
  const probabilityText = probability === null
    ? 'No disponible'
    : `${(probability * 100).toFixed(2)}%`;

  return `
    <strong>${row.parroquia || 'Sin nombre'}</strong><br>
    Canton: ${row.canton || 'No disponible'}<br>
    Provincia: ${row.provincia || 'No disponible'}<br>
    Riesgo: <strong>${row.riesgo_categoria || 'Sin dato'}</strong><br>
    Probabilidad: ${probabilityText}
  `;
}

function riskToClass(risk) {
  if (risk === 'Alto') return 'high';
  if (risk === 'Medio') return 'medium';
  if (risk === 'Bajo') return 'low';
  return 'nodata';
}

function createRiskPinIcon(risk) {
  const cls = riskToClass(risk);
  return L.divIcon({
    className: 'risk-pin-wrapper',
    html: `<span class="risk-pin risk-${cls}"></span>`,
    iconSize: [20, 20],
    iconAnchor: [10, 18],
    popupAnchor: [0, -14],
  });
}

function addPredictionPins(map, predictions, infoControl) {
  const group = L.featureGroup();

  (predictions || []).forEach((row) => {
    const lat = toNumber(row.latitud);
    const lon = toNumber(row.longitud);
    if (lat === null || lon === null) return;

    const marker = L.marker([lat, lon], {
      icon: createRiskPinIcon(row.riesgo_categoria),
      keyboard: true,
    });

    marker.bindPopup(markerPopupHTML(row));

    marker.on('mouseover', () => {
      infoControl.update({
        parroquia: row.parroquia,
        canton: row.canton,
        provincia: row.provincia,
        risk: row.riesgo_categoria || 'Sin dato',
      });
    });

    marker.on('mouseout', () => {
      infoControl.update();
    });

    group.addLayer(marker);
  });

  group.addTo(map);
  return group;
}

function createInfoControl(map) {
  const info = L.control({ position: 'topright' });

  info.onAdd = function onAdd() {
    this._div = L.DomUtil.create('div', 'info-box');
    this.update();
    return this._div;
  };

  info.update = function update(enriched) {
    if (!enriched) {
      this._div.innerHTML = '<b>Detalle de parroquia</b>Pasa el cursor sobre un alfiler o una parroquia.';
      return;
    }

    this._div.innerHTML = `
      <b>${enriched.parroquia}</b>
      Canton: ${enriched.canton}<br>
      Provincia: ${enriched.provincia}<br>
      Riesgo: ${enriched.risk || 'No disponible'}
    `;
  };

  info.addTo(map);
  return info;
}

function addLegend(map) {
  const legend = L.control({ position: 'bottomright' });

  legend.onAdd = function onAdd() {
    const div = L.DomUtil.create('div', 'legend');
    div.innerHTML = `
      <h4>Riesgo (areas + alfileres)</h4>
      <div class="item"><span class="pin-dot pin-high"></span>Alto (Guayas)</div>
      <div class="item"><span class="pin-dot pin-medium"></span>Medio (Guayas)</div>
      <div class="item"><span class="pin-dot pin-low"></span>Bajo (Guayas)</div>
      <div class="item"><span class="swatch" style="background:#2b2d33"></span>Fuera de Guayas</div>
    `;
    return div;
  };

  legend.addTo(map);
}

function renderArcGISLayer(map, arcgisUrl, indexes, infoControl, onReady, onFail) {
  if (!window.L?.esri?.featureLayer || !arcgisUrl) {
    onFail();
    return null;
  }

  let readyNotified = false;
  let failNotified = false;

  const nonGuayasLayer = L.esri.featureLayer({
    url: arcgisUrl,
    where: "DPA_DESPRO <> 'GUAYAS'",
    precision: 5,
    simplifyFactor: 0.35,
    style() {
      return getFeatureStyle({ isGuayas: false });
    },
  });

  const guayasLayer = L.esri.featureLayer({
    url: arcgisUrl,
    where: "DPA_DESPRO = 'GUAYAS'",
    precision: 5,
    simplifyFactor: 0.35,
    style(feature) {
      const enriched = enrichFeature(feature.properties || {}, indexes);
      return getFeatureStyle({ ...enriched, isGuayas: true });
    },
  });

  guayasLayer.bindPopup((layer) => {
    const props = layer.feature?.properties || {};
    const enriched = enrichFeature(props, indexes);
    return popupHTML({ ...enriched, isGuayas: true });
  });

  guayasLayer.on('mouseover', (event) => {
    const props = event.layer.feature?.properties || {};
    const enriched = enrichFeature(props, indexes);
    event.layer.setStyle(getHoverStyle({ ...enriched, isGuayas: true }));
    infoControl.update({ ...enriched, isGuayas: true });
  });

  guayasLayer.on('mouseout', (event) => {
    const props = event.layer.feature?.properties || {};
    const enriched = enrichFeature(props, indexes);
    event.layer.setStyle(getFeatureStyle({ ...enriched, isGuayas: true }));
    infoControl.update();
  });

  guayasLayer.on('click', (event) => {
    map.fitBounds(event.layer.getBounds(), { padding: [18, 18] });
  });

  const readyHandler = () => {
    if (readyNotified) return;
    readyNotified = true;
    const bounds = guayasLayer.getBounds();
    onReady(bounds);
  };

  guayasLayer.on('createfeature', readyHandler);
  guayasLayer.on('load', readyHandler);

  const failHandler = () => {
    if (readyNotified || failNotified) return;
    failNotified = true;
    console.warn('No fue posible cargar ArcGIS. Se mantiene mapa con alfileres.');
    onFail();
  };

  nonGuayasLayer.on('requesterror', failHandler);
  nonGuayasLayer.on('error', failHandler);
  guayasLayer.on('requesterror', failHandler);
  guayasLayer.on('error', failHandler);

  nonGuayasLayer.addTo(map);
  guayasLayer.addTo(map);
  return { nonGuayasLayer, guayasLayer };
}

function initMap() {
  const summary = parseJSONScript('summary-data') || { latitud_centro: -2.17, longitud_centro: -79.9 };
  const predictions = parseJSONScript('predictions-data') || [];
  const arcgisUrl = parseJSONScript('arcgis-url-data') || '';

  const indexes = buildPredictionIndexes(predictions);

  const map = L.map('map', {
    zoomControl: true,
    minZoom: 6,
    maxZoom: 14,
  }).setView([summary.latitud_centro, summary.longitud_centro], 7);

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    maxZoom: 20,
  }).addTo(map);

  const infoControl = createInfoControl(map);
  addLegend(map);

  const pinLayer = addPredictionPins(map, predictions, infoControl);
  const pinBounds = pinLayer.getBounds();

  if (pinBounds && pinBounds.isValid()) {
    map.fitBounds(pinBounds.pad(0.28), { padding: [10, 10] });
  }

  let arcLoaded = false;

  renderArcGISLayer(
    map,
    arcgisUrl,
    indexes,
    infoControl,
    () => {
      arcLoaded = true;
    },
    () => {
      arcLoaded = false;
    },
  );

  const resetButton = document.getElementById('btn-reset-map');
  if (resetButton) {
    resetButton.addEventListener('click', () => {
      if (pinBounds && pinBounds.isValid()) {
        map.fitBounds(pinBounds.pad(0.28), { padding: [10, 10] });
      } else {
        map.setView([summary.latitud_centro, summary.longitud_centro], 7);
      }
    });
  }

  setTimeout(() => {
    map.invalidateSize();
    if (!arcLoaded && pinBounds && pinBounds.isValid()) {
      map.fitBounds(pinBounds.pad(0.28), { padding: [10, 10] });
    }
  }, 800);
}

document.addEventListener('DOMContentLoaded', () => {
  animateCounters();
  initMap();
});
