// public/ui/app.js
// Public browser-only PV forecast calculator (today only)
//
// - Timezone fixed: Europe/Prague
// - PR fixed: 0.82
// - Request throttle: 30s per browser (localStorage)

const TZ = "Europe/Prague";
const PR = 0.82;
const THROTTLE_MS = 30_000;
const DEFAULTS = {
  lat: 49.19483604326329,
  lon: 16.60870320672247,
  kwp: 10,
  tilt: 25,
  az: 200
};

let chart = null;
let map = null;
let marker = null;
let suppressCoordinateSync = false;

function $(id) { return document.getElementById(id); }

function setError(msg) { $("err").textContent = msg || ""; }

function setMapStatus(msg) {
  const el = $("mapStatus");
  if (el) el.textContent = msg || "";
}

function setCooldownInfo(msRemaining) {
  if (!msRemaining || msRemaining <= 0) {
    $("cooldownInfo").textContent = "";
    return;
  }
  const s = Math.ceil(msRemaining / 1000);
  $("cooldownInfo").textContent = `Další výpočet za ${s}s`;
}

function getLastRunTs() {
  const v = localStorage.getItem("pv_calc_last_run_ts");
  return v ? Number(v) : 0;
}

function setLastRunTs(ts) {
  localStorage.setItem("pv_calc_last_run_ts", String(ts));
}

function validateInputs(lat, lon, kwp, tilt, az) {
  const errs = [];
  if (!Number.isFinite(lat) || lat < -90 || lat > 90) errs.push("Latitude musí být v rozsahu -90 až 90.");
  if (!Number.isFinite(lon) || lon < -180 || lon > 180) errs.push("Longitude musí být v rozsahu -180 až 180.");
  if (!Number.isFinite(kwp) || kwp <= 0 || kwp > 100) errs.push("kWp musí být v rozsahu 0 až 100.");
  if (!Number.isFinite(tilt) || tilt < 0 || tilt > 90) errs.push("Sklon musí být v rozsahu 0 až 90 stupňů.");
  if (!Number.isFinite(az) || az < 0 || az >= 360) errs.push("Azimut musí být v rozsahu 0 až <360 stupňů (od severu).");
  return errs;
}

function toNum(x) {
  const v = Number(x);
  return Number.isFinite(v) ? v : null;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function round2(x) {
  return Math.round(x * 100) / 100;
}

function formatCoord(value) {
  return Number.isFinite(value) ? value.toFixed(5) : "–";
}

function normalizeAzimuth(value) {
  if (!Number.isFinite(value)) return 0;
  return ((Math.round(value) % 360) + 360) % 360;
}

function azimuthLabel(az) {
  const labels = [
    "Sever",
    "Severo-severovýchod",
    "Severovýchod",
    "Východo-severovýchod",
    "Východ",
    "Východo-jihovýchod",
    "Jihovýchod",
    "Jiho-jihovýchod",
    "Jih",
    "Jiho-jihozápad",
    "Jihozápad",
    "Západo-jihozápad",
    "Západ",
    "Západo-severozápad",
    "Severozápad",
    "Severo-severozápad"
  ];
  return labels[Math.round(normalizeAzimuth(az) / 22.5) % 16];
}

async function fetchOpenMeteoHourly(lat, lon, tiltDeg, azFromNorthDeg) {
  const url = new URL("https://api.open-meteo.com/v1/forecast");
  url.searchParams.set("latitude", String(lat));
  url.searchParams.set("longitude", String(lon));
  url.searchParams.set("timezone", TZ);

  // Open-Meteo expects:
  // - tilt: degrees from horizontal (0..90)
  // - azimuth: degrees from South (0=S, -90=E, +90=W, ±180=N)
  // UI input az is degrees clockwise from North (0=N, 90=E, 180=S, 270=W)
  const azOpen = ((((azFromNorthDeg - 180) + 180) % 360) - 180);

  url.searchParams.set("tilt", String(tiltDeg));
  url.searchParams.set("azimuth", String(azOpen));
  url.searchParams.set("forecast_days", "1");
  url.searchParams.set("hourly", "global_tilted_irradiance,shortwave_radiation,cloud_cover");

  const r = await fetch(url.toString(), { method: "GET" });
  if (!r.ok) throw new Error(`Open-Meteo HTTP ${r.status}`);

  const j = await r.json();
  if (!j.hourly || !j.hourly.time) throw new Error("Open-Meteo response missing hourly time series.");

  const times = j.hourly.time;
  const gti = j.hourly.global_tilted_irradiance || null;
  const swr = j.hourly.shortwave_radiation || null;
  const cloud = j.hourly.cloud_cover || null;

  const irrSource = Array.isArray(gti) ? "gti" : (Array.isArray(swr) ? "shortwave_radiation" : null);
  if (!irrSource) throw new Error("Open-Meteo did not return irradiance series.");

  const irrArr = irrSource === "gti" ? gti : swr;

  return times.map((t, i) => ({
    time: String(t),
    irr_wm2: toNum(irrArr[i]),
    cloud_cover: cloud ? toNum(cloud[i]) : null,
    irr_source: irrSource
  }));
}

function pvKwFromIrr(kwp, irr_wm2) {
  const irr = Number.isFinite(irr_wm2) ? irr_wm2 : 0;
  if (irr <= 0) return 0;
  return kwp * (irr / 1000.0) * PR;
}

function renderChart(labels, pvKw) {
  const ctx = $("chart").getContext("2d");
  if (chart) {
    chart.destroy();
    chart = null;
  }

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Predikovaný výkon (kW)",
        data: pvKw,
        tension: 0.35,
        pointRadius: 0,
        borderColor: "#ff7a18",
        borderWidth: 3,
        fill: true,
        backgroundColor: "rgba(255, 122, 24, 0.16)"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        y: {
          title: { display: true, text: "kW" },
          beginAtZero: true,
          grid: { color: "rgba(21, 31, 52, 0.08)" }
        },
        x: {
          title: { display: true, text: `Čas (${TZ})` },
          grid: { display: false }
        }
      },
      plugins: {
        legend: { display: true }
      }
    }
  });
}

function parseDatePart(timeStr) {
  return timeStr ? String(timeStr).slice(0, 10) : null;
}

function updateCoordinatePreview(lat, lon) {
  $("latPreview").textContent = formatCoord(lat);
  $("lonPreview").textContent = formatCoord(lon);
}

function updateAzimuthUI() {
  const az = normalizeAzimuth(Number($("az").value));
  $("az").value = String(az);
  $("azRange").value = String(az);
  $("azValue").textContent = `${az}°`;
  $("azLabel").textContent = azimuthLabel(az);
  $("azimuthGraphic").style.setProperty("--azimuth", `${az}deg`);
}

function updateTiltUI() {
  const tilt = clamp(Number($("tilt").value) || 0, 0, 90);
  $("tilt").value = String(tilt);
  $("tiltRange").value = String(tilt);
  $("tiltValue").textContent = `${tilt}°`;
}

function setCoordinates(lat, lon, options = {}) {
  const { updateMap = true, mapStatus = "" } = options;
  const latNum = Number(lat);
  const lonNum = Number(lon);

  if (!Number.isFinite(latNum) || !Number.isFinite(lonNum)) return;

  suppressCoordinateSync = true;
  $("lat").value = latNum.toFixed(5);
  $("lon").value = lonNum.toFixed(5);
  suppressCoordinateSync = false;

  updateCoordinatePreview(latNum, lonNum);

  if (updateMap && marker && map) {
    marker.setLatLng([latNum, lonNum]);
    map.panTo([latNum, lonNum], { animate: true });
  }

  if (mapStatus) setMapStatus(mapStatus);
}

function syncMapFromInputs() {
  if (suppressCoordinateSync) return;
  const lat = Number($("lat").value);
  const lon = Number($("lon").value);
  updateCoordinatePreview(lat, lon);

  if (marker && map && Number.isFinite(lat) && Number.isFinite(lon)) {
    marker.setLatLng([lat, lon]);
    map.panTo([lat, lon], { animate: false });
  }
}

function initMap() {
  if (!window.L) {
    setMapStatus("Mapu se nepodařilo načíst. Souřadnice lze zadat ručně.");
    return;
  }

  const lat = Number($("lat").value) || DEFAULTS.lat;
  const lon = Number($("lon").value) || DEFAULTS.lon;

  map = L.map("map", {
    zoomControl: true,
    scrollWheelZoom: true
  }).setView([lat, lon], 13);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
  }).addTo(map);

  marker = L.marker([lat, lon], { draggable: true }).addTo(map);

  map.on("click", (event) => {
    setCoordinates(event.latlng.lat, event.latlng.lng, {
      updateMap: true,
      mapStatus: "Poloha byla nastavena kliknutím do mapy."
    });
  });

  marker.on("dragend", () => {
    const position = marker.getLatLng();
    setCoordinates(position.lat, position.lng, {
      updateMap: false,
      mapStatus: "Marker byl přesunut na novou polohu."
    });
  });

  setTimeout(() => map.invalidateSize(), 0);
}

function requestBrowserLocation() {
  if (!navigator.geolocation) {
    setMapStatus("Tento prohlížeč nepodporuje zjištění aktuální polohy.");
    return;
  }

  setMapStatus("Zjišťuji aktuální polohu...");

  navigator.geolocation.getCurrentPosition(
    (position) => {
      setCoordinates(position.coords.latitude, position.coords.longitude, {
        updateMap: true,
        mapStatus: "Aktuální poloha byla načtena z prohlížeče."
      });
      if (map) map.setZoom(15);
    },
    () => {
      setMapStatus("Přístup k poloze nebyl povolen nebo se polohu nepodařilo zjistit.");
    },
    { enableHighAccuracy: true, timeout: 8000 }
  );
}

async function run() {
  setError("");

  const now = Date.now();
  const last = getLastRunTs();
  const remaining = THROTTLE_MS - (now - last);
  if (remaining > 0) {
    setCooldownInfo(remaining);
    throw new Error(`Z důvodu limitu lze spustit výpočet nejdříve za ${Math.ceil(remaining / 1000)}s.`);
  }

  const lat = Number($("lat").value);
  const lon = Number($("lon").value);
  const kwp = Number($("kwp").value);
  const tilt = Number($("tilt").value);
  const az = Number($("az").value);

  const errs = validateInputs(lat, lon, kwp, tilt, az);
  if (errs.length) throw new Error(errs.join("\n"));

  $("btnRun").disabled = true;

  const hourly = await fetchOpenMeteoHourly(lat, lon, tilt, az);
  const todayDate = parseDatePart(hourly[0]?.time);
  if (!todayDate) throw new Error("Nelze určit dnešní datum z Open-Meteo.");

  const labels = [];
  const pvKw = [];
  let kwhSum = 0;

  for (const r of hourly) {
    if (parseDatePart(r.time) !== todayDate) continue;

    const hhmm = r.time.slice(11, 16);
    const kw = pvKwFromIrr(kwp, r.irr_wm2);

    labels.push(hhmm);
    pvKw.push(round2(kw));
    kwhSum += kw;
  }

  $("kwhToday").textContent = round2(kwhSum).toFixed(2);
  $("note").textContent = `Datum: ${todayDate} • Irr source: ${hourly[0]?.irr_source || "?"} • PR=${PR}`;

  renderChart(labels, pvKw);

  setLastRunTs(Date.now());
  setCooldownInfo(THROTTLE_MS);
}

function initDefaults() {
  const saved = JSON.parse(localStorage.getItem("pv_calc_defaults") || "null");
  const values = saved || DEFAULTS;

  $("lat").value = values.lat ?? DEFAULTS.lat;
  $("lon").value = values.lon ?? DEFAULTS.lon;
  $("kwp").value = values.kwp ?? DEFAULTS.kwp;
  $("tilt").value = values.tilt ?? DEFAULTS.tilt;
  $("az").value = values.az ?? DEFAULTS.az;
  $("tiltRange").value = values.tilt ?? DEFAULTS.tilt;
  $("azRange").value = values.az ?? DEFAULTS.az;

  updateCoordinatePreview(Number($("lat").value), Number($("lon").value));
  updateTiltUI();
  updateAzimuthUI();
}

function saveDefaults() {
  const obj = {
    lat: $("lat").value,
    lon: $("lon").value,
    kwp: $("kwp").value,
    tilt: $("tilt").value,
    az: $("az").value
  };
  localStorage.setItem("pv_calc_defaults", JSON.stringify(obj));
}

function hookEvents() {
  $("btnRun").addEventListener("click", async () => {
    try {
      $("btnRun").disabled = true;
      await run();
      saveDefaults();
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      $("btnRun").disabled = false;
    }
  });

  $("btnLocate").addEventListener("click", requestBrowserLocation);

  $("tilt").addEventListener("input", updateTiltUI);
  $("tiltRange").addEventListener("input", () => {
    $("tilt").value = $("tiltRange").value;
    updateTiltUI();
  });

  $("az").addEventListener("input", updateAzimuthUI);
  $("azRange").addEventListener("input", () => {
    $("az").value = $("azRange").value;
    updateAzimuthUI();
  });

  $("lat").addEventListener("input", syncMapFromInputs);
  $("lon").addEventListener("input", syncMapFromInputs);

  ["lat", "lon", "kwp", "tilt", "az"].forEach((id) => {
    $(id).addEventListener("change", saveDefaults);
  });
  ["tiltRange", "azRange"].forEach((id) => {
    $(id).addEventListener("change", saveDefaults);
  });

  const now = Date.now();
  const last = getLastRunTs();
  const remaining = THROTTLE_MS - (now - last);
  if (remaining > 0) setCooldownInfo(remaining);

  setInterval(() => {
    const now2 = Date.now();
    const last2 = getLastRunTs();
    const rem2 = THROTTLE_MS - (now2 - last2);
    setCooldownInfo(rem2);
  }, 500);
}

initDefaults();
initMap();
hookEvents();
