// public/ui/app.js
// Public browser-only PV forecast calculator (today only)
//
// - Timezone fixed: Europe/Prague
// - PR fixed: 0.82
// - Request throttle: 30s per browser (localStorage)

const TZ = "Europe/Prague";
const PR = 0.82;
const THROTTLE_MS = 30_000;
const LANG_STORAGE_KEY = "pv_calc_lang";
const COOKIE_CONSENT_KEY = "pv_cookie_consent";
const GA_MEASUREMENT_ID = "G-CS8YDWNBV1";
const DEFAULTS = {
  lat: 49.19483604326329,
  lon: 16.60870320672247,
  kwp: 10,
  tilt: 25,
  az: 200
};
const TRANSLATIONS = {
  cs: {
    pageTitle: "FVE predikce výroby",
    heroTitle: "FVE predikce výroby",
    timezoneLabel: `Časové pásmo: ${TZ}`,
    weatherSource: "Zdroj počasí: Open-Meteo",
    locationTitle: "1. Umístění",
    mapHelp: "Klikněte do mapy nebo přetáhněte marker na přesné místo instalace.",
    latLabel: "Zeměpisná šířka",
    lonLabel: "Zeměpisná délka",
    latPlaceholder: "např. 49.1694",
    lonPlaceholder: "např. 16.5097",
    paramsTitle: "2. Parametry elektrárny",
    kwpLabel: "Výkon elektrárny (kWp)",
    kwpPlaceholder: "např. 8,2",
    tiltLabel: "Sklon střechy (°)",
    tiltPlaceholder: "0–90, např. 25",
    panelTiltLabel: "Sklon panelů",
    azimuthLabel: "Azimut (° od severu)",
    azPlaceholder: "0–360, např. 221",
    scaleNorth: "0° N",
    scaleEast: "90° E",
    scaleSouth: "180° S",
    scaleWest: "270° W",
    azimuthHelp: "Šipka ukazuje směr, kterým panely míří. 180° znamená jih, 90° východ a 270° západ.",
    runButton: "Spočítat dnešní výrobu",
    resultTitle: "Výsledek",
    todayPrediction: "Predikce dnes (kWh)",
    tomorrowPrediction: "Predikce zítra (kWh)",
    todayChartTitle: "Dnes",
    tomorrowChartTitle: "Zítra",
    cookieTitle: "Cookies a analytika",
    cookieText: "Tento web používá pouze nezbytné úložiště pro zapamatování vašeho nastavení a volitelně analytické cookies Google Analytics pro anonymní statistiky návštěvnosti. Analytika se načte až po vašem souhlasu.",
    cookieAccept: "Povolit analytiku",
    cookieReject: "Pouze nezbytné",
    cookieSettingsButton: "Nastavení cookies",
    cooldown: "Další výpočet za {seconds}s",
    mapClickSet: "Poloha byla nastavena kliknutím do mapy.",
    mapDragSet: "Marker byl přesunut na novou polohu.",
    mapUnavailable: "Mapu se nepodařilo načíst. Souřadnice lze zadat ručně.",
    mapUnavailableInteractive: "Interaktivní mapa je v tomto prostředí nedostupná.",
    dateError: "Nelze určit dnešní datum z Open-Meteo.",
    throttleError: "Z důvodu limitu lze spustit výpočet nejdříve za {seconds}s.",
    latError: "Latitude musí být v rozsahu -90 až 90.",
    lonError: "Longitude musí být v rozsahu -180 až 180.",
    kwpError: "kWp musí být v rozsahu 0 až 100.",
    tiltError: "Sklon musí být v rozsahu 0 až 90 stupňů.",
    azError: "Azimut musí být v rozsahu 0 až <360 stupňů (od severu).",
    chartTodayDataset: "Predikovaný výkon dnes (kW)",
    chartTomorrowDataset: "Predikovaný výkon zítra (kW)",
    chartTimeAxis: `Čas (${TZ})`,
    azimuthDirections: [
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
    ]
  },
  en: {
    pageTitle: "PV Production Forecast",
    heroTitle: "PV Production Forecast",
    timezoneLabel: `Time zone: ${TZ}`,
    weatherSource: "Weather source: Open-Meteo",
    locationTitle: "1. Location",
    mapHelp: "Click on the map or drag the marker to the exact installation location.",
    latLabel: "Latitude",
    lonLabel: "Longitude",
    latPlaceholder: "e.g. 49.1694",
    lonPlaceholder: "e.g. 16.5097",
    paramsTitle: "2. Plant Parameters",
    kwpLabel: "Installed capacity (kWp)",
    kwpPlaceholder: "e.g. 8.2",
    tiltLabel: "Roof tilt (°)",
    tiltPlaceholder: "0–90, e.g. 25",
    panelTiltLabel: "Panel tilt",
    azimuthLabel: "Azimuth (° from north)",
    azPlaceholder: "0–360, e.g. 221",
    scaleNorth: "0° N",
    scaleEast: "90° E",
    scaleSouth: "180° S",
    scaleWest: "270° W",
    azimuthHelp: "The arrow shows the direction the panels face. 180° means south, 90° east, and 270° west.",
    runButton: "Calculate today's production",
    resultTitle: "Results",
    todayPrediction: "Today's forecast (kWh)",
    tomorrowPrediction: "Tomorrow's forecast (kWh)",
    todayChartTitle: "Today",
    tomorrowChartTitle: "Tomorrow",
    cookieTitle: "Cookies and analytics",
    cookieText: "This site uses only essential storage to remember your settings and, optionally, Google Analytics cookies for anonymous traffic statistics. Analytics is loaded only after you give consent.",
    cookieAccept: "Allow analytics",
    cookieReject: "Essential only",
    cookieSettingsButton: "Cookie settings",
    cooldown: "Next calculation in {seconds}s",
    mapClickSet: "Location was set by clicking on the map.",
    mapDragSet: "The marker was moved to a new location.",
    mapUnavailable: "The map could not be loaded. Coordinates can still be entered manually.",
    mapUnavailableInteractive: "Interactive map is unavailable in this environment.",
    dateError: "Unable to determine today's date from Open-Meteo.",
    throttleError: "Due to the request limit, the next calculation can run in {seconds}s.",
    latError: "Latitude must be in the range -90 to 90.",
    lonError: "Longitude must be in the range -180 to 180.",
    kwpError: "kWp must be in the range 0 to 100.",
    tiltError: "Tilt must be in the range 0 to 90 degrees.",
    azError: "Azimuth must be in the range 0 to <360 degrees (from north).",
    chartTodayDataset: "Predicted power today (kW)",
    chartTomorrowDataset: "Predicted power tomorrow (kW)",
    chartTimeAxis: `Time (${TZ})`,
    azimuthDirections: [
      "North",
      "North-northeast",
      "Northeast",
      "East-northeast",
      "East",
      "East-southeast",
      "Southeast",
      "South-southeast",
      "South",
      "South-southwest",
      "Southwest",
      "West-southwest",
      "West",
      "West-northwest",
      "Northwest",
      "North-northwest"
    ]
  }
};

let todayChart = null;
let tomorrowChart = null;
let map = null;
let marker = null;
let currentLang = "cs";
let currentMapStatusKey = "mapHelp";
let lastTodaySeries = null;
let lastTomorrowSeries = null;
let analyticsLoaded = false;

function $(id) { return document.getElementById(id); }

function setError(msg) { $("err").textContent = msg || ""; }

function getCookieConsent() {
  return localStorage.getItem(COOKIE_CONSENT_KEY);
}

function setCookieConsent(value) {
  localStorage.setItem(COOKIE_CONSENT_KEY, value);
}

function loadAnalytics() {
  if (analyticsLoaded) return;
  analyticsLoaded = true;
  window.dataLayer = window.dataLayer || [];
  window.gtag = function gtag() { window.dataLayer.push(arguments); };
  window.gtag("js", new Date());
  window.gtag("config", GA_MEASUREMENT_ID);

  const script = document.createElement("script");
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`;
  document.head.appendChild(script);
}

function updateCookieUi() {
  const consent = getCookieConsent();
  $("cookieBanner").hidden = consent === "accepted" || consent === "rejected";
  $("cookieSettingsBtn").hidden = false;
}

function openCookieBanner() {
  $("cookieBanner").hidden = false;
}

function closeCookieBanner() {
  $("cookieBanner").hidden = true;
}

function applyCookieConsent(consent) {
  setCookieConsent(consent);
  if (consent === "accepted") loadAnalytics();
  closeCookieBanner();
  updateCookieUi();
}

function t(key, vars = {}) {
  const dict = TRANSLATIONS[currentLang] || TRANSLATIONS.cs;
  let text = dict[key] ?? TRANSLATIONS.cs[key] ?? key;
  for (const [name, value] of Object.entries(vars)) {
    text = text.replaceAll(`{${name}}`, String(value));
  }
  return text;
}

function setMapStatus(msg, key = null) {
  const el = $("mapStatus");
  if (el) el.textContent = msg || "";
  if (key) currentMapStatusKey = key;
}

function setMapStatusKey(key, vars = {}) {
  currentMapStatusKey = key;
  setMapStatus(t(key, vars));
}

function setCooldownInfo(msRemaining) {
  if (!msRemaining || msRemaining <= 0) {
    $("cooldownInfo").textContent = "";
    return;
  }
  const s = Math.ceil(msRemaining / 1000);
  $("cooldownInfo").textContent = t("cooldown", { seconds: s });
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
  if (!Number.isFinite(lat) || lat < -90 || lat > 90) errs.push(t("latError"));
  if (!Number.isFinite(lon) || lon < -180 || lon > 180) errs.push(t("lonError"));
  if (!Number.isFinite(kwp) || kwp <= 0 || kwp > 100) errs.push(t("kwpError"));
  if (!Number.isFinite(tilt) || tilt < 0 || tilt > 90) errs.push(t("tiltError"));
  if (!Number.isFinite(az) || az < 0 || az >= 360) errs.push(t("azError"));
  return errs;
}

function toNum(x) {
  const v = Number(x);
  return Number.isFinite(v) ? v : null;
}

function parseLocaleNumber(value) {
  if (typeof value === "number") return Number.isFinite(value) ? value : NaN;
  const normalized = String(value ?? "").trim().replace(",", ".");
  return Number(normalized);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function round2(x) {
  return Math.round(x * 100) / 100;
}

function normalizeAzimuth(value) {
  if (!Number.isFinite(value)) return 0;
  return ((Math.round(value) % 360) + 360) % 360;
}

function azimuthLabel(az) {
  const labels = t("azimuthDirections");
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
  url.searchParams.set("forecast_days", "2");
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

function renderChart(canvasId, existingChart, labels, pvKw, datasetLabel, colors) {
  const ctx = $(canvasId).getContext("2d");
  if (existingChart) existingChart.destroy();

  return new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: datasetLabel,
        data: pvKw,
        tension: 0.35,
        pointRadius: 0,
        borderColor: colors.border,
        borderWidth: 3,
        fill: true,
        backgroundColor: colors.fill
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
          title: { display: true, text: t("chartTimeAxis") },
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

function addDays(dateStr, dayCount) {
  const [year, month, day] = String(dateStr).split("-").map(Number);
  const date = new Date(Date.UTC(year, month - 1, day));
  date.setUTCDate(date.getUTCDate() + dayCount);
  return date.toISOString().slice(0, 10);
}

function buildDaySeries(hourly, targetDate, kwp) {
  const labels = [];
  const pvKw = [];
  let kwhSum = 0;

  for (const r of hourly) {
    if (parseDatePart(r.time) !== targetDate) continue;

    const hhmm = r.time.slice(11, 16);
    const kw = pvKwFromIrr(kwp, r.irr_wm2);

    labels.push(hhmm);
    pvKw.push(round2(kw));
    kwhSum += kw;
  }

  return {
    labels,
    pvKw,
    kwh: round2(kwhSum).toFixed(2)
  };
}

function updateAzimuthUI() {
  const az = normalizeAzimuth(Number($("az").value));
  $("az").value = String(az);
  $("azRange").value = String(az);
  $("azValue").textContent = `${az}°`;
  $("azLabel").textContent = azimuthLabel(az);
  const needle = $("azNeedleGroup");
  if (needle) needle.setAttribute("transform", `rotate(${az} 120 120)`);
}

function applyTranslations() {
  document.documentElement.lang = currentLang;
  document.title = t("pageTitle");
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  $("langCs").classList.toggle("active", currentLang === "cs");
  $("langEn").classList.toggle("active", currentLang === "en");
  setMapStatusKey(currentMapStatusKey);
  updateTiltUI();
  updateAzimuthUI();
  if (lastTodaySeries && lastTomorrowSeries) {
    todayChart = renderChart(
      "chartToday",
      todayChart,
      lastTodaySeries.labels,
      lastTodaySeries.pvKw,
      t("chartTodayDataset"),
      { border: "#ff7a18", fill: "rgba(255, 122, 24, 0.16)" }
    );
    tomorrowChart = renderChart(
      "chartTomorrow",
      tomorrowChart,
      lastTomorrowSeries.labels,
      lastTomorrowSeries.pvKw,
      t("chartTomorrowDataset"),
      { border: "#d97706", fill: "rgba(217, 119, 6, 0.14)" }
    );
  }
}

function setLanguage(lang) {
  currentLang = lang === "en" ? "en" : "cs";
  localStorage.setItem(LANG_STORAGE_KEY, currentLang);
  applyTranslations();
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

  $("lat").value = latNum.toFixed(5);
  $("lon").value = lonNum.toFixed(5);

  if (updateMap && marker && map) {
    marker.setLatLng([latNum, lonNum]);
    map.panTo([latNum, lonNum], { animate: true });
  }

  if (mapStatus) setMapStatus(mapStatus);
}

function syncMapFromInputs() {
  const lat = Number($("lat").value);
  const lon = Number($("lon").value);

  if (marker && map && Number.isFinite(lat) && Number.isFinite(lon)) {
    marker.setLatLng([lat, lon]);
    map.panTo([lat, lon], { animate: false });
  }
}

function showMapFallback(message) {
  $("map").hidden = true;
  if (message) setMapStatus(message);
}

function initMap() {
  if (!window.L) {
    showMapFallback(t("mapUnavailable"));
    return;
  }

  const lat = Number($("lat").value) || DEFAULTS.lat;
  const lon = Number($("lon").value) || DEFAULTS.lon;

  try {
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
        mapStatus: t("mapClickSet")
      });
      currentMapStatusKey = "mapClickSet";
    });

    marker.on("dragend", () => {
      const position = marker.getLatLng();
      setCoordinates(position.lat, position.lng, {
        updateMap: false,
        mapStatus: t("mapDragSet")
      });
      currentMapStatusKey = "mapDragSet";
    });

    setTimeout(() => {
      if (map) map.invalidateSize();
    }, 0);
  } catch (error) {
    map = null;
    marker = null;
    showMapFallback(t("mapUnavailableInteractive"));
  }
}

async function run() {
  setError("");

  const now = Date.now();
  const last = getLastRunTs();
  const remaining = THROTTLE_MS - (now - last);
  if (remaining > 0) {
    setCooldownInfo(remaining);
    throw new Error(t("throttleError", { seconds: Math.ceil(remaining / 1000) }));
  }

  const lat = Number($("lat").value);
  const lon = Number($("lon").value);
  const kwp = parseLocaleNumber($("kwp").value);
  const tilt = Number($("tilt").value);
  const az = Number($("az").value);

  const errs = validateInputs(lat, lon, kwp, tilt, az);
  if (errs.length) throw new Error(errs.join("\n"));

  $("btnRun").disabled = true;

  const hourly = await fetchOpenMeteoHourly(lat, lon, tilt, az);
  const todayDate = parseDatePart(hourly[0]?.time);
  if (!todayDate) throw new Error(t("dateError"));
  const tomorrowDate = addDays(todayDate, 1);

  const todaySeries = buildDaySeries(hourly, todayDate, kwp);
  const tomorrowSeries = buildDaySeries(hourly, tomorrowDate, kwp);
  lastTodaySeries = todaySeries;
  lastTomorrowSeries = tomorrowSeries;

  $("kwhToday").textContent = todaySeries.kwh;
  $("kwhTomorrow").textContent = tomorrowSeries.kwh;

  todayChart = renderChart(
    "chartToday",
    todayChart,
    todaySeries.labels,
    todaySeries.pvKw,
    t("chartTodayDataset"),
    { border: "#ff7a18", fill: "rgba(255, 122, 24, 0.16)" }
  );

  tomorrowChart = renderChart(
    "chartTomorrow",
    tomorrowChart,
    tomorrowSeries.labels,
    tomorrowSeries.pvKw,
    t("chartTomorrowDataset"),
    { border: "#d97706", fill: "rgba(217, 119, 6, 0.14)" }
  );

  setLastRunTs(Date.now());
  setCooldownInfo(THROTTLE_MS);
}

function initDefaults() {
  currentLang = localStorage.getItem(LANG_STORAGE_KEY) || "cs";
  const saved = JSON.parse(localStorage.getItem("pv_calc_defaults") || "null");
  const values = saved || DEFAULTS;

  $("lat").value = values.lat ?? DEFAULTS.lat;
  $("lon").value = values.lon ?? DEFAULTS.lon;
  $("kwp").value = values.kwp ?? DEFAULTS.kwp;
  $("tilt").value = values.tilt ?? DEFAULTS.tilt;
  $("az").value = values.az ?? DEFAULTS.az;
  $("tiltRange").value = values.tilt ?? DEFAULTS.tilt;
  $("azRange").value = values.az ?? DEFAULTS.az;

  applyTranslations();
  updateCookieUi();
  if (getCookieConsent() === "accepted") loadAnalytics();
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
  $("langCs").addEventListener("click", () => setLanguage("cs"));
  $("langEn").addEventListener("click", () => setLanguage("en"));
  $("cookieAccept").addEventListener("click", () => applyCookieConsent("accepted"));
  $("cookieReject").addEventListener("click", () => applyCookieConsent("rejected"));
  $("cookieSettingsBtn").addEventListener("click", openCookieBanner);
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
