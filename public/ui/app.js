// public/ui/app.js
// Public browser-only PV forecast calculator (today only)
//
// - Timezone fixed: Europe/Prague
// - PR fixed: 0.82
// - Request throttle: 30s per browser (localStorage)

const TZ = "Europe/Prague";
const PR = 0.82;
const THROTTLE_MS = 30_000;

let chart = null;

function $(id) { return document.getElementById(id); }

function setError(msg) { $("err").textContent = msg || ""; }

function setCooldownInfo(msRemaining) {
  if (!msRemaining || msRemaining <= 0) { $("cooldownInfo").textContent = ""; return; }
  const s = Math.ceil(msRemaining / 1000);
  $("cooldownInfo").textContent = `Další výpočet za ${s}s`;
}

function getLastRunTs() {
  const v = localStorage.getItem("pv_calc_last_run_ts");
  return v ? Number(v) : 0;
}
function setLastRunTs(ts) { localStorage.setItem("pv_calc_last_run_ts", String(ts)); }

function validateInputs(lat, lon, kwp, tilt, az) {
  const errs = [];
  if (!Number.isFinite(lat) || lat < -90 || lat > 90) errs.push("Latitude musí být v rozsahu -90 až 90.");
  if (!Number.isFinite(lon) || lon < -180 || lon > 180) errs.push("Longitude musí být v rozsahu -180 až 180.");
  if (!Number.isFinite(kwp) || kwp <= 0 || kwp > 100) errs.push("kWp musí být v rozsahu 0 až 100.");
  if (!Number.isFinite(tilt) || tilt < 0 || tilt > 90) errs.push("Sklon musí být v rozsahu 0 až 90 stupňů.");
  if (!Number.isFinite(az) || az < 0 || az >= 360) errs.push("Azimut musí být v rozsahu 0 až <360 stupňů (od severu).");
  return errs;
}

function toNum(x) { const v = Number(x); return Number.isFinite(v) ? v : null; }

async function fetchOpenMeteoHourly(lat, lon) {
  const url = new URL("https://api.open-meteo.com/v1/forecast");
  url.searchParams.set("latitude", String(lat));
  url.searchParams.set("longitude", String(lon));
  url.searchParams.set("timezone", TZ);
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

function round2(x) { return Math.round(x * 100) / 100; }

function renderChart(labels, pvKw) {
  const ctx = $("chart").getContext("2d");
  if (chart) { chart.destroy(); chart = null; }

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Predikovaný výkon (kW)",
        data: pvKw,
        tension: 0.25,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        y: { title: { display: true, text: "kW" }, beginAtZero: true },
        x: { title: { display: true, text: `Čas (${TZ})` } }
      },
      plugins: { legend: { display: true } }
    }
  });
}

function parseDatePart(timeStr) { return timeStr ? String(timeStr).slice(0, 10) : null; }

async function run() {
  setError("");

  const now = Date.now();
  const last = getLastRunTs();
  const remaining = THROTTLE_MS - (now - last);
  if (remaining > 0) {
    setCooldownInfo(remaining);
    throw new Error(`Z důvodu limitu lze spustit výpočet nejdříve za ${Math.ceil(remaining/1000)}s.`);
  }

  const lat = Number($("lat").value);
  const lon = Number($("lon").value);
  const kwp = Number($("kwp").value);
  const tilt = Number($("tilt").value); // reserved for future
  const az = Number($("az").value);     // reserved for future

  const errs = validateInputs(lat, lon, kwp, tilt, az);
  if (errs.length) throw new Error(errs.join("\n"));

  $("btnRun").disabled = true;

  const hourly = await fetchOpenMeteoHourly(lat, lon);
  const todayDate = parseDatePart(hourly[0]?.time);
  if (!todayDate) throw new Error("Nelze určit dnešní datum z Open-Meteo.");

  const labels = [];
  const pvKw = [];
  let kwhSum = 0;

  for (const r of hourly) {
    const t = r.time;
    if (parseDatePart(t) !== todayDate) continue;

    const hhmm = t.includes("T") ? t.slice(11, 16) : t.slice(11, 16);
    const kw = pvKwFromIrr(kwp, r.irr_wm2);

    labels.push(hhmm);
    pvKw.push(round2(kw));
    kwhSum += kw; // 1 hour step
  }

  $("kwhToday").textContent = round2(kwhSum).toFixed(2);
  $("note").textContent = `Datum: ${todayDate} • Irr source: ${hourly[0]?.irr_source || "?"} • PR=${PR}`;

  renderChart(labels, pvKw);

  setLastRunTs(Date.now());
  setCooldownInfo(THROTTLE_MS);
}

function initDefaults() {
  const saved = JSON.parse(localStorage.getItem("pv_calc_defaults") || "null");
  if (saved) {
    $("lat").value = saved.lat ?? "";
    $("lon").value = saved.lon ?? "";
    $("kwp").value = saved.kwp ?? "";
    $("tilt").value = saved.tilt ?? "";
    $("az").value = saved.az ?? "";
  } else {
    // UPDATED DEFAULTS
    $("lat").value = "49.19483604326329";
    $("lon").value = "16.60870320672247";
    $("kwp").value = "10";
    $("tilt").value = "25";
    $("az").value = "200";
  }
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
hookEvents();
