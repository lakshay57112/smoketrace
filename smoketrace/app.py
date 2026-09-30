"""
SmokeTrace — "Who is polluting my air?"
Build with AI: Code for Communities (Track 2 — Clean Air & Climate Resilience)

Features
1. Trace-back  : wind back-trajectory (Open-Meteo) + satellite fires (NASA FIRMS)
2. Photo check : Gemini classifies smoke source + severity, pins it on a shared map
3. Early warning: 48h PM2.5 forecast + plain-language advice
4. One-tap action: Gemini drafts a complaint letter with evidence
"""

import io
import json
import math
import os
import uuid
from datetime import datetime, timedelta

import folium
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="SmokeTrace", page_icon="🌫️", layout="wide")

REPORTS_FILE = "reports.json"
TRACE_HOURS = 24          # how far back we trace the air
CORRIDOR_KM = 30          # a fire counts as "upwind" if within this distance of the path
SEARCH_DEG = 5.0          # FIRMS search box half-size (~550 km)


def get_secret(name, default=""):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


GEMINI_KEY = get_secret("GEMINI_API_KEY")
FIRMS_KEY = get_secret("FIRMS_MAP_KEY")
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-flash-latest")

# ----------------------------------------------------------------------------
# Geo helpers
# ----------------------------------------------------------------------------
R_EARTH = 6371.0


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def move(lat, lon, bearing_deg, dist_km):
    """Move from (lat, lon) along a compass bearing by dist_km."""
    b = math.radians(bearing_deg)
    p1, l1 = math.radians(lat), math.radians(lon)
    d = dist_km / R_EARTH
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1),
                         math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), math.degrees(l2)


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def compass(deg):
    dirs = ["north", "north-east", "east", "south-east",
            "south", "south-west", "west", "north-west"]
    return dirs[int((deg + 22.5) // 45) % 8]


def dist_to_path(lat, lon, path):
    return min(haversine(lat, lon, p[0], p[1]) for p in path)


# ----------------------------------------------------------------------------
# Data fetchers (cached)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def geocode(city):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": city, "count": 1, "language": "en"}, timeout=15)
    r.raise_for_status()
    res = r.json().get("results") or []
    if not res:
        return None
    g = res[0]
    return {"name": g["name"], "country": g.get("country", ""),
            "admin": g.get("admin1", ""), "lat": g["latitude"], "lon": g["longitude"]}


@st.cache_data(ttl=1800, show_spinner=False)
def get_wind(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "kmh", "past_days": 1, "forecast_days": 1,
        "timezone": "auto"}, timeout=15)
    r.raise_for_status()
    j = r.json()
    df = pd.DataFrame({"time": pd.to_datetime(j["hourly"]["time"]),
                       "speed": j["hourly"]["wind_speed_10m"],
                       "dir": j["hourly"]["wind_direction_10m"]})
    now_local = datetime.utcnow() + timedelta(seconds=j.get("utc_offset_seconds", 0))
    return df.dropna(), now_local


@st.cache_data(ttl=1800, show_spinner=False)
def get_air(lat, lon):
    r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
        "latitude": lat, "longitude": lon, "hourly": "pm2_5,pm10,us_aqi",
        "past_days": 1, "forecast_days": 3, "timezone": "auto"}, timeout=15)
    r.raise_for_status()
    j = r.json()
    df = pd.DataFrame({"time": pd.to_datetime(j["hourly"]["time"]),
                       "pm2_5": j["hourly"]["pm2_5"],
                       "pm10": j["hourly"]["pm10"],
                       "aqi": j["hourly"]["us_aqi"]})
    now_local = datetime.utcnow() + timedelta(seconds=j.get("utc_offset_seconds", 0))
    return df, now_local


@st.cache_data(ttl=1800, show_spinner=False)
def get_fires(lat, lon, key, days=2):
    """NASA FIRMS active fires (VIIRS NOAA-20 + SNPP) in a box around the city."""
    w, s = round(lon - SEARCH_DEG, 3), round(lat - SEARCH_DEG, 3)
    e, n = round(lon + SEARCH_DEG, 3), round(lat + SEARCH_DEG, 3)
    frames = []
    for src in ["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT"]:
        url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
               f"{key}/{src}/{w},{s},{e},{n}/{days}")
        try:
            r = requests.get(url, timeout=30)
            if r.ok and r.text.startswith("latitude"):
                frames.append(pd.read_csv(io.StringIO(r.text)))
        except requests.RequestException:
            pass
    if not frames:
        return pd.DataFrame(columns=["latitude", "longitude", "frp", "acq_date", "acq_time"])
    df = pd.concat(frames, ignore_index=True)
    df["frp"] = pd.to_numeric(df.get("frp", 0), errors="coerce").fillna(0)
    return df


# ----------------------------------------------------------------------------
# Core logic: back-trajectory + upwind fire matching
# ----------------------------------------------------------------------------
def back_trajectory(lat, lon, wind_df, now_local, hours=TRACE_HOURS):
    """Walk backwards in time, one hour per step, moving *upwind*.
    Open-Meteo wind_direction = direction wind comes FROM, so the air parcel
    was upwind (in that direction) one hour earlier."""
    past = wind_df[wind_df["time"] <= now_local].sort_values("time", ascending=False).head(hours)
    path = [(lat, lon, 0)]
    cur_lat, cur_lon = lat, lon
    for i, row in enumerate(past.itertuples(), start=1):
        cur_lat, cur_lon = move(cur_lat, cur_lon, row.dir, row.speed * 1.0)
        path.append((cur_lat, cur_lon, i))
    return path


def match_fires(fires, path, city_lat, city_lon):
    if fires.empty:
        return fires.assign(dist_path=[], dist_city=[], upwind=[])
    f = fires.copy()
    f["dist_path"] = [dist_to_path(a, b, path) for a, b in zip(f.latitude, f.longitude)]
    f["dist_city"] = [haversine(city_lat, city_lon, a, b) for a, b in zip(f.latitude, f.longitude)]
    f["upwind"] = f["dist_path"] <= CORRIDOR_KM
    return f


def summarize_trace(f, city_lat, city_lon):
    up = f[f["upwind"]] if not f.empty else f
    if up.empty:
        return {"count": 0}
    w = up["frp"].clip(lower=1)
    c_lat = float((up.latitude * w).sum() / w.sum())
    c_lon = float((up.longitude * w).sum() / w.sum())
    return {
        "count": int(len(up)),
        "total_frp": float(up["frp"].sum()),
        "dist_km": haversine(city_lat, city_lon, c_lat, c_lon),
        "direction": compass(bearing(city_lat, city_lon, c_lat, c_lon)),
        "centroid": (c_lat, c_lon),
    }


# ----------------------------------------------------------------------------
# Air quality helpers
# ----------------------------------------------------------------------------
def pm_band(pm):
    """India NAAQS-style 24h PM2.5 bands (ug/m3), used as simple labels."""
    if pm is None or pd.isna(pm):
        return "Unknown", "#9e9e9e"
    if pm <= 30: return "Good", "#2e7d32"
    if pm <= 60: return "Satisfactory", "#9ccc65"
    if pm <= 90: return "Moderate", "#fbc02d"
    if pm <= 120: return "Poor", "#fb8c00"
    if pm <= 250: return "Very Poor", "#e53935"
    return "Severe", "#6a1b9a"


def danger_windows(air_df, now_local, threshold=90):
    fut = air_df[(air_df["time"] > now_local) &
                 (air_df["time"] <= now_local + timedelta(hours=48))].copy()
    fut["bad"] = fut["pm2_5"] >= threshold
    windows, start, peak = [], None, 0
    for row in fut.itertuples():
        if row.bad and start is None:
            start, peak = row.time, row.pm2_5
        elif row.bad:
            peak = max(peak, row.pm2_5)
        elif start is not None:
            windows.append((start, prev, peak)); start = None
        prev = row.time
    if start is not None:
        windows.append((start, prev, peak))
    return fut, windows


def fmt_window(w):
    s, e, p = w
    return f"{s:%a %d %b, %I %p} – {e:%I %p}  (peak PM2.5 ≈ {p:.0f} µg/m³)"


# ----------------------------------------------------------------------------
# Gemini helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def gemini_client(key):
    from google import genai
    return genai.Client(api_key=key)


FALLBACK_MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-flash-lite-latest",
                   "gemini-2.5-flash-lite"]


def gemini_text(prompt, image=None, as_json=False):
    """Call Gemini with retries + automatic fallback to other models when one is
    busy (503), rate-limited (429) or not available (404)."""
    if not GEMINI_KEY:
        return None
    import time
    from google.genai import types
    client = gemini_client(GEMINI_KEY)
    contents = [image, prompt] if image is not None else [prompt]
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json" if as_json else "text/plain",
        temperature=0.3)
    models = [GEMINI_MODEL] + [m for m in FALLBACK_MODELS if m != GEMINI_MODEL]
    last_err = None
    for model in models:
        for attempt in range(2):
            try:
                resp = client.models.generate_content(model=model, contents=contents, config=cfg)
                if resp.text:
                    return resp.text
            except Exception as ex:
                last_err = ex
                msg = str(ex)
                if any(c in msg for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                                          "overloaded", "high demand")):
                    time.sleep(1.5 * (attempt + 1))
                    continue          # retry same model once, then next model
                if "404" in msg or "NOT_FOUND" in msg or "not found" in msg.lower():
                    break             # model not available -> next model
                raise                 # real error (bad key etc.)
    raise RuntimeError("Gemini is busy right now. Please try again in a few seconds. "
                       f"({str(last_err)[:120]})")


PHOTO_PROMPT = """You are an air-pollution field inspector in India.
Look at this photo and identify the most likely pollution source.
Return ONLY JSON with these keys:
{
 "is_pollution": true/false,
 "source_type": one of ["Crop residue burning","Garbage / waste burning","Industrial chimney","Construction / road dust","Vehicle exhaust","Brick kiln","Forest / vegetation fire","Other","No visible pollution"],
 "confidence": 0-100,
 "severity": 1-5 (5 = thick dense smoke, immediate health risk),
 "evidence": "one sentence: what in the photo shows this",
 "health_risk": "one short sentence for nearby residents",
 "authority": "which Indian authority should act (e.g. State Pollution Control Board, Municipal Corporation, District Agriculture Officer)"
}"""


def load_reports():
    try:
        with open(REPORTS_FILE) as fh:
            return json.load(fh)
    except Exception:
        return []


def save_report(rep):
    reps = load_reports()
    reps.append(rep)
    with open(REPORTS_FILE, "w") as fh:
        json.dump(reps, fh)


SEV_COLOR = {1: "#9ccc65", 2: "#fbc02d", 3: "#fb8c00", 4: "#e53935", 5: "#6a1b9a"}

# Clean, minimal base map (Esri Light Gray Canvas: free, no API key)
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"


def base_map(lat, lon, zoom=7):
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=None,
                   control_scale=False, prefer_canvas=True)
    folium.TileLayer(ESRI.format("World_Light_Gray_Base"), attr="Esri",
                     name="Light", max_zoom=16).add_to(m)
    folium.TileLayer(ESRI.format("World_Light_Gray_Reference"), attr="Esri",
                     name="Labels", overlay=True, control=False, max_zoom=16).add_to(m)
    return m


def you_are_here(m, lat, lon):
    """Apple-style blue location dot with soft halo."""
    folium.CircleMarker((lat, lon), radius=18, stroke=False, fill=True,
                        fill_color="#0A84FF", fill_opacity=0.15).add_to(m)
    folium.CircleMarker((lat, lon), radius=7, color="white", weight=3, fill=True,
                        fill_color="#0A84FF", fill_opacity=1, tooltip="You").add_to(m)

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Segoe UI", sans-serif;}
.big {font-size: 1.05rem; line-height: 1.6;}
.card {padding: 1rem 1.25rem; border-radius: 16px; background: rgba(128,128,128,.08);
       border: 1px solid rgba(128,128,128,.12); margin-bottom: .75rem;}
.legend {font-size: .85rem; opacity: .8; margin-top: .25rem;}
iframe {border-radius: 16px !important; box-shadow: 0 4px 24px rgba(0,0,0,.08);}
div[data-testid="stMetric"] {background: rgba(128,128,128,.08); border-radius: 16px; padding: .75rem 1rem;}
.leaflet-control-attribution {font-size: 9px !important; opacity: .6;}
.leaflet-bar a {border-radius: 10px !important;}
</style>""", unsafe_allow_html=True)

st.title("🌫️ SmokeTrace")
st.caption("Who is polluting my air? — trace smoke to its source, report it, and act.")

with st.sidebar:
    st.header("📍 Your location")
    city = st.text_input("City", value=st.session_state.get("city", "Chandigarh"))
    lang = st.radio("Language for advice & complaint", ["English", "Hindi"], horizontal=True)
    st.divider()
    st.markdown("**Status**")
    st.write("Gemini:", "✅ connected" if GEMINI_KEY else "⚠️ add GEMINI_API_KEY")
    st.write("NASA FIRMS:", "✅ connected" if FIRMS_KEY else "⚠️ add FIRMS_MAP_KEY")
    st.caption("Data: Open-Meteo (wind, CAMS air quality), NASA FIRMS (VIIRS fires), Google Gemini.")

loc = None
if city:
    try:
        loc = geocode(city)
    except requests.RequestException as ex:
        st.error(f"Location lookup failed: {ex}")
if not loc:
    st.warning("Enter a valid city in the sidebar to begin.")
    st.stop()
st.session_state["city"] = city
LAT, LON = loc["lat"], loc["lon"]
st.markdown(f"**{loc['name']}**, {loc['admin']}, {loc['country']}  ·  {LAT:.3f}, {LON:.3f}")

# Current air snapshot
air_df, air_now = None, None
try:
    air_df, air_now = get_air(LAT, LON)
    cur = air_df[air_df["time"] <= air_now].tail(1)
    if not cur.empty:
        pm_now = float(cur["pm2_5"].iloc[0])
        band, col = pm_band(pm_now)
        c1, c2, c3 = st.columns(3)
        c1.metric("PM2.5 now (µg/m³)", f"{pm_now:.0f}")
        c2.markdown(f"<div class='card'>Air quality<br><b style='color:{col};font-size:1.4rem'>{band}</b></div>",
                    unsafe_allow_html=True)
        c3.metric("x WHO daily limit (15)", f"{pm_now / 15:.1f}×")
except requests.RequestException as ex:
    st.info(f"Air-quality data unavailable right now ({ex}).")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🧭 1. Trace it back", "📸 2. Photo check", "⏰ 3. 48h warning", "✉️ 4. Take action"])

# ---------------------------------------------------------------- TAB 1
with tab1:
    st.subheader("Where is my smoke coming from?")
    st.caption(f"We follow the wind backwards for {TRACE_HOURS} hours and look for satellite-detected "
               f"fires within {CORRIDOR_KM} km of that path.")
    if not FIRMS_KEY:
        st.warning("Add a free NASA FIRMS MAP_KEY to enable satellite fire tracing.")
    try:
        wind_df, w_now = get_wind(LAT, LON)
        path = back_trajectory(LAT, LON, wind_df, w_now)
        fires = get_fires(LAT, LON, FIRMS_KEY) if FIRMS_KEY else pd.DataFrame(
            columns=["latitude", "longitude", "frp", "acq_date", "acq_time"])
        mf = match_fires(fires, path, LAT, LON)
        summ = summarize_trace(mf, LAT, LON)
        st.session_state["trace"] = summ

        origin = path[-1]
        src_dir = compass(bearing(LAT, LON, origin[0], origin[1]))
        src_km = haversine(LAT, LON, origin[0], origin[1])

        if summ["count"] > 0:
            st.markdown(
                f"<div class='card big'>🔥 Your air today most likely passed over "
                f"<b>{summ['count']} active fires</b>, centred about "
                f"<b>{summ['dist_km']:.0f} km {summ['direction']}</b> of you "
                f"(total fire power {summ['total_frp']:.0f} MW).</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='card big'>🌬️ Over the last {TRACE_HOURS} h your air travelled "
                f"~{src_km:.0f} km from the <b>{src_dir}</b>. No satellite fires were found along "
                f"that path — local sources (traffic, garbage burning, dust, industry) are the likely "
                f"cause. Use <b>Photo check</b> to report them.</div>", unsafe_allow_html=True)

        m = base_map(LAT, LON)
        if not mf.empty:
            other = mf[~mf["upwind"]]
            up = mf[mf["upwind"]]
            for r in other.head(1500).itertuples():
                folium.CircleMarker((r.latitude, r.longitude), radius=2, stroke=False,
                                    fill=True, fill_color="#8E8E93", fill_opacity=0.35).add_to(m)
        # air path: soft glow + crisp line
        line = [(p[0], p[1]) for p in path]
        folium.PolyLine(line, color="#0A84FF", weight=10, opacity=0.15).add_to(m)
        folium.PolyLine(line, color="#0A84FF", weight=3, opacity=0.9,
                        tooltip="Air path (last 24 h)").add_to(m)
        for p in path[::6][1:]:
            folium.CircleMarker((p[0], p[1]), radius=4, color="#0A84FF", weight=2, fill=True,
                                fill_color="white", fill_opacity=1,
                                tooltip=f"{p[2]} h ago").add_to(m)
        if not mf.empty:
            for r in up.itertuples():
                rad = 4 + min(r.frp, 60) / 12
                folium.CircleMarker((r.latitude, r.longitude), radius=rad * 2.2, stroke=False,
                                    fill=True, fill_color="#FF453A", fill_opacity=0.15).add_to(m)
                folium.CircleMarker((r.latitude, r.longitude), radius=rad, color="white", weight=1.5,
                                    fill=True, fill_color="#FF453A", fill_opacity=0.95,
                                    tooltip=f"Fire · {r.frp:.0f} MW · {r.dist_city:.0f} km away · "
                                            f"{r.acq_date}").add_to(m)
        for rep in load_reports():
            folium.CircleMarker((rep["lat"], rep["lon"]), radius=6, color="white", weight=2,
                                fill=True, fill_color="#BF5AF2", fill_opacity=1,
                                tooltip=f"Citizen report: {rep['source_type']}").add_to(m)
        you_are_here(m, LAT, LON)
        pts = [(p[0], p[1]) for p in path]
        if not mf.empty and summ["count"] > 0:
            pts += list(zip(mf[mf["upwind"]].latitude, mf[mf["upwind"]].longitude))
        lats, lons = [p[0] for p in pts], [p[1] for p in pts]
        m.fit_bounds([[min(lats) - 0.3, min(lons) - 0.3], [max(lats) + 0.3, max(lons) + 0.3]])
        st_folium(m, height=480, use_container_width=True, returned_objects=[])
        st.markdown("<div class='legend'><span style='color:#0A84FF'>●</span> You &amp; air path &nbsp;&nbsp;"
                    "<span style='color:#FF453A'>●</span> Fires affecting you &nbsp;&nbsp;"
                    "<span style='color:#8E8E93'>●</span> Other fires &nbsp;&nbsp;"
                    "<span style='color:#BF5AF2'>●</span> Citizen reports</div>", unsafe_allow_html=True)
        if not mf.empty and summ["count"] > 0:
            with st.expander("Upwind fire list"):
                st.dataframe(mf[mf["upwind"]].sort_values("frp", ascending=False)[
                    ["latitude", "longitude", "frp", "acq_date", "acq_time", "dist_city"]].head(50),
                    width="stretch")
    except requests.RequestException as ex:
        st.error(f"Could not load wind/fire data: {ex}")

# ---------------------------------------------------------------- TAB 2
with tab2:
    st.subheader("Spotted smoke? Report it.")
    st.caption("Satellites miss small urban fires. Your photo fills the gap.")
    colA, colB = st.columns([1, 1])
    with colA:
        up_img = st.file_uploader("Upload a photo of smoke/dust", type=["jpg", "jpeg", "png", "webp"])
        cam = st.camera_input("…or take one now") if st.toggle("Use camera") else None
        img_file = up_img or cam
        rlat = st.number_input("Latitude of the smoke", value=float(LAT), format="%.5f")
        rlon = st.number_input("Longitude of the smoke", value=float(LON), format="%.5f")
        st.caption("Tip: long-press the spot in Google Maps to copy exact coordinates.")
    with colB:
        if img_file:
            img = Image.open(img_file).convert("RGB")
            img.thumbnail((1280, 1280))
            st.image(img, width="stretch")
            if st.button("🔍 Analyse with Gemini", type="primary", disabled=not GEMINI_KEY):
                with st.spinner("Gemini is inspecting the photo…"):
                    try:
                        res = json.loads(gemini_text(PHOTO_PROMPT, image=img, as_json=True))
                        st.session_state["photo_result"] = res
                        st.session_state["photo_loc"] = (rlat, rlon)
                        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=80)
                        st.session_state["photo_bytes"] = buf.getvalue()
                        if res.get("is_pollution", True):
                            save_report({"id": str(uuid.uuid4())[:8], "lat": rlat, "lon": rlon,
                                         "source_type": res.get("source_type"),
                                         "severity": res.get("severity"),
                                         "time": datetime.now().isoformat(timespec="minutes")})
                    except Exception as ex:
                        st.error(f"Gemini analysis failed: {ex}")
    res = st.session_state.get("photo_result")
    if res:
        sev = int(res.get("severity", 1) or 1)
        st.markdown(
            f"<div class='card big'><b>{res.get('source_type')}</b> "
            f"· confidence {res.get('confidence')}% · severity "
            f"<b style='color:{SEV_COLOR.get(sev, '#999')}'>{'●' * sev}{'○' * (5 - sev)}</b><br>"
            f"🔎 {res.get('evidence')}<br>❤️ {res.get('health_risk')}<br>"
            f"🏛️ Should act: <b>{res.get('authority')}</b></div>", unsafe_allow_html=True)
        st.success("Pinned to the community hotspot map. Go to **Take action** to file a complaint.")

    reps = load_reports()
    if reps:
        st.markdown(f"**Community hotspot map** — {len(reps)} report(s)")
        hm = base_map(LAT, LON, zoom=11)
        you_are_here(hm, LAT, LON)
        for r in reps:
            folium.CircleMarker((r["lat"], r["lon"]), radius=5 + 1.5 * int(r.get("severity") or 1),
                                color="white", weight=2, fill=True,
                                fill_color=SEV_COLOR.get(int(r.get("severity") or 1), "#999"),
                                fill_opacity=0.95,
                                tooltip=f"{r['source_type']} · sev {r.get('severity')} · {r['time']}").add_to(hm)
        st_folium(hm, height=380, use_container_width=True, returned_objects=[], key="hotspots")

# ---------------------------------------------------------------- TAB 3
with tab3:
    st.subheader("Next 48 hours: when is it dangerous?")
    if air_df is None or air_df[air_df["time"] > air_now].empty:
        st.info("Forecast unavailable right now.")
    else:
        fut, wins = danger_windows(air_df, air_now)
        chart = fut.set_index("time")[["pm2_5"]].rename(columns={"pm2_5": "PM2.5 (µg/m³)"})
        st.line_chart(chart, height=260)
        st.caption("Forecast: Copernicus CAMS via Open-Meteo. Danger threshold = 90 µg/m³ (India 'Poor').")
        if wins:
            st.error("⚠️ Danger windows:\n\n" + "\n\n".join("• " + fmt_window(w) for w in wins))
        else:
            st.success("No 'Poor' or worse hours forecast in the next 48 h.")

        best = fut.nsmallest(3, "pm2_5")
        worst = fut.nlargest(3, "pm2_5")
        if st.button("🗣️ Give me simple advice", type="primary"):
            summary = {
                "city": loc["name"],
                "danger_windows": [fmt_window(w) for w in wins],
                "cleanest_hours": [f"{t:%a %I %p} ({p:.0f})" for t, p in zip(best.time, best.pm2_5)],
                "worst_hours": [f"{t:%a %I %p} ({p:.0f})" for t, p in zip(worst.time, worst.pm2_5)],
                "upwind_fires": st.session_state.get("trace", {}).get("count", 0),
            }
            prompt = (f"You are a friendly public-health assistant. Using this 48-hour PM2.5 forecast "
                      f"summary: {json.dumps(summary)}\nWrite 5 short, practical bullet points in "
                      f"{lang} for a family: when to go outside/exercise, when to keep kids and elderly "
                      f"indoors, windows/ventilation, masks (N95), and one line on why (mention fires "
                      f"if any). Use specific times. Plain language, no jargon.")
            with st.spinner("Writing advice…"):
                txt = None
                try:
                    txt = gemini_text(prompt)
                except Exception as ex:
                    st.warning(f"Gemini unavailable ({ex}); showing basic advice.")
                if not txt:
                    b0 = best.iloc[0]
                    txt = (f"- Best time outdoors: **{b0.time:%a %I %p}** (PM2.5 ≈ {b0.pm2_5:.0f}).\n"
                           + ("".join(f"- Stay indoors / wear N95: {fmt_window(w)}\n" for w in wins)
                              or "- No danger windows forecast.\n")
                           + "- Keep windows closed during peaks; ventilate in the cleanest hours.")
                st.markdown(txt)

# ---------------------------------------------------------------- TAB 4
with tab4:
    st.subheader("Turn evidence into action")
    trace = st.session_state.get("trace", {})
    photo = st.session_state.get("photo_result")
    ploc = st.session_state.get("photo_loc", (LAT, LON))
    name = st.text_input("Your name (optional)", "")
    extra = st.text_area("Anything to add? (e.g. 'burning happens every night at 9 PM')", "")
    ev = {
        "city": f"{loc['name']}, {loc['admin']}",
        "report_time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "pm25_now": (None if air_df is None or air_df[air_df['time'] <= air_now].empty else round(float(
            air_df[air_df['time'] <= air_now].tail(1)['pm2_5'].iloc[0]), 1)),
        "satellite_upwind_fires_24h": trace.get("count", 0),
        "fire_cluster": (f"{trace['dist_km']:.0f} km {trace['direction']}" if trace.get("count") else None),
        "citizen_photo": photo,
        "photo_location": f"{ploc[0]:.5f}, {ploc[1]:.5f}",
        "citizen_note": extra,
    }
    with st.expander("Evidence that will be attached"):
        st.json(ev)
    if st.button("✍️ Draft my complaint", type="primary", disabled=not GEMINI_KEY):
        prompt = (f"Write a formal but concise complaint letter in {lang} to the appropriate Indian "
                  f"authority (use '{(photo or {}).get('authority', 'State Pollution Control Board')}') "
                  f"about an air-pollution incident. Sender: {name or 'A concerned resident'}. "
                  f"Evidence (JSON): {json.dumps(ev, default=str)}.\n"
                  f"Structure: Subject line; To; body with date/time, exact location (lat/long), "
                  f"what was observed, satellite + air-quality evidence with numbers, health impact, "
                  f"specific requested action and a 7-day response request; closing. "
                  f"Reference the Air (Prevention and Control of Pollution) Act, 1981. "
                  f"Do not invent facts that are not in the evidence.")
        with st.spinner("Drafting…"):
            try:
                st.session_state["letter"] = gemini_text(prompt)
            except Exception as ex:
                st.error(f"Gemini failed: {ex}")
    letter = st.session_state.get("letter")
    if letter:
        letter = st.text_area("Your complaint (edit freely)", letter, height=380)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download letter (.txt)", letter, file_name="smoketrace_complaint.txt")
        if st.session_state.get("photo_bytes"):
            c2.download_button("⬇️ Download photo evidence", st.session_state["photo_bytes"],
                               file_name="smoke_evidence.jpg", mime="image/jpeg")
        st.info("Send it via email to your State Pollution Control Board, or file on the CPCB "
                "'Sameer' app / your city's municipal grievance portal.")

st.divider()
st.caption("SmokeTrace · prototype for Build with AI: Code for Communities · Track 2 — Clean Air & "
           "Climate Resilience. Trajectory uses surface wind at your location — an approximation, "
           "not a full atmospheric model.")
