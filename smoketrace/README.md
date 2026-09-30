# 🌫️ SmokeTrace — *Who is polluting my air?*

**Build with AI: Code for Communities — Track 2: Clean Air & Climate Resilience**

AQI apps tell you *how bad* the air is. SmokeTrace tells you **where it's coming from**, **when it will get worse**, and **helps you do something about it**.

## Features
| # | Feature | How it works |
|---|---------|--------------|
| 1 | **Trace it back** | Follows the wind backwards 24 h (Open-Meteo hourly wind) and matches the path with NASA FIRMS VIIRS satellite fire detections within 30 km → "Your smoke is coming from 38 fires, ~160 km north-west." |
| 2 | **Photo check** | Citizens upload/take a photo. **Gemini (multimodal)** classifies the source (crop burning, garbage fire, chimney, dust, kiln…), rates severity 1–5 and names the responsible authority. Reports are pinned on a shared hotspot map, catching small urban fires that satellites miss. |
| 3 | **48 h warning** | CAMS PM2.5 forecast (via Open-Meteo) → danger windows plus plain-language family advice from Gemini in English or Hindi. |
| 4 | **One-tap action** | Gemini drafts a formal complaint (citing the Air Act, 1981) with photo, GPS, time, satellite and PM2.5 evidence, ready to download and send. |

## Tech stack
Python · Streamlit · Google Gemini (`google-genai`) · Folium · Open-Meteo (weather + air quality) · NASA FIRMS

## Run locally
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your keys
streamlit run app.py
```
Keys:
- Gemini: https://aistudio.google.com/apikey
- NASA FIRMS MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/

## Impact
- Over 1 billion people in South Asia breathe air above WHO limits. North India's winter smog is driven partly by fires hundreds of km away.
- Pollution sources become **visible and accountable**, and citizens become a **distributed sensor network**.
- Designed as a **Digital Public Good**: open data, open source and multilingual, so any BRICS city can use it by typing its name.

## Limitations & roadmap
- The trajectory uses surface wind at the user's location (an approximation). Next: HYSPLIT / multi-level winds along the path.
- Reports are stored in a local JSON file. Next: Firestore/BigQuery for a persistent shared map.
- Next: WhatsApp bot intake, more Indian languages, direct filing to CPCB/SPCB portals, a dashboard for authorities.
