"""Streamlit dashboard.

    CROPIQ_CONFIG=configs/default.yaml python -m streamlit run src/cropiq/dashboard/app.py
"""
from __future__ import annotations

import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from cropiq.config import load_config
from cropiq.data.sensors import FEATURES, load_sensor_csv, simulate_field
from cropiq.pipeline.inference import CropAdvisor, ModelUnavailable

st.set_page_config(page_title="Crop Health & Irrigation Intelligence", page_icon="🌿", layout="wide")
CONFIG_PATH = os.environ.get("CROPIQ_CONFIG", "configs/default.yaml")

# ----------------------------------------------------------------------------- design tokens
INK, MUTED, LINE, CARD, BG = "#17261E", "#5B6B62", "#E3E9E4", "#FFFFFF", "#F4F7F4"
GREEN, GREEN_DK = "#1F7A4D", "#14573A"
LEVEL = {  # risk level -> (accent, soft background, label)
    "low": ("#2E7D32", "#EAF5EC", "Low risk"),
    "moderate": ("#B7791F", "#FBF3E0", "Moderate risk"),
    "high": ("#C2570C", "#FCEEE2", "High risk"),
    "critical": ("#B42318", "#FDECEA", "Critical risk"),
}
ACTION = {"wait": ("#2E7D32", "Hold irrigation"), "irrigate": ("#1D6FB8", "Irrigate now"), "alert": ("#B42318", "Sensor / water alert")}
SENSORS = {  # feature -> (label, unit, colour)
    "soil_moisture": ("Soil moisture", "%", "#1D6FB8"),
    "temperature": ("Temperature", "°C", "#C2570C"),
    "humidity": ("Humidity", "%", "#2A9D8F"),
}

CSS = f"""
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }}
html, body, [class*="css"] {{ font-family: "Inter", "Segoe UI", system-ui, sans-serif; color: {INK}; }}

.hero {{ background: linear-gradient(120deg, {GREEN_DK} 0%, {GREEN} 60%, #3E9B6B 100%); border-radius: 16px;
         padding: 28px 32px; color: #fff; margin-bottom: 1.4rem; box-shadow: 0 6px 22px rgba(20,87,58,.22); }}
.hero h1 {{ font-size: 1.85rem; font-weight: 700; margin: 0 0 4px 0; letter-spacing: -.01em; color: #fff; }}
.hero p {{ margin: 0; font-size: .98rem; opacity: .88; }}
.hero .tags {{ margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }}
.hero .tag {{ font-size: .74rem; padding: 3px 10px; border-radius: 999px; background: rgba(255,255,255,.16);
              border: 1px solid rgba(255,255,255,.28); }}

.step {{ display: flex; align-items: center; gap: 10px; margin: 0 0 .7rem 0; }}
.step .n {{ width: 26px; height: 26px; border-radius: 50%; background: {GREEN}; color: #fff; font-weight: 700;
            font-size: .85rem; display: flex; align-items: center; justify-content: center; }}
.step .t {{ font-size: 1.08rem; font-weight: 650; color: {INK}; }}
.step .s {{ font-size: .82rem; color: {MUTED}; margin-left: 4px; }}

[data-testid="stVerticalBlockBorderWrapper"] {{ background: {CARD}; border: 1px solid {LINE} !important;
        border-radius: 14px !important; box-shadow: 0 1px 2px rgba(16,40,28,.04); }}

.kpi {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; padding: 12px 14px; }}
.kpi .l {{ font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; color: {MUTED}; font-weight: 600; }}
.kpi .v {{ font-size: 1.55rem; font-weight: 700; line-height: 1.25; }}
.kpi .u {{ font-size: .85rem; color: {MUTED}; font-weight: 500; margin-left: 2px; }}
.kpi .d {{ font-size: .78rem; color: {MUTED}; }}

.banner {{ border-radius: 14px; padding: 18px 22px; border: 1px solid; border-left-width: 6px; }}
.banner .lvl {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; font-weight: 700; }}
.banner .msg {{ font-size: 1.2rem; font-weight: 650; margin-top: 4px; color: {INK}; line-height: 1.35; }}

.rec {{ display: flex; gap: 12px; padding: 11px 14px; background: {CARD}; border: 1px solid {LINE};
        border-radius: 10px; margin-bottom: 8px; font-size: .95rem; line-height: 1.45; }}
.rec .i {{ flex: 0 0 24px; height: 24px; border-radius: 6px; background: #E7F3EC; color: {GREEN_DK};
           font-weight: 700; font-size: .8rem; display: flex; align-items: center; justify-content: center; }}

.pill {{ display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .78rem; font-weight: 650;
         border: 1px solid; }}
.big {{ font-size: 1.55rem; font-weight: 700; line-height: 1.2; }}
.sub {{ font-size: .82rem; color: {MUTED}; }}
.meter {{ height: 9px; border-radius: 999px; background: #E8EEE9; overflow: hidden; margin: 6px 0 4px; }}
.meter > div {{ height: 100%; border-radius: 999px; }}
.disclaimer {{ font-size: .78rem; color: {MUTED}; margin-top: 10px; }}

.chip {{ display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; background: {CARD};
         border: 1px solid {LINE}; border-radius: 10px; margin-bottom: 8px; font-size: .9rem; font-weight: 550; }}
.dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 8px; }}

div.stButton > button[kind="primary"] {{ background: {GREEN}; border: 0; border-radius: 10px; padding: .65rem 1.4rem;
        font-weight: 650; box-shadow: 0 2px 8px rgba(31,122,77,.28); }}
div.stButton > button[kind="primary"]:hover {{ background: {GREEN_DK}; }}
[data-testid="stSidebar"] {{ background: #EDF2EE; border-right: 1px solid {LINE}; }}
[data-testid="stFileUploader"] section {{ border: 1.5px dashed #B9C9BE; background: #FAFCFA; border-radius: 12px; }}
</style>
"""


def title(n: int, text: str, sub: str = "") -> None:
    st.markdown(f"<div class='step'><div class='n'>{n}</div><div class='t'>{text}</div><div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def kpi(label: str, value: str, unit: str = "", note: str = "") -> str:
    return f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}<span class='u'>{unit}</span></div><div class='d'>{note}</div></div>"


def pill(text: str, colour: str) -> str:
    return f"<span class='pill' style='color:{colour};border-color:{colour}55;background:{colour}14'>{text}</span>"


def meter(frac: float, colour: str) -> str:
    return f"<div class='meter'><div style='width:{max(0.0, min(1.0, frac)) * 100:.0f}%;background:{colour}'></div></div>"


def pretty(name: str) -> str:
    return name.split("___")[-1].replace("_", " ").strip().capitalize()


def style_chart(chart: alt.Chart) -> alt.Chart:
    return (chart.configure_view(strokeWidth=0).configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#EEF2EF",
            domainColor=LINE, tickColor=LINE, labelFontSize=11).configure(background="transparent"))


def sensor_charts(df: pd.DataFrame) -> None:
    hours = np.arange(len(df)) - len(df) + 1
    cols = st.columns(3)
    for col, key in zip(cols, FEATURES):
        label, unit, colour = SENSORS[key]
        s = df[key].to_numpy()
        delta = s[-1] - s[0]
        with col:
            st.markdown(kpi(label, f"{s[-1]:.1f}", unit, f"{delta:+.1f}{unit} over {len(df)} h"), unsafe_allow_html=True)
            d = pd.DataFrame({"hour": hours, "value": s})
            base = alt.Chart(d).encode(x=alt.X("hour:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
                                       y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)))
            area = base.mark_area(color=colour, opacity=0.14, line={"color": colour, "strokeWidth": 2}, interpolate="monotone")
            st.altair_chart(style_chart(area.properties(height=110)), use_container_width=True)


def prob_bars(items: pd.DataFrame, colour: str, height: int = 150) -> alt.Chart:
    ch = (alt.Chart(items).mark_bar(cornerRadiusEnd=4, color=colour, size=16)
          .encode(y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                  x=alt.X("p:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                  tooltip=["label", alt.Tooltip("p:Q", format=".1%")]))
    return style_chart(ch.properties(height=height))


@st.cache_resource(show_spinner="Loading models...")
def get_advisor() -> CropAdvisor:
    return CropAdvisor(load_config(CONFIG_PATH))


def sensor_input(cfg) -> pd.DataFrame | None:
    mode = st.radio("Source", ["Simulated field", "Upload CSV", "Manual"], horizontal=True, label_visibility="collapsed")
    window = cfg.sensors.window_hours
    if mode == "Simulated field":
        c1, c2 = st.columns([1, 2])
        seed = c1.number_input("Field seed", 0, 9999, 7)
        day = c2.slider("Day of the season", 3, cfg.sensors.days_per_field - 1, 40)
        df = simulate_field(int(seed), cfg)
        end = day * 24
        return df.iloc[end - window : end][FEATURES].reset_index(drop=True)
    if mode == "Upload CSV":
        up = st.file_uploader("CSV with soil_moisture, temperature, humidity (>= 24 hourly rows)", type="csv")
        if up is None:
            return None
        return load_sensor_csv(up)[FEATURES].tail(window).reset_index(drop=True)
    c1, c2, c3 = st.columns(3)
    sm = c1.slider("Soil moisture (%)", 0.0, 60.0, 32.0)
    t = c2.slider("Temperature (°C)", 5.0, 45.0, 29.0)
    rh = c3.slider("Humidity (%)", 10.0, 100.0, 70.0)
    return pd.DataFrame({"soil_moisture": [sm] * window, "temperature": [t] * window, "humidity": [rh] * window})


def render_advisory(result: dict) -> None:
    adv = result["advisory"]
    accent, soft, label = LEVEL.get(adv["risk_level"], LEVEL["moderate"])
    st.markdown(f"<div class='banner' style='background:{soft};border-color:{accent}55;border-left-color:{accent}'>"
                f"<div class='lvl' style='color:{accent}'>{label}</div><div class='msg'>{adv['summary']}</div></div>",
                unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("##### Recommended actions")
        for i, rec in enumerate(adv["recommendations"], 1):
            st.markdown(f"<div class='rec'><div class='i'>{i}</div><div>{rec}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='disclaimer'>{adv['disclaimer']}</div>", unsafe_allow_html=True)

    ia, ir = result.get("image_analysis"), result.get("irrigation")
    with right:
        if ia:
            with st.container(border=True):
                st.markdown("##### Leaf analysis")
                sev = ia["severity"]
                sev_col = {"none": "#2E7D32", "mild": "#B7791F", "moderate": "#C2570C", "severe": "#B42318"}.get(sev, MUTED)
                st.markdown(f"<div class='big'>{pretty(ia['disease'])}</div>"
                            f"<div class='sub'>Top prediction &middot; {ia['confidence']:.0%} calibrated confidence</div>"
                            + meter(ia["confidence"], GREEN)
                            + f"<div style='margin-top:6px'>{pill('Severity: ' + sev, sev_col)}</div>", unsafe_allow_html=True)
                if ia.get("anomaly_score") is not None:
                    ratio = ia["anomaly_score"] / max(ia["anomaly_threshold"], 1e-9)
                    flag = ia.get("anomaly_flag", False)
                    col = "#B42318" if flag else "#2E7D32"
                    st.markdown(f"<div class='sub' style='margin-top:14px'>Anomaly screen (healthy-leaf VAE)</div>"
                                + meter(min(1.0, ratio / 2), col)
                                + f"<div class='sub'>score {ia['anomaly_score']:.4f} &middot; alert above {ia['anomaly_threshold']:.4f} &middot; "
                                  + pill("Atypical leaf" if flag else "Within normal range", col) + "</div>", unsafe_allow_html=True)
        if ir:
            with st.container(border=True):
                st.markdown("##### Irrigation decision")
                col, text = ACTION.get(ir["action"], (INK, ir["action"]))
                st.markdown(f"<div class='big' style='color:{col}'>{text}</div>"
                            f"<div class='sub'>Spiking network on the last 24 h of sensor data</div>", unsafe_allow_html=True)

    if ia or ir:
        st.write("")
        c1, c2 = st.columns(2, gap="large")
        if ia:
            with c1, st.container(border=True):
                st.markdown("##### Disease probabilities")
                d = pd.DataFrame(ia["top_k"]).rename(columns={"probability": "p"})
                d["label"] = d["disease"].map(pretty)
                st.altair_chart(prob_bars(d, GREEN), use_container_width=True)
        if ir:
            with c2, st.container(border=True):
                st.markdown("##### Action probabilities")
                d = pd.DataFrame({"label": [a.capitalize() for a in ir["probabilities"]], "p": list(ir["probabilities"].values())})
                st.altair_chart(prob_bars(d, "#1D6FB8"), use_container_width=True)
                spikes = ir.get("spikes_per_decision", {})
                if spikes:
                    st.caption("Mean spike rate per layer: " + " · ".join(f"{k} {v:.3f}" for k, v in spikes.items()))


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    cfg = load_config(CONFIG_PATH)
    advisor = get_advisor()
    avail = advisor.available()

    st.markdown("<div class='hero'><h1>Crop Health &amp; Irrigation Intelligence</h1>"
                "<p>Leaf disease screening and sensor-driven irrigation advice, fused into one advisory.</p>"
                "<div class='tags'><span class='tag'>ResNet classifier</span><span class='tag'>Diffusion-augmented training</span>"
                "<span class='tag'>VAE anomaly screen</span><span class='tag'>Spiking neural controller</span></div></div>",
                unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Model status")
        for name, nice in (("classifier", "Disease classifier"), ("vae", "Anomaly screen (VAE)"), ("snn", "Irrigation SNN")):
            ok = avail[name]
            st.markdown(f"<div class='chip'><span><span class='dot' style='background:{'#2E7D32' if ok else '#B42318'}'></span>{nice}</span>"
                        f"<span style='color:{MUTED};font-weight:500'>{'Ready' if ok else 'Missing'}</span></div>", unsafe_allow_html=True)
        for name, err in avail["errors"].items():
            st.caption(f"{name}: {err}")
        st.markdown("---")
        st.caption("Decision support only. Not a diagnosis; consult an agronomist for treatment decisions.")

    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        title(1, "Leaf photo", "single leaf, plain background works best")
        up = st.file_uploader("Upload a single-leaf photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        image = Image.open(up) if up else None
        if image is not None:
            st.image(image, use_container_width=True)
    with right, st.container(border=True):
        title(2, "Field sensors", "hourly, last 24 h")
        readings = sensor_input(cfg)
        if readings is not None:
            sensor_charts(readings)

    st.write("")
    go = st.button("Get advisory", type="primary", disabled=image is None and readings is None)
    if go:
        try:
            with st.spinner("Analysing..."):
                result = advisor.advise(image, readings)
        except (ModelUnavailable, ValueError) as exc:
            st.error(str(exc))
            return
        st.write("")
        render_advisory(result)


if __name__ == "__main__":
    main()
