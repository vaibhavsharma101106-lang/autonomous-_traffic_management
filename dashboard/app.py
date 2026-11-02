"""
Streamlit Dashboard — AI Traffic Management System
IT 207 Human Intelligence Project

7-page interactive dashboard:
    1. Home / Overview
    2. Video Analysis
    3. Traffic Monitoring
    4. Prediction
    5. Signal Optimization
    6. Experiments
    7. About / Methodology
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import (
    CONFIG, init_database, get_traffic_history, create_analysis_job,
    update_analysis_job, get_analysis_jobs, get_analysis_job,
    classify_traffic_state, state_to_color_hex, insert_traffic_observation,
)
from src.analysis.traffic_analyser import generate_synthetic_time_series, TrafficAnalyser
from src.prediction.predictor import (
    XGBoostPredictor, LSTMPredictor, RandomForestPredictor,
    train_all_models, compare_models, evaluate_model,
)
from src.optimization.signal_controller import (
    FixedTimeController, RuleBasedController, AIAdaptiveController,
    LaneState, run_controller_comparison,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Traffic Management — IT 207",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1f77b4;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333;
    }
    .state-FREE      { color: #00c800; font-weight: bold; }
    .state-MODERATE  { color: #dcdc00; font-weight: bold; }
    .state-CONGESTED { color: #ff8c00; font-weight: bold; }
    .state-SEVERE    { color: #dc0000; font-weight: bold; }
    div[data-testid="stSidebar"] { background: #0f1117; }
</style>
""", unsafe_allow_html=True)


# ── Shared state & caching ────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    return init_database()


@st.cache_resource
def get_analyser():
    db = get_db()
    return TrafficAnalyser(db_conn=db)


@st.cache_data(ttl=30)
def load_traffic_history(lane_id=None, limit=500):
    db = get_db()
    return get_traffic_history(db, lane_id=lane_id, limit=limit)


@st.cache_resource
def get_models():
    """Load or train prediction models (cached for session)."""
    return {"xgb": None, "lstm": None, "rf": None, "trained": False}


@st.cache_resource(show_spinner="Loading YOLO11n weights (first run may download them)...")
def get_vehicle_detector(confidence=0.45, imgsz=640, roi_json="{}"):
    """Keep the expensive YOLO model loaded across Streamlit reruns."""
    from src.detection.detector import VehicleDetector
    return VehicleDetector(confidence=confidence, imgsz=imgsz,
                           roi_config=json.loads(roi_json) or None)


@st.cache_resource(show_spinner="Loading ByteTrack model...")
def get_vehicle_tracker(confidence=0.45, imgsz=640, roi_json="{}"):
    """Cache the tracking model so reruns do not reload its weights."""
    from src.tracking.tracker import VehicleTracker
    return VehicleTracker(confidence=confidence, imgsz=imgsz,
                          roi_config=json.loads(roi_json) or None)


def _ensure_synthetic_data():
    """Generate synthetic data if DB is empty."""
    db  = get_db()
    df  = get_traffic_history(db, limit=10)
    if df.empty:
        synth = generate_synthetic_time_series(n_hours=48, seed=42)
        for _, row in synth.iterrows():
            insert_traffic_observation(db, row.to_dict())
        st.cache_data.clear()


def _state_badge(state: str) -> str:
    colors = {
        "FREE":      ("#00c800", "black"),
        "MODERATE":  ("#dcdc00", "black"),
        "CONGESTED": ("#ff8c00", "black"),
        "SEVERE":    ("#dc0000", "white"),
    }
    bg, fg = colors.get(state, ("#888", "white"))
    return f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:4px;font-weight:bold;">{state}</span>'


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
PAGES = [
    "🏠 Home",
    "🎥 Video Analysis",
    "📊 Traffic Monitoring",
    "🔮 Prediction",
    "🚦 Signal Optimization",
    "🧪 Experiments",
    "ℹ️ About & Methodology",
    "🗂️ Analysis History",
]

with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1f77b4/white?text=AI+Traffic", width=200)
    st.markdown("### Navigation")
    page = st.radio("", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**IT 207 Human Intelligence**")
    st.markdown("*Human-Inspired AI for*  \n*Traffic Management*")
    st.markdown(f"🕐 {datetime.now().strftime('%H:%M:%S')}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.markdown('<div class="main-header">🚦 AI Traffic Management System</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Human-Inspired AI for Intelligent Traffic Management and Urban Congestion Reduction — IT 207</div>',
                unsafe_allow_html=True)

    # KPI Cards
    _ensure_synthetic_data()
    df_all = load_traffic_history(limit=200)

    col1, col2, col3, col4 = st.columns(4)
    total_veh = int(df_all["vehicle_count"].sum()) if not df_all.empty else 0
    avg_dens  = round(df_all["density"].mean() * 100, 1) if not df_all.empty else 0
    latest_state = df_all["traffic_state"].iloc[-1] if not df_all.empty else "UNKNOWN"
    n_records    = len(df_all)

    col1.metric("📹 Observations Recorded", f"{n_records:,}")
    col2.metric("🚗 Total Vehicles Counted", f"{total_veh:,}")
    col3.metric("📈 Avg. Density", f"{avg_dens:.1f}%")
    col4.metric("🔴 Latest State", latest_state)

    st.markdown("---")

    # System architecture overview
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("System Architecture")
        st.markdown("""
        ```
        Traffic Video / Historical Data
                ↓
          YOLO11n Vehicle Detection
                ↓
          ByteTrack Vehicle Tracking
                ↓
          Lane-wise Traffic Analysis ──→ SQLite DB
                ↓
          Congestion Prediction
          (RF / XGBoost / LSTM)
                ↓
          Adaptive Signal Optimization
                ↓
          SUMO Simulation & Evaluation
                ↓
          Streamlit Dashboard
        ```
        """)

    with col_r:
        st.subheader("Human Cognitive Inspirations")
        st.markdown("""
        | Cognitive Concept | AI Implementation |
        |---|---|
        | 🎯 **Selective Attention** | ROI-based lane filtering — focus on relevant regions |
        | 🧠 **Working Memory** | Rolling traffic history buffer for temporal context |
        | 📚 **Learning** | ML training on historical patterns (RF / XGBoost / LSTM) |
        | ⚖️ **Decision-Making** | Multi-criteria signal optimization with fairness |
        """)

    # Technology stack
    st.subheader("Technology Stack")
    cols = st.columns(6)
    techs = [
        ("🤖 YOLO11n", "Vehicle\nDetection"),
        ("👁️ ByteTrack", "Multi-Object\nTracking"),
        ("📊 XGBoost", "Congestion\nPrediction"),
        ("🧬 LSTM", "Deep Learning\nForecast"),
        ("🚥 SUMO", "Traffic\nSimulation"),
        ("⚡ FastAPI", "Backend\nAPI"),
    ]
    for col, (name, desc) in zip(cols, techs):
        col.markdown(f"**{name}**\n\n{desc}")

    # Dataset info
    st.subheader("Datasets")
    st.markdown("""
    | Dataset | Type | Status | Notes |
    |---|---|---|---|
    | **UA-DETRAC** | Real traffic video | External — download script provided | 100 annotated sequences |
    | **AI City Challenge** | Vehicle counting | External — requires registration | Historical tracking data |
    | **SUMO Synthetic** | Simulated traffic | ✅ Generated locally | 4 scenarios × 3600s |
    | **Synthetic Time-Series** | Generated | ✅ Available | 48h, 2-lane, realistic patterns |
    """)

    # Quick traffic chart
    if not df_all.empty:
        st.subheader("Recent Traffic Activity")
        df_plot = df_all.copy()
        df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
        fig = px.line(df_plot, x="timestamp", y="vehicle_count", color="lane_id",
                      title="Vehicle Count Over Time",
                      color_discrete_map={"lane1": "#1f77b4", "lane2": "#ff7f0e"})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: VIDEO ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.markdown("## 🎥 Video Analysis")
    st.markdown("Upload a traffic video to run YOLO11n vehicle detection and ByteTrack tracking.")

    uploaded = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])
    limit_mb = int(os.getenv("TRAFFIC_MAX_UPLOAD_MB", "500"))
    sample_rate, max_frames, enable_track = st.columns(3)
    sample_rate = sample_rate.slider("Frame sample interval", 1, 30, 5)
    max_frames = max_frames.slider("Maximum sampled frames", 1, 500, 100)
    enable_track = enable_track.checkbox("Enable ByteTrack", value=True)
    confidence = st.slider("Detection confidence", 0.05, 0.95, 0.45, 0.05)
    inference_size = st.select_slider("Model input size", options=[320, 480, 640, 960, 1280], value=640)
    default_roi_json = json.dumps(CONFIG["analysis"].get("roi_default", {}), indent=2)
    roi_text = st.text_area("Lane ROI polygons (JSON coordinates in the resized video frame)",
                            value=default_roi_json, height=120)

    if uploaded is not None:
        st.caption(f"{uploaded.name} · {uploaded.size / 1024 / 1024:.1f} MB (limit {limit_mb} MB)")
        if uploaded.size > limit_mb * 1024 * 1024:
            st.error(f"Upload exceeds the configured {limit_mb} MB limit.")
        else:
            fingerprint = f"{uploaded.file_id}:{uploaded.name}:{uploaded.size}"
            if st.session_state.get("upload_fingerprint") != fingerprint:
                from src.detection.detector import validate_video_file
                job_id = uuid.uuid4().hex
                upload_dir = ROOT / "data" / "raw" / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                video_path = upload_dir / f"{job_id}{Path(uploaded.name).suffix.lower()}"
                with video_path.open("wb") as destination:
                    for chunk in uploaded:
                        destination.write(chunk)
                try:
                    metadata = validate_video_file(str(video_path), limit_mb * 1024 * 1024)
                    create_analysis_job(get_db(), job_id, Path(uploaded.name).name,
                                        str(video_path), uploaded.size, metadata)
                    st.session_state["upload_fingerprint"] = fingerprint
                    st.session_state["analysis_job_id"] = job_id
                except Exception as exc:
                    video_path.unlink(missing_ok=True)
                    st.error(f"Video validation failed: {exc}")

            try:
                roi_config = json.loads(roi_text)
                if not isinstance(roi_config, dict) or any(
                    not isinstance(points, list) or len(points) < 3 or
                    any(not isinstance(point, list) or len(point) != 2 or
                        not all(isinstance(v, (int, float)) for v in point) for point in points)
                    for points in roi_config.values()
                ):
                    raise ValueError("Each lane must map to a polygon with at least three [x, y] points.")
            except (json.JSONDecodeError, ValueError) as exc:
                roi_config = None
                st.error(f"Invalid lane ROI JSON: {exc}")
            job_id = st.session_state.get("analysis_job_id") if roi_config is not None else None
            job = get_analysis_job(get_db(), job_id) if job_id else None
            if job and job["status"] == "completed":
                st.success(f"Validated upload ready · {job['source_width']}×{job['source_height']} · {job['duration_seconds']:.1f}s")
            if job and st.button("🔍 Analyse video", type="primary", disabled=job["status"] == "running"):
                output_path = ROOT / "data" / "processed" / f"{job_id}_annotated.mp4"
                progress = st.progress(0.0, text="Starting video analysis…")
                update_analysis_job(get_db(), job_id, status="running", sample_rate=sample_rate,
                                    max_frames=max_frames, confidence=confidence,
                                    inference_size=inference_size, output_path=str(output_path), error=None)
                try:
                    from src.detection.detector import process_video_frames
                    roi_json = json.dumps(roi_config, sort_keys=True)
                    detector = get_vehicle_detector(confidence, inference_size, roi_json)
                    tracker = get_vehicle_tracker(confidence, inference_size, roi_json) if enable_track else None
                    detector.reset_stats()
                    analyser = get_analyser()
                    analysed_lane_series, totals = {}, {}
                    def on_progress(done, expected):
                        progress.progress(min(done / max(expected, 1), 1.0), text=f"Analyzed {done} of up to {expected} sampled frames")
                    results = process_video_frames(str(job["video_path"]), detector,
                        sample_rate=sample_rate, max_frames=max_frames,
                        output_path=str(output_path), tracker=tracker,
                        max_inference_dim=1280, progress_callback=on_progress)
                    detection_count = tracking_count = 0
                    for result in results:
                        detections = result.get("detections", [])
                        detection_count += len(detections)
                        tracking_count += result.get("track_count", 0)
                        analysis = analyser.analyse(result["lane_counts"], result["class_totals"],
                                                    source="video", analysis_job_id=job_id)
                        for lane, metric in analysis["lane_metrics"].items():
                            analysed_lane_series.setdefault(lane, []).append({"timestamp_s": result["timestamp_s"], "count": metric["vehicle_count"]})
                        for cls, count in result["class_totals"].items():
                            totals[cls] = totals.get(cls, 0) + int(count)
                    metrics = {"frames": len(results), "mean_latency_ms": detector.avg_latency_ms,
                               "mean_sampled_vehicles_per_frame": detection_count / max(len(results), 1),
                               "tracking_enabled": bool(tracker)}
                    update_analysis_job(get_db(), job_id, status="completed", processed_frames=len(results),
                                        detection_count=detection_count, tracking_count=tracking_count,
                                        class_totals_json=totals, metrics_json=metrics)
                    progress.progress(1.0, text="Analysis complete")
                    if results:
                        st.video(str(output_path))
                        st.success(f"Processed {len(results)} sampled frames. Vehicles detected: {detection_count}; tracked: {tracking_count}.")
                        st.json({"vehicle_classes": totals, "metrics": metrics})
                        for lane, series in analysed_lane_series.items():
                            st.plotly_chart(px.line(pd.DataFrame(series), x="timestamp_s", y="count", title=f"{lane} — vehicles per sampled frame"), use_container_width=True)
                    else:
                        st.warning("No frames were processed. Check the video decoder and sample settings.")
                except Exception as exc:
                    update_analysis_job(get_db(), job_id, status="failed", error=str(exc))
                    st.error(f"Video analysis failed: {exc}")
    else:
        # Demo with synthetic data
        st.info("👆 Upload a traffic video, or view the demo below using synthetic data.")
        demo_data = generate_synthetic_time_series(n_hours=1, seed=7)
        demo_lane1 = demo_data[demo_data["lane_id"] == "lane1"].head(60)
        fig = px.area(demo_lane1, x="timestamp", y="vehicle_count",
                      title="Demo: Synthetic Traffic Video Analysis",
                      color_discrete_sequence=["#1f77b4"])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: TRAFFIC MONITORING
# ═════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.markdown("## 📊 Traffic Monitoring")
    _ensure_synthetic_data()

    refresh = st.checkbox("Auto-refresh every 5 seconds", value=False)

    df = load_traffic_history(limit=300)
    if df.empty:
        st.warning("No data. Generating synthetic data...")
        synth = generate_synthetic_time_series(n_hours=24, seed=42)
        df    = synth
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Lane filter
    lanes_avail = df["lane_id"].unique().tolist()
    sel_lanes   = st.multiselect("Filter Lanes", lanes_avail, default=lanes_avail)
    df_filt     = df[df["lane_id"].isin(sel_lanes)]

    # KPI metrics per lane
    st.subheader("Current Lane Status")
    lane_cols = st.columns(len(sel_lanes))
    for col, lane in zip(lane_cols, sel_lanes):
        latest = df_filt[df_filt["lane_id"] == lane].iloc[-1] if not df_filt[df_filt["lane_id"] == lane].empty else None
        if latest is not None:
            with col:
                state = latest["traffic_state"]
                color = state_to_color_hex(state)
                col.markdown(
                    f"""
                    <div style="background:#1e1e2e;border-radius:10px;padding:12px;text-align:center;border:2px solid {color};">
                        <h4>{lane.upper()}</h4>
                        <div style="color:{color};font-size:1.4rem;font-weight:bold;">{state}</div>
                        <div>🚗 {int(latest['vehicle_count'])} vehicles</div>
                        <div>📏 Queue: {latest['queue_length']:.0f} m</div>
                        <div>💨 Density: {latest['density']*100:.0f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # Time-series charts
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚗 Vehicle Count", "📈 Density", "🔴 Traffic State", "🗂️ Composition"
    ])

    with tab1:
        fig = px.line(df_filt, x="timestamp", y="vehicle_count", color="lane_id",
                      title="Lane-wise Vehicle Count Over Time",
                      labels={"vehicle_count": "Vehicles", "timestamp": "Time"})
        fig.update_layout(height=380, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = px.area(df_filt, x="timestamp", y="density", color="lane_id",
                       title="Traffic Density [0–1]",
                       labels={"density": "Density", "timestamp": "Time"})
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        # Heatmap of traffic state
        state_map  = {"FREE": 0, "MODERATE": 1, "CONGESTED": 2, "SEVERE": 3}
        df_filt2   = df_filt.copy()
        df_filt2["state_num"] = df_filt2["traffic_state"].map(state_map)
        pivot = df_filt2.pivot_table(
            index=df_filt2["timestamp"].dt.floor("30min"),
            columns="lane_id",
            values="state_num",
            aggfunc="mean"
        )
        fig3 = px.imshow(pivot.T,
                          color_continuous_scale=["#00c800", "#dcdc00", "#ff8c00", "#dc0000"],
                          labels=dict(color="State (0=FREE,3=SEVERE)"),
                          title="Traffic State Heatmap (30-min intervals)")
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        # Stacked bar of vehicle classes
        class_cols = ["cars", "buses", "trucks", "motorcycles"]
        available  = [c for c in class_cols if c in df_filt.columns]
        if available:
            df_cls = df_filt.groupby("lane_id")[available].sum().reset_index()
            df_melt = df_cls.melt(id_vars="lane_id", var_name="Class", value_name="Count")
            fig4 = px.bar(df_melt, x="lane_id", y="Count", color="Class",
                          title="Vehicle Class Composition per Lane",
                          barmode="stack")
            fig4.update_layout(height=350)
            st.plotly_chart(fig4, use_container_width=True)

    # Raw data table
    with st.expander("📋 Raw Data Table"):
        st.dataframe(df_filt.tail(100), use_container_width=True)

    if refresh:
        time.sleep(5)
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4: PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.markdown("## 🔮 Congestion Prediction")
    st.markdown("""
    Predicts lane traffic demand **5 minutes ahead** using Random Forest,
    XGBoost, and LSTM models trained on historical observations.
    """)

    _ensure_synthetic_data()
    models_state = get_models()

    col_a, col_b = st.columns([2, 1])
    with col_b:
        st.subheader("Model Training")
        n_hours_train = st.slider("Training data (hours)", 12, 168, 72)
        if st.button("🏋️ Train All Models", type="primary"):
            with st.spinner("Training RF, XGBoost, and LSTM models..."):
                try:
                    db  = get_db()
                    df  = get_traffic_history(db, limit=20000)
                    if len(df) < 100:
                        df = generate_synthetic_time_series(n_hours=n_hours_train, seed=42)

                    df_lane = df[df["lane_id"] == "lane1"].reset_index(drop=True)
                    n  = len(df_lane)
                    n_tr = int(n * 0.70)
                    n_va = int(n * 0.15)
                    train_df = df_lane.iloc[:n_tr]
                    val_df   = df_lane.iloc[n_tr:n_tr+n_va]
                    test_df  = df_lane.iloc[n_tr+n_va:]

                    results = train_all_models(train_df, val_df, test_df)
                    st.session_state["trained_models"] = results
                    st.session_state["test_df"]        = test_df
                    st.success("✅ All models trained!")

                except Exception as e:
                    st.error(f"Training failed: {e}")

    with col_a:
        st.subheader("Model Comparison (Test Set)")
        if "trained_models" in st.session_state:
            comp = st.session_state["trained_models"].get("test_comparison")
            if comp is not None and not comp.empty:
                st.dataframe(
                    comp.style.format("{:.4f}").highlight_min(
                        subset=["mae", "rmse"], color="#1a472a"
                    ).highlight_max(
                        subset=["r2"], color="#1a472a"
                    ),
                    use_container_width=True
                )

                fig_bar = px.bar(
                    comp.reset_index(), x="model",
                    y=["mae", "rmse"],
                    barmode="group", title="MAE and RMSE by Model",
                    color_discrete_map={"mae": "#1f77b4", "rmse": "#ff7f0e"}
                )
                fig_bar.update_layout(height=280)
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Click 'Train All Models' to generate comparison metrics.")

    st.markdown("---")
    st.subheader("Forecast Visualisation")

    if "trained_models" in st.session_state and "test_df" in st.session_state:
        results   = st.session_state["trained_models"]
        test_df   = st.session_state["test_df"]
        y_true    = test_df["vehicle_count"].values

        model_choice = st.selectbox("Select Model",
                                     ["XGBoost", "RandomForest", "LSTM"])
        model_key    = {"XGBoost": "XGBoost",
                         "RandomForest": "RandomForest",
                         "LSTM": "LSTM"}[model_choice]

        try:
            mdl      = results[model_key]["model"]
            y_pred   = mdl.predict(test_df.copy())
            n        = min(len(y_true), len(y_pred), 200)

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                y=y_true[:n], name="Actual",
                line=dict(color="#1f77b4"), mode="lines"
            ))
            fig_fc.add_trace(go.Scatter(
                y=y_pred[:n], name="Predicted",
                line=dict(color="#ff7f0e", dash="dash"), mode="lines"
            ))
            fig_fc.update_layout(
                title=f"{model_choice} — Actual vs. Predicted Vehicle Count",
                xaxis_title="Time Step", yaxis_title="Vehicles",
                height=380, hovermode="x unified"
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Error distribution
            errors = y_true[:n] - y_pred[:n]
            fig_err = px.histogram(errors, nbins=30,
                                    title="Prediction Error Distribution",
                                    labels={"value": "Error (vehicles)"},
                                    color_discrete_sequence=["#9467bd"])
            fig_err.update_layout(height=240)
            st.plotly_chart(fig_err, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction visualisation failed: {e}")
    else:
        # Show demo forecast
        st.info("Train models above to see actual forecast. Showing demo:")
        np.random.seed(7)
        t    = np.arange(60)
        true = 10 + 5 * np.sin(t / 5) + np.random.normal(0, 1.5, 60)
        pred = true + np.random.normal(0, 2, 60)
        fig_demo = go.Figure()
        fig_demo.add_trace(go.Scatter(y=true, name="Actual", line=dict(color="#1f77b4")))
        fig_demo.add_trace(go.Scatter(y=pred, name="Predicted (demo)",
                                       line=dict(color="#ff7f0e", dash="dash")))
        fig_demo.update_layout(title="Demo Forecast (Train models for real results)",
                                height=300)
        st.plotly_chart(fig_demo, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5: SIGNAL OPTIMIZATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.markdown("## 🚦 Adaptive Signal Optimization")
    st.markdown("""
    Computes optimal green durations using three controllers.
    Inspired by human **Decision-Making**: evaluate multiple criteria
    (demand, waiting time, fairness) before allocating signal time.
    """)

    st.subheader("Current Lane Configuration")
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    lane_data = {}
    for col, ln in zip([col_l1, col_l2, col_l3, col_l4],
                        ["Lane 1 (North)", "Lane 2 (South)",
                         "Lane 3 (East)", "Lane 4 (West)"]):
        with col:
            lid = ln.split()[1].lower() + "_" + ln.split("(")[1][:-1].lower()
            # Simplified lane id
            lid = f"lane{['1','2','3','4'][['Lane 1', 'Lane 2', 'Lane 3', 'Lane 4'].index(ln.split(' (')[0])]}"
            count = col.slider(f"🚗 {ln} vehicles", 0, 40, 10, key=f"cnt_{lid}")
            wait  = col.slider(f"⏱️ {ln} wait (s)", 0, 180, 30, key=f"wait_{lid}")
            queue = col.slider(f"📏 {ln} queue (m)", 0.0, 200.0, float(count*5), key=f"q_{lid}")
            lane_data[lid] = {"count": count, "wait": wait, "queue": queue}

    ctrl_type = st.radio(
        "Controller",
        ["Fixed-Time", "Rule-Based Adaptive", "AI-Assisted Adaptive"],
        horizontal=True
    )

    if st.button("⚡ Compute Optimal Signal Timings", type="primary"):
        lanes = list(lane_data)
        ctrl_map = {
            "Fixed-Time":           FixedTimeController(lanes),
            "Rule-Based Adaptive":  RuleBasedController(lanes),
            "AI-Assisted Adaptive": AIAdaptiveController(lanes),
        }
        controller = ctrl_map[ctrl_type]

        lane_states = {
            lid: LaneState(
                lane_id=lid,
                vehicle_count=d["count"],
                waiting_time=d["wait"],
                queue_length=d["queue"],
            )
            for lid, d in lane_data.items()
        }

        cycle = controller.compute(lane_states)

        st.success(f"✅ {ctrl_type} — Cycle #{cycle.cycle_number} "
                   f"(total {cycle.total_cycle_time:.1f}s)")

        # Visual signal display
        st.subheader("Signal Allocations")
        sig_cols = st.columns(len(lanes))
        total_green = sum(cycle.assignments.values())

        for col, lane_id in zip(sig_cols, lanes):
            green = cycle.assignments.get(lane_id, 0)
            pct   = green / max(total_green, 1) * 100
            col.markdown(
                f"""
                <div style="background:#111;border-radius:10px;padding:12px;text-align:center;">
                    <div style="font-size:2.5rem;">🟢</div>
                    <div style="font-weight:bold;">{lane_id.upper()}</div>
                    <div style="font-size:1.5rem;color:#00c800;">{green:.1f}s</div>
                    <div style="color:#aaa;">{pct:.0f}% of cycle</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Bar chart
        fig_sig = px.bar(
            x=list(cycle.assignments.keys()),
            y=list(cycle.assignments.values()),
            color=list(cycle.assignments.values()),
            color_continuous_scale=["#dc0000", "#dcdc00", "#00c800"],
            labels={"x": "Lane", "y": "Green Duration (s)"},
            title=f"{ctrl_type} — Green Duration Allocation",
        )
        fig_sig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_sig, use_container_width=True)

        # Reasoning
        if cycle.reasoning:
            with st.expander("📋 Controller Reasoning"):
                for lane, reason in cycle.reasoning.items():
                    st.markdown(f"**{lane}**: {reason}")

    st.markdown("---")
    st.subheader("Controller Comparison")
    vph_compare = st.slider("Vehicle arrival rate (vph)", 100, 1500, 600)
    n_cyc       = st.slider("Simulation cycles", 20, 200, 100)

    if st.button("🔬 Compare All Controllers"):
        with st.spinner("Running comparison simulation..."):
            lanes    = ["lane1", "lane2", "lane3", "lane4"]
            arrivals = {ln: vph_compare / 3600.0 for ln in lanes}
            results  = run_controller_comparison(lanes, arrivals, n_cyc)

            df_comp = pd.DataFrame([
                {
                    "Controller":       name.replace("_", " ").title(),
                    "Avg Wait (s)":     r["avg_waiting_time"],
                    "Avg Queue (m)":    r["avg_queue_length"],
                    "Throughput":       r["throughput"],
                    "Congestion Rate":  r["congestion_rate"],
                }
                for name, r in results.items()
            ])

            st.dataframe(df_comp.set_index("Controller"), use_container_width=True)

            fig_cmp = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Average Waiting Time (s)", "Vehicle Throughput"]
            )
            fig_cmp.add_trace(
                go.Bar(x=df_comp["Controller"], y=df_comp["Avg Wait (s)"],
                       marker_color=["#dc0000", "#ff8c00", "#00c800"],
                       name="Avg Wait"),
                row=1, col=1
            )
            fig_cmp.add_trace(
                go.Bar(x=df_comp["Controller"], y=df_comp["Throughput"],
                       marker_color=["#dc0000", "#ff8c00", "#00c800"],
                       name="Throughput"),
                row=1, col=2
            )
            fig_cmp.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Congestion reduction vs fixed
            if "fixed" in results and "ai_assisted" in results:
                reduction = (
                    (results["fixed"]["avg_waiting_time"] -
                     results["ai_assisted"]["avg_waiting_time"])
                    / max(results["fixed"]["avg_waiting_time"], 0.01)
                    * 100
                )
                st.metric(
                    "Congestion Reduction (AI vs Fixed)",
                    f"{reduction:.1f}%",
                    delta=f"-{reduction:.1f}% waiting time"
                )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6: EXPERIMENTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    st.markdown("## 🧪 Experiments & Results")
    st.markdown("Controlled comparison of controllers across four traffic scenarios.")

    # Run experiments button
    if st.button("▶️ Run Full Experiment Battery (SUMO / Fallback)", type="primary"):
        with st.spinner("Running experiments... (this may take 2-5 minutes)"):
            try:
                from src.simulation.sumo_runner import run_sumo_experiments
                results_dir = str(ROOT / "experiments" / "results")
                Path(results_dir).mkdir(parents=True, exist_ok=True)
                df_exp = run_sumo_experiments(results_dir)
                st.session_state["exp_results"] = df_exp
                st.success(f"✅ Experiments complete — {len(df_exp)} results")
            except Exception as e:
                # Fallback: run software-only simulation
                st.warning(f"SUMO unavailable ({e}). Running software simulation...")
                scenarios = CONFIG["experiments"]["scenarios"]
                rows = []
                for sc in scenarios:
                    lanes = ["lane1", "lane2", "lane3", "lane4"]
                    base_rate = sc["vehicle_rate_vph"] / 3600.0
                    # Uneven demand for relevant scenario
                    if "uneven" in sc["name"]:
                        arrivals = {"lane1": base_rate * 1.8, "lane2": base_rate * 0.5,
                                    "lane3": base_rate * 1.2, "lane4": base_rate * 0.4}
                    else:
                        arrivals = {ln: base_rate for ln in lanes}

                    ctrlrs = {
                        "fixed":       FixedTimeController(lanes),
                        "rule_based":  RuleBasedController(lanes),
                        "ai_assisted": AIAdaptiveController(lanes),
                    }
                    for cname, ctrl in ctrlrs.items():
                        from src.optimization.signal_controller import SignalSimulator
                        sim     = SignalSimulator(lanes, ctrl, arrival_rates=arrivals)
                        metrics = sim.run(n_cycles=200)
                        rows.append({
                            "scenario":          sc["name"],
                            "controller":        cname,
                            "avg_waiting_time":  metrics["avg_waiting_time"],
                            "avg_queue_length":  metrics["avg_queue_length"],
                            "throughput":        metrics["throughput"],
                            "congestion_rate":   metrics["congestion_rate"],
                            "source":            "software_simulation",
                        })
                df_exp = pd.DataFrame(rows)
                csv_path = ROOT / "experiments" / "results" / "experiment_results.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df_exp.to_csv(csv_path, index=False)
                st.session_state["exp_results"] = df_exp
                st.success(f"✅ Software simulation complete — {len(df_exp)} results")

    # Display results
    if "exp_results" in st.session_state:
        df_exp = st.session_state["exp_results"]
        st.subheader("Experiment Results")
        st.dataframe(df_exp, use_container_width=True)

        # Waiting time chart
        st.subheader("Average Waiting Time by Controller and Scenario")
        fig_exp1 = px.bar(
            df_exp, x="scenario", y="avg_waiting_time", color="controller",
            barmode="group",
            title="Average Waiting Time (seconds)",
            color_discrete_map={
                "fixed":       "#dc0000",
                "rule_based":  "#ff8c00",
                "ai_assisted": "#00c800",
            }
        )
        fig_exp1.update_layout(height=380)
        st.plotly_chart(fig_exp1, use_container_width=True)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            fig_exp2 = px.bar(
                df_exp, x="scenario", y="throughput", color="controller",
                barmode="group", title="Vehicle Throughput",
                color_discrete_map={
                    "fixed":       "#dc0000",
                    "rule_based":  "#ff8c00",
                    "ai_assisted": "#00c800",
                }
            )
            fig_exp2.update_layout(height=320)
            st.plotly_chart(fig_exp2, use_container_width=True)

        with col_e2:
            fig_exp3 = px.bar(
                df_exp, x="scenario", y="avg_queue_length", color="controller",
                barmode="group", title="Average Queue Length (m)",
                color_discrete_map={
                    "fixed":       "#dc0000",
                    "rule_based":  "#ff8c00",
                    "ai_assisted": "#00c800",
                }
            )
            fig_exp3.update_layout(height=320)
            st.plotly_chart(fig_exp3, use_container_width=True)

        # Congestion reduction summary
        st.subheader("Congestion Reduction vs Fixed-Time Baseline")
        scenarios_list = df_exp["scenario"].unique()
        reduction_rows = []
        for sc in scenarios_list:
            df_sc = df_exp[df_exp["scenario"] == sc]
            fixed_wait = df_sc[df_sc["controller"] == "fixed"]["avg_waiting_time"]
            if fixed_wait.empty:
                continue
            fixed_wait = fixed_wait.iloc[0]
            for ctrl in ["rule_based", "ai_assisted"]:
                ctrl_wait = df_sc[df_sc["controller"] == ctrl]["avg_waiting_time"]
                if ctrl_wait.empty:
                    continue
                ctrl_wait = ctrl_wait.iloc[0]
                reduction = (fixed_wait - ctrl_wait) / max(fixed_wait, 0.01) * 100
                reduction_rows.append({
                    "Scenario":    sc,
                    "Controller":  ctrl.replace("_", " ").title(),
                    "Reduction (%)": round(reduction, 1),
                })
        if reduction_rows:
            df_red = pd.DataFrame(reduction_rows)
            fig_red = px.bar(df_red, x="Scenario", y="Reduction (%)",
                              color="Controller", barmode="group",
                              title="Waiting Time Reduction vs Fixed-Time (%)",
                              color_discrete_map={
                                  "Rule Based": "#ff8c00",
                                  "Ai Assisted": "#00c800",
                              })
            fig_red.add_hline(y=0, line_dash="dash", line_color="white")
            fig_red.update_layout(height=320)
            st.plotly_chart(fig_red, use_container_width=True)
    else:
        st.info("Click 'Run Full Experiment Battery' to generate results.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7: ABOUT & METHODOLOGY
# ═════════════════════════════════════════════════════════════════════════════
elif page == PAGES[7]:
    st.markdown("## 🗂️ Recorded Video Analysis History")
    jobs = get_analysis_jobs(get_db(), limit=200)
    if not jobs:
        st.info("No uploaded-video analysis jobs have been recorded yet.")
    for job in jobs:
        with st.expander(f"{job['filename']} · {job['status']} · {job['created_at']}"):
            st.write(f"Job ID: `{job['job_id']}`")
            st.write(f"Source: {job['source_width']}×{job['source_height']} · {job['duration_seconds'] or 0:.1f} seconds")
            if job["error"]:
                st.error(job["error"])
            if job["status"] == "completed":
                st.json({"classes": job["class_totals"], "metrics": job["metrics"]})
                output = Path(job["output_path"] or "")
                if output.is_file():
                    st.video(str(output))
                    st.download_button("Download annotated video", output.read_bytes(),
                                       file_name=f"{job['job_id']}_annotated.mp4", mime="video/mp4",
                                       key=f"video-{job['job_id']}")
elif page == PAGES[6]:
    st.markdown("## ℹ️ About & Methodology")

    tabs = st.tabs([
        "Human Intelligence Theory",
        "System Architecture",
        "AI Methodology",
        "Limitations & Ethics",
    ])

    with tabs[0]:
        st.markdown("""
        ## Human Intelligence Theory — IT 207

        This project demonstrates how four human cognitive processes inspire
        computational intelligence in a traffic management system.

        ### A. Selective Attention
        > *"The taking possession by the mind, in clear and vivid form, of one out of what seem
        > several simultaneously possible objects or trains of thought."*
        > — William James, 1890 (cited in Broadbent, 1958)

        **Human process:** Humans do not process all visual information equally.
        The brain filters irrelevant stimuli and focuses resources on task-relevant regions.

        **AI implementation:**
        - Configurable ROI polygons define traffic lanes as 'attention windows'
        - YOLO detections outside ROI regions are discarded
        - Reduces false positives from pedestrians, signage, and parked vehicles
        - Computational resources focused on relevant spatial regions

        ---

        ### B. Working Memory
        > *"A system for temporarily holding and manipulating information while carrying out
        > complex cognitive tasks."*
        > — Baddeley (2000), Working Memory and Language

        **Human process:** Short-term retention of recent sensory experience
        to support ongoing decision-making.

        **AI implementation:**
        - Rolling `MemoryBuffer` retains the last N traffic observations per lane
        - Lag features (t-1 through t-12) encode recent vehicle counts
        - Rolling mean/std/max encode pattern statistics over recent windows
        - The LSTM hidden state functions as a compressed temporal memory
        - The SQLite database serves as an extended long-term memory store

        ---

        ### C. Learning
        > *"The relatively permanent change in behavior or knowledge that results from
        > experience."* — General cognitive psychology literature

        **Human process:** Humans learn traffic patterns from repeated daily experience,
        building expectations about peak hours and congestion patterns.

        **AI implementation:**
        - Random Forest: ensemble of decision trees trained on feature-engineered time-series
        - XGBoost: gradient-boosted trees with temporal lag and rolling features
        - LSTM: sequence model learning long-range temporal dependencies
        - Time-aware train/val/test splits prevent data leakage

        ---

        ### D. Decision-Making
        > *"Decision making is a process of choosing from among a number of options."*
        > — Beach & Mitchell (1987)

        **Human process:** A traffic officer evaluates multiple competing demands
        (lane volume, waiting vehicles, fairness) before deciding which lane to release.

        **AI implementation:**
        - AI-Assisted controller computes weighted multi-criteria scores
        - Weights: demand (50%), waiting urgency (20%), fairness (30%)
        - Hard constraints: min/max green duration, starvation prevention
        - Predictions from the LSTM inform future demand
        - Three controllers (fixed / rule-based / AI) offer interpretable comparison
        """)

    with tabs[1]:
        st.markdown("""
        ## System Architecture

        ```
        ┌─────────────────────────────────────────────────────┐
        │              Traffic Video / Dataset                 │
        └────────────────────────┬────────────────────────────┘
                                 │
                    ┌────────────▼──────────────┐
                    │   Video Preprocessing      │
                    │   (OpenCV frame sampling)   │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────▼───────────────┐
                    │   YOLO11n Vehicle Detection │
                    │   + ROI Selective Attention │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────▼───────────────┐
                    │  ByteTrack Vehicle Tracking │
                    │  (Persistent IDs + Trails)  │
                    └────────────┬───────────────┘
                                 │
               ┌─────────────────▼────────────────────┐
               │       TrafficAnalyser                 │
               │  Lane Counting · Density Estimation   │
               │  Queue Length · State Classification  │
               │  MemoryBuffer (Working Memory)        │
               └──────┬──────────────┬────────────────┘
                      │              │
             ┌────────▼────┐  ┌──────▼──────────────────┐
             │  SQLite DB  │  │    Prediction Module      │
             │  (History)  │  │  RF · XGBoost · LSTM     │
             └─────────────┘  └──────────┬───────────────┘
                                         │
                    ┌────────────────────▼───────────────┐
                    │     Signal Optimization             │
                    │  Fixed / Rule-Based / AI-Assisted   │
                    └────────────────────┬───────────────┘
                                         │
                    ┌────────────────────▼───────────────┐
                    │  SUMO Simulation (TraCI) / Fallback │
                    └────────────────────┬───────────────┘
                                         │
                    ┌────────────────────▼───────────────┐
                    │   FastAPI Backend + Streamlit UI    │
                    └────────────────────────────────────┘
        ```
        """)

    with tabs[2]:
        st.markdown("""
        ## AI Methodology

        ### Vehicle Detection
        - **Model**: Ultralytics YOLO11n (pretrained on COCO)
        - **Classes detected**: car (2), motorcycle (3), bus (5), truck (7)
        - **Inference**: FP16 on CUDA, FP32 CPU fallback
        - **Confidence threshold**: 0.45, IoU: 0.50

        ### Vehicle Tracking
        - **Algorithm**: ByteTrack (Byte-level association)
        - **Mechanism**: Maintains two detection pools (high/low confidence)
          to avoid losing occluded vehicles
        - **Trajectory**: Circular deque (30 positions) per track ID
        - **Unique counting**: Set-based deduplication per lane

        ### Congestion Prediction
        | Model | Type | Inputs | Horizon |
        |---|---|---|---|
        | Random Forest | Ensemble | Tabular lag/rolling features | 5 min |
        | XGBoost | Gradient Boosted | Tabular lag/rolling features | 5 min |
        | LSTM | Deep Learning | Sequence of 12 steps | 5 min |

        - **Feature engineering**: 12 lags, rolling mean/std/max (3,6,12 windows),
          hour-of-day, day-of-week, peak indicator, first and second differences
        - **Splits**: 70% train / 15% val / 15% test (time-ordered, no leakage)
        - **LSTM training**: Adam optimizer, early stopping (patience=15), gradient clipping

        ### Signal Optimization
        | Controller | Description | Complexity |
        |---|---|---|
        | Fixed-Time | Equal green for all lanes | O(1) |
        | Rule-Based | Proportional to vehicle count + waiting time | O(lanes) |
        | AI-Assisted | Multi-criteria weighted score + prediction | O(lanes) |

        All controllers enforce:
        - Minimum green duration: 10s (safety)
        - Maximum green duration: 90s (fairness)
        - Yellow phase: 3s
        - All-red clearance: 2s
        - Starvation prevention: forced priority after N missed cycles
        """)

    with tabs[3]:
        st.markdown("""
        ## Limitations and Ethical Considerations

        ### Technical Limitations
        1. **Camera calibration**: Speed estimates are heuristic — not physical measurements
           without proper camera calibration and world-coordinate mapping.
        2. **Dataset access**: UA-DETRAC and AI City Challenge datasets require registration.
           Results on external datasets may differ from synthetic data results.
        3. **Simulation fidelity**: SUMO simulations use simplified network models.
           Real-world performance may differ significantly.
        4. **YOLO detection bias**: YOLO11n is trained on COCO, which may have uneven
           representation of vehicle types by size, colour, and viewpoint.
        5. **Poor lighting**: Detection accuracy degrades in night, rain, and fog.
        6. **Occlusion**: Heavily occluded vehicles may be missed or double-counted.
        7. **Prototype scope**: This is a research prototype, not a deployable traffic
           control system.

        ### Ethical Considerations

        **Privacy**
        - Traffic cameras record continuous video from public spaces
        - This system does NOT store identifiable information (faces, plates)
        - Vehicle bounding boxes are processed in-memory and discarded
        - Recommendation: anonymise video data before storage in production

        **Fairness**
        - Detection models may perform differently across vehicle types
        - Signal optimization includes fairness constraints to prevent
          systematic disadvantage to low-volume lanes
        - Uneven detection accuracy could bias green allocation

        **Data Licensing**
        - UA-DETRAC: academic research use only
        - SUMO: Eclipse Public Licence 2.0
        - YOLO11n weights: AGPL-3.0 / Ultralytics commercial licence

        **Disclaimer**
        - This system demonstrates AI-inspired concepts from human cognition
        - It does NOT claim to reproduce the biological human brain
        - It does NOT measure or quantify human intelligence
        - Congestion reduction figures are derived from simulation only
          and should not be extrapolated to real-world traffic

        ### References
        - Broadbent, D.E. (1958). Perception and Communication. Pergamon Press.
        - Baddeley, A.D. (2000). The episodic buffer: A new component of working memory. *Trends in Cognitive Sciences*, 4(11), 417–423.
        - Jung, R.E., & Haier, R.J. (2007). The Parieto-Frontal Integration Theory (P-FIT) of intelligence. *Brain and Behavioral Sciences*, 30(2), 135–154.
        - Kahneman, D., & Tversky, A. (1979). Prospect theory. *Econometrica*, 47(2), 263–292.
        - Wang, Z., et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. *ECCV 2022*.
        - SUMO documentation: https://sumo.dlr.de/docs/
        - Ultralytics YOLO: https://docs.ultralytics.com/
        """)

    st.markdown("---")
    st.markdown("""
    **IT 207 Human Intelligence Project** | AI Traffic Management System v1.0.0  
    Built with: Python 3.11 · YOLO11n · ByteTrack · XGBoost · LSTM · SUMO · FastAPI · Streamlit
    """)
