"""
AI Traffic Management System — Core Utilities
IT 207 Human Intelligence Project

Provides shared configuration, database access, feature engineering,
video utilities, and traffic state classification.
"""

import os
import time
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger


# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Module-level config
CONFIG = load_config()


# ── Traffic state ─────────────────────────────────────────────────────────────
TRAFFIC_STATES = ["FREE", "MODERATE", "CONGESTED", "SEVERE"]
STATE_COLORS_BGR = {
    "FREE":      (0, 200, 0),
    "MODERATE":  (0, 220, 220),
    "CONGESTED": (0, 140, 255),
    "SEVERE":    (0, 0, 220),
}
STATE_COLORS_HEX = {
    "FREE":      "#00c800",
    "MODERATE":  "#00dcdc",
    "CONGESTED": "#ff8c00",
    "SEVERE":    "#dc0000",
}


def classify_traffic_state(vehicle_count: int,
                            thresholds: Optional[Dict] = None) -> str:
    """
    Classify traffic state based on vehicle count.

    Human Cognition Connection (Decision-Making):
        Like a traffic officer who evaluates lane volume and makes a categorical
        judgement (clear / busy / jammed), this function maps quantitative
        observations to discrete states used downstream for decisions.
    """
    if thresholds is None:
        thresholds = CONFIG["analysis"]["density_thresholds"]
    if vehicle_count <= thresholds["free"]:
        return "FREE"
    elif vehicle_count <= thresholds["moderate"]:
        return "MODERATE"
    elif vehicle_count <= thresholds["congested"]:
        return "CONGESTED"
    return "SEVERE"


def state_to_color_bgr(state: str) -> Tuple[int, int, int]:
    """Return an OpenCV BGR colour for a traffic state."""
    return STATE_COLORS_BGR.get(state, (128, 128, 128))


def state_to_color_hex(state: str) -> str:
    """Return a hex colour string for Plotly charts."""
    return STATE_COLORS_HEX.get(state, "#888888")


# ── Database ──────────────────────────────────────────────────────────────────
def get_db_path() -> Path:
    """Absolute path to the SQLite database."""
    return PROJECT_ROOT / CONFIG["paths"]["database"]


def init_database(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Create database tables if they do not already exist.

    Human Cognition Connection (Working Memory):
        The database serves as the system's external working memory — persisting
        recent traffic observations so that prediction and optimisation modules
        can retrieve contextual history, mirroring how humans retain short-term
        situational awareness while making decisions.
    """
    if db_path is None:
        db_path = get_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS traffic_observations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_job_id TEXT,
            timestamp       TEXT    NOT NULL,
            source          TEXT    NOT NULL DEFAULT 'video',
            lane_id         TEXT    NOT NULL,
            vehicle_count   INTEGER NOT NULL DEFAULT 0,
            arrival_count   INTEGER NOT NULL DEFAULT 0,
            tracking_enabled INTEGER NOT NULL DEFAULT 0,
            density         REAL    NOT NULL DEFAULT 0.0,
            queue_length    REAL    NOT NULL DEFAULT 0.0,
            traffic_state   TEXT    NOT NULL DEFAULT 'FREE',
            cars            INTEGER DEFAULT 0,
            buses           INTEGER DEFAULT 0,
            trucks          INTEGER DEFAULT 0,
            motorcycles     INTEGER DEFAULT 0,
            avg_speed       REAL    DEFAULT 0.0,
            created_at      TEXT    DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS signal_decisions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            controller_type TEXT    NOT NULL,
            lane_id         TEXT    NOT NULL,
            green_duration  REAL    NOT NULL,
            waiting_time    REAL    NOT NULL DEFAULT 0.0,
            queue_length    REAL    NOT NULL DEFAULT 0.0,
            vehicle_count   INTEGER NOT NULL DEFAULT 0,
            cycle_number    INTEGER DEFAULT 0,
            created_at      TEXT    DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            model_name       TEXT    NOT NULL,
            lane_id          TEXT    NOT NULL,
            horizon_steps    INTEGER NOT NULL,
            predicted_count  REAL    NOT NULL,
            predicted_state  TEXT    NOT NULL,
            actual_count     REAL,
            mae              REAL,
            created_at       TEXT    DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS experiment_results (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name     TEXT    NOT NULL,
            scenario            TEXT    NOT NULL,
            controller_type     TEXT    NOT NULL,
            avg_waiting_time    REAL,
            avg_queue_length    REAL,
            throughput          REAL,
            avg_travel_time     REAL,
            congestion_reduction REAL,
            duration_seconds    INTEGER,
            created_at          TEXT    DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS analysis_jobs (
            job_id              TEXT PRIMARY KEY,
            filename            TEXT NOT NULL,
            video_path          TEXT NOT NULL,
            output_path         TEXT,
            status              TEXT NOT NULL DEFAULT 'uploaded',
            error               TEXT,
            sample_rate         INTEGER,
            max_frames          INTEGER,
            confidence          REAL,
            inference_size      INTEGER,
            file_size_bytes     INTEGER,
            duration_seconds    REAL,
            source_fps          REAL,
            source_width        INTEGER,
            source_height       INTEGER,
            processed_frames    INTEGER DEFAULT 0,
            detection_count     INTEGER DEFAULT 0,
            tracking_count      INTEGER DEFAULT 0,
            class_totals_json   TEXT DEFAULT '{}',
            metrics_json        TEXT DEFAULT '{}',
            created_at          TEXT NOT NULL,
            updated_at          TEXT NOT NULL
        );
    """)
    observation_columns = {
        row[1] for row in cur.execute("PRAGMA table_info(traffic_observations)")
    }
    if "analysis_job_id" not in observation_columns:
        cur.execute("ALTER TABLE traffic_observations ADD COLUMN analysis_job_id TEXT")
    if "arrival_count" not in observation_columns:
        cur.execute("ALTER TABLE traffic_observations ADD COLUMN arrival_count INTEGER NOT NULL DEFAULT 0")
    if "tracking_enabled" not in observation_columns:
        cur.execute("ALTER TABLE traffic_observations ADD COLUMN tracking_enabled INTEGER NOT NULL DEFAULT 0")
    conn.commit()
    return conn


def insert_traffic_observation(conn: sqlite3.Connection,
                                data: Dict[str, Any]) -> int:
    """Insert a single traffic observation row."""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO traffic_observations
        (analysis_job_id, timestamp, source, lane_id, vehicle_count, arrival_count, tracking_enabled, density, queue_length,
         traffic_state, cars, buses, trucks, motorcycles, avg_speed)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data.get("analysis_job_id"),
        data.get("timestamp", datetime.now().isoformat()),
        data.get("source", "video"),
        data.get("lane_id", "lane1"),
        data.get("vehicle_count", 0),
        data.get("arrival_count", 0),
        int(bool(data.get("tracking_enabled", False))),
        data.get("density", 0.0),
        data.get("queue_length", 0.0),
        data.get("traffic_state", "FREE"),
        data.get("cars", 0),
        data.get("buses", 0),
        data.get("trucks", 0),
        data.get("motorcycles", 0),
        data.get("avg_speed", 0.0),
    ))
    conn.commit()
    return cur.lastrowid


_ANALYSIS_JOB_FIELDS = {
    "output_path", "status", "error", "sample_rate", "max_frames",
    "confidence", "inference_size", "file_size_bytes", "duration_seconds",
    "source_fps", "source_width", "source_height", "processed_frames",
    "detection_count", "tracking_count", "class_totals_json", "metrics_json",
}


def create_analysis_job(conn: sqlite3.Connection, job_id: str, filename: str,
                        video_path: str, file_size_bytes: int,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
    """Persist an uploaded-video job and its validated source metadata."""
    metadata = metadata or {}
    now = datetime.now().isoformat()
    conn.execute("""
        INSERT INTO analysis_jobs
        (job_id, filename, video_path, status, file_size_bytes, duration_seconds,
         source_fps, source_width, source_height, created_at, updated_at)
        VALUES (?, ?, ?, 'uploaded', ?, ?, ?, ?, ?, ?, ?)
    """, (
        job_id, filename, video_path, int(file_size_bytes),
        metadata.get("duration_seconds"), metadata.get("fps"),
        metadata.get("width"), metadata.get("height"), now, now,
    ))
    conn.commit()


def update_analysis_job(conn: sqlite3.Connection, job_id: str, **fields) -> None:
    """Update only known analysis-job fields; JSON values are serialized."""
    unknown = set(fields) - _ANALYSIS_JOB_FIELDS
    if unknown:
        raise ValueError(f"Unsupported analysis-job fields: {sorted(unknown)}")
    if not fields:
        return
    for name in ("class_totals_json", "metrics_json"):
        if name in fields and not isinstance(fields[name], str):
            fields[name] = json.dumps(fields[name], sort_keys=True)
    fields["updated_at"] = datetime.now().isoformat()
    assignments = ", ".join(f"{name} = ?" for name in fields)
    cursor = conn.execute(
        f"UPDATE analysis_jobs SET {assignments} WHERE job_id = ?",
        (*fields.values(), job_id),
    )
    if cursor.rowcount == 0:
        raise KeyError(f"Analysis job not found: {job_id}")
    conn.commit()


def _decode_analysis_job(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    row["class_totals"] = json.loads(row.pop("class_totals_json") or "{}")
    row["metrics"] = json.loads(row.pop("metrics_json") or "{}")
    return row


def get_analysis_job(conn: sqlite3.Connection, job_id: str) -> Optional[Dict[str, Any]]:
    cursor = conn.execute("SELECT * FROM analysis_jobs WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    return _decode_analysis_job(dict(zip([col[0] for col in cursor.description], row)) if row else None)


def get_analysis_jobs(conn: sqlite3.Connection, limit: int = 100) -> List[Dict[str, Any]]:
    cursor = conn.execute(
        "SELECT * FROM analysis_jobs ORDER BY created_at DESC LIMIT ?",
        (max(1, min(int(limit), 1000)),),
    )
    columns = [col[0] for col in cursor.description]
    return [_decode_analysis_job(dict(zip(columns, row))) for row in cursor.fetchall()]


def get_traffic_history(conn: sqlite3.Connection,
                         lane_id: Optional[str] = None,
                         limit: int = 1000,
                         source: Optional[str] = None) -> pd.DataFrame:
    """Load traffic observation history from database."""
    conditions, params = [], []
    if lane_id:
        conditions.append("lane_id = ?")
        params.append(lane_id)
    if source:
        conditions.append("source = ?")
        params.append(source)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)
    query = f"SELECT * FROM traffic_observations {where} ORDER BY timestamp DESC LIMIT ?"
    df = pd.read_sql_query(query, conn, params=params)
    return df.iloc[::-1].reset_index(drop=True)   # chronological order


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame,
                       target_col: str = "vehicle_count",
                       lags: int = 12,
                       rolling_windows: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Create lag, rolling, and calendar features for ML prediction.

    Human Cognition Connection (Learning):
        Analogous to a human traffic officer who recalls how busy the road was
        10 minutes ago and notices patterns (rush hour, day-of-week effects).
        Lag features represent 'recent memory'; rolling stats represent
        'pattern recognition over experience'.
    """
    if rolling_windows is None:
        rolling_windows = [3, 6, 12]
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    for lag in range(1, lags + 1):
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)

    for w in rolling_windows:
        df[f"{target_col}_rmean{w}"] = df[target_col].rolling(w, min_periods=1).mean()
        df[f"{target_col}_rstd{w}"]  = df[target_col].rolling(w, min_periods=1).std().fillna(0)
        df[f"{target_col}_rmax{w}"]  = df[target_col].rolling(w, min_periods=1).max()

    if "timestamp" in df.columns:
        df["hour"]        = df["timestamp"].dt.hour
        df["minute"]      = df["timestamp"].dt.minute
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_peak"]     = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)

    df[f"{target_col}_diff1"] = df[target_col].diff().fillna(0)
    df[f"{target_col}_diff2"] = df[target_col].diff(2).fillna(0)

    return df.dropna().reset_index(drop=True)


# ── Video Utilities ───────────────────────────────────────────────────────────
def get_video_info(video_path: str) -> Dict[str, Any]:
    """Extract basic metadata from a video file."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":          fps,
        "total_frames": total,
        "duration_s":   total / fps,
    }
    cap.release()
    return info


def draw_roi(frame: np.ndarray,
             roi_points: List[List[int]],
             color: Tuple[int, int, int] = (0, 255, 0),
             label: str = "",
             thickness: int = 2) -> np.ndarray:
    """Overlay an ROI polygon and label on a video frame."""
    import cv2
    pts = np.array(roi_points, np.int32).reshape((-1, 1, 2))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (*color[:3], 40))
    cv2.polylines(frame, [pts], True, color, thickness)
    alpha = 0.15
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    if label and roi_points:
        cx = int(np.mean([p[0] for p in roi_points]))
        cy = int(np.mean([p[1] for p in roi_points]))
        cv2.putText(frame, label, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return frame


def is_point_in_roi(point: Tuple[float, float],
                    roi_points: List[List[int]]) -> bool:
    """Return True if the point lies inside the ROI polygon."""
    import cv2
    pts = np.array(roi_points, np.int32)
    return cv2.pointPolygonTest(pts,
                                 (float(point[0]), float(point[1])),
                                 False) >= 0


def put_text_bg(frame: np.ndarray, text: str,
                org: Tuple[int, int],
                font_scale: float = 0.55,
                color: Tuple[int, int, int] = (255, 255, 255),
                bg_color: Tuple[int, int, int] = (0, 0, 0),
                thickness: int = 1) -> np.ndarray:
    """Draw text with a filled background rectangle for readability."""
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 2, y - th - bl - 2),
                  (x + tw + 2, y + bl + 2), bg_color, -1)
    cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


# ── Simple Timer ──────────────────────────────────────────────────────────────
class Timer:
    """Wall-clock timer context manager."""
    def __init__(self):
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

    def start(self):
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        return self.elapsed_ms


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(level: str = "INFO"):
    """Configure loguru for the project."""
    import sys
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
    )
    log_file = PROJECT_ROOT / "experiments" / "system.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), level="DEBUG", rotation="10 MB", retention="7 days")


setup_logging(CONFIG.get("logging", {}).get("level", "INFO"))
