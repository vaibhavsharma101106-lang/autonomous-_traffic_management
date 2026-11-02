"""
Module 1: Vehicle Detection
IT 207 Human Intelligence Project

Implements YOLO11n-based vehicle detection with selective attention:
- Region-of-interest filtering (selective attention)
- Vehicle classification: car, motorcycle, bus, truck
- Confidence-scored bounding box visualisation
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

from src.utils import (
    CONFIG, PROJECT_ROOT, Timer,
    draw_roi, is_point_in_roi, put_text_bg,
    state_to_color_bgr,
)

# ── COCO class mapping for vehicles ──────────────────────────────────────────
VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "car":        (0, 200, 255),
    "motorcycle": (255, 100, 0),
    "bus":        (0, 60, 200),
    "truck":      (0, 140, 100),
    "unknown":    (128, 128, 128),
}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def resolve_model_weights(model_path: Optional[str] = None) -> Path:
    """Resolve YOLO weights from the project root, independent of cwd."""
    configured_path = Path(model_path or CONFIG["detection"]["model"]).expanduser()
    if configured_path.is_absolute():
        return configured_path.resolve()

    # Bare model names use the shared, predictable weights directory. Explicit
    # relative paths in config remain relative to the project root.
    if model_path is None and len(configured_path.parts) == 1:
        configured_path = Path("models") / "weights" / configured_path
    elif model_path is not None and len(configured_path.parts) == 1:
        configured_path = Path("models") / "weights" / configured_path
    return (PROJECT_ROOT / configured_path).resolve()


def load_yolo_model(model_path: Optional[str] = None):
    """Load the official YOLO model, letting Ultralytics fetch missing weights."""
    weights_path = resolve_model_weights(model_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading Ultralytics model weights: {weights_path}")

    try:
        # Ultralytics downloads known model assets on first use. Passing the
        # project-root path makes that download persistent and predictable.
        model = YOLO(str(weights_path))
        if not weights_path.is_file():
            checkpoint = Path(getattr(model, "ckpt_path", ""))
            if checkpoint.is_file() and checkpoint.resolve() != weights_path:
                import shutil
                shutil.copy2(checkpoint, weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Ultralytics loaded the model without saving weights to {weights_path}")
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Could not load YOLO weights at {weights_path}. Ultralytics could not "
            "load the local file or download the official model. Check the internet "
            "connection and that the installed Ultralytics version supports YOLO11. "
            f"Original error: {exc}"
        ) from exc


class VehicleDetector:
    """
    YOLO11n vehicle detector with selective-attention ROI filtering.

    Human Intelligence Mapping — Selective Attention:
        The human visual system does not process an entire scene uniformly;
        it selectively focuses on regions of interest based on task relevance.
        This detector implements the same principle: configurable ROI polygons
        define the 'attentional focus', and detections outside those regions
        are suppressed, reducing noise and computational waste.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 confidence: Optional[float] = None,
                 iou: Optional[float] = None,
                 device: Optional[str] = None,
                 roi_config: Optional[Dict[str, List[List[int]]]] = None,
                 imgsz: Optional[int] = None):

        cfg = CONFIG["detection"]
        self.confidence = confidence or cfg["confidence"]
        self.iou        = iou        or cfg["iou"]
        self.imgsz      = int(imgsz or cfg.get("imgsz", 640))
        self.half       = cfg.get("half", False)

        # Device selection
        raw_device = device or cfg.get("device", "auto")
        if raw_device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = raw_device

        logger.info(f"Loading YOLO model on {self.device}")
        self.model = load_yolo_model(model_path)
        self.model.to(self.device)
        if self.half and self.device == "cuda":
            self.model.model.half()

        # ROI configuration (selective attention)
        self.roi_config = roi_config or CONFIG["analysis"].get("roi_default", {})
        self._roi_arrays = {
            lane: np.array(pts, np.int32)
            for lane, pts in self.roi_config.items()
        }

        # Inference stats
        self._inference_times: List[float] = []
        logger.success(f"VehicleDetector ready — device={self.device}, "
                       f"conf={self.confidence}, ROIs={list(self.roi_config)}")

    # ── Core detection ─────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Run YOLO detection on a single frame.

        Returns a dict with:
            detections  — list of dicts per vehicle
            lane_counts — vehicle counts per configured ROI lane
            class_totals— totals per class across all lanes
            frame_annot — annotated frame (BGR)
            latency_ms  — inference wall-clock time
        """
        with Timer() as t:
            results = self.model.predict(
                frame,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.imgsz,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
                device=self.device,
            )
        latency = t.elapsed_ms
        self._inference_times.append(latency)

        detections: List[Dict] = []
        frame_annot = frame.copy()

        if results and results[0].boxes is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()    # x1,y1,x2,y2
            confs   = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = VEHICLE_CLASSES[cls_id]
                color = CLASS_COLORS_BGR[class_name]

                # Determine which ROI lanes contain this vehicle centroid
                assigned_lanes = self._assign_lanes(cx, cy)

                det = {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": cx, "cy": cy,
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "track_id": -1,   # filled by tracker
                    "lanes": assigned_lanes,
                }
                detections.append(det)

                # Draw bounding box
                cv2.rectangle(frame_annot, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f}"
                put_text_bg(frame_annot, label, (x1, y1 - 6),
                            color=color, bg_color=(0, 0, 0))

        # Draw ROI overlays
        for lane_name, pts in self.roi_config.items():
            frame_annot = draw_roi(
                frame_annot, pts,
                color=(0, 255, 100),
                label=lane_name.upper()
            )

        # Aggregate counts
        lane_counts = self._count_by_lane(detections)
        class_totals = self._count_by_class(detections)

        # HUD overlay
        frame_annot = self._draw_hud(frame_annot, lane_counts, class_totals, latency)

        return {
            "detections":  detections,
            "lane_counts": lane_counts,
            "class_totals": class_totals,
            "frame_annot": frame_annot,
            "latency_ms":  latency,
        }

    # ── ROI helpers ────────────────────────────────────────────────────────
    def _assign_lanes(self, cx: int, cy: int) -> List[str]:
        lanes = []
        for lane, pts in self._roi_arrays.items():
            if cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0:
                lanes.append(lane)
        return lanes

    def _count_by_lane(self, detections: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {lane: 0 for lane in self.roi_config}
        for det in detections:
            for lane in det["lanes"]:
                counts[lane] = counts.get(lane, 0) + 1
        return counts

    def _count_by_class(self, detections: List[Dict]) -> Dict[str, int]:
        totals: Dict[str, int] = {name: 0 for name in VEHICLE_CLASSES.values()}
        for det in detections:
            cls = det["class_name"]
            totals[cls] = totals.get(cls, 0) + 1
        return totals

    # ── HUD ───────────────────────────────────────────────────────────────
    def _draw_hud(self, frame: np.ndarray,
                  lane_counts: Dict[str, int],
                  class_totals: Dict[str, int],
                  latency_ms: float) -> np.ndarray:
        h, w = frame.shape[:2]
        # Background panel
        cv2.rectangle(frame, (0, 0), (280, 22 + 20 * (len(lane_counts) + len(class_totals) + 2)),
                      (0, 0, 0), -1)
        y = 18
        cv2.putText(frame, f"Inference: {latency_ms:.1f}ms",
                    (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y += 20
        for lane, cnt in lane_counts.items():
            cv2.putText(frame, f"  {lane}: {cnt} vehicles",
                        (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1, cv2.LINE_AA)
            y += 18
        for cls, cnt in class_totals.items():
            if cnt > 0:
                color = CLASS_COLORS_BGR.get(cls, (200, 200, 200))
                cv2.putText(frame, f"  {cls}: {cnt}",
                            (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                y += 16
        return frame

    # ── Stats ─────────────────────────────────────────────────────────────
    @property
    def avg_latency_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return float(np.mean(self._inference_times[-100:]))

    def reset_stats(self):
        self._inference_times.clear()

    def update_roi(self, roi_config: Dict[str, List[List[int]]]):
        """Dynamically update ROI configuration."""
        self.roi_config  = roi_config
        self._roi_arrays = {
            lane: np.array(pts, np.int32)
            for lane, pts in roi_config.items()
        }
        logger.info(f"ROI updated: {list(roi_config)}")


# ── Batch video processing ────────────────────────────────────────────────────
def validate_video_file(video_path: str,
                        max_size_bytes: int = 500 * 1024 * 1024) -> Dict[str, float]:
    """Validate an uploaded video and return decoder metadata before analysis."""
    path = Path(video_path)
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '{path.suffix or '(no extension)'}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("The uploaded video file is empty.")
    if size > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        raise ValueError(f"Video is too large ({size / (1024 * 1024):.1f} MB). Maximum is {max_mb:.0f} MB.")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("OpenCV could not decode this video. Try an MP4 encoded with H.264.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        readable, first_frame = cap.read()
        if not readable or first_frame is None:
            raise ValueError("The video opened, but no frames could be decoded.")
        if width <= 0 or height <= 0 or frame_count <= 0:
            raise ValueError("The video has invalid dimensions or no frames.")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("The video reports an invalid frame rate.")
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / fps,
            "file_size_bytes": size,
        }
    finally:
        cap.release()


def process_video_frames(video_path: str,
                          detector: VehicleDetector,
                          sample_rate: int = 5,
                          max_frames: Optional[int] = None,
                          output_path: Optional[str] = None,
                          tracker=None,
                          max_inference_dim: int = 1280,
                          progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict]:
    """
    Process a video file frame-by-frame, returning detection results.

    Args:
        video_path:   Path to input video.
        detector:     Initialised VehicleDetector.
        sample_rate:  Analyse every Nth frame (reduces compute).
        max_frames:   Stop after this many analysed frames (None = all).
        output_path:  If provided, write annotated video here.
        tracker:      Optional ByteTrack VehicleTracker used instead of detection-only inference.
        max_inference_dim: Resize the longest frame dimension before inference and output.
        progress_callback: Optional callback receiving (processed, expected) frame counts.

    Returns:
        List of per-frame detection result dicts.
    """
    if sample_rate < 1:
        raise ValueError("Frame sample rate must be at least 1.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("Maximum frame count must be at least 1 when provided.")
    if max_inference_dim < 320:
        raise ValueError("Inference frame size must be at least 320 pixels.")
    metadata = validate_video_file(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    total = metadata["frame_count"]
    scale = min(1.0, max_inference_dim / max(width, height))
    output_width = max(1, int(round(width * scale)))
    output_height = max(1, int(round(height * scale)))
    expected = max(1, int(np.ceil(total / sample_rate)))
    if max_frames is not None:
        expected = min(expected, max_frames)

    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, max(fps / sample_rate, 0.1),
                                 (output_width, output_height))
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Could not create annotated video output: {output_path}")

    results_list: List[Dict] = []
    frame_idx, analysed = 0, 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if (frame_idx - 1) % sample_rate != 0:
                continue

            inference_frame = (
                cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
                if scale < 1.0 else frame
            )
            if tracker is not None:
                started = time.perf_counter()
                tracked = tracker.track_frame(inference_frame)
                latency_ms = (time.perf_counter() - started) * 1000
                detections = tracked["tracks"]
                result = {
                    **tracked,
                    "detections": detections,
                    "track_count": len(detections),
                    "latency_ms": latency_ms,
                }
            else:
                result = detector.detect(inference_frame)
                result["track_count"] = 0

            result["frame_index"] = frame_idx
            result["timestamp_s"] = frame_idx / fps
            result["source_width"] = width
            result["source_height"] = height
            results_list.append(result)

            if writer:
                writer.write(result["frame_annot"])

            analysed += 1
            if progress_callback:
                progress_callback(analysed, expected)
            if max_frames is not None and analysed >= max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()

    latencies = [result["latency_ms"] for result in results_list]
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    logger.info(f"Processed {analysed} frames from {Path(video_path).name} "
                f"(avg latency {avg_latency:.1f} ms)")
    return results_list
