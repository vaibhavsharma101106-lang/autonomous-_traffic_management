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
from typing import Dict, List, Optional, Tuple

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
                 roi_config: Optional[Dict[str, List[List[int]]]] = None):

        cfg = CONFIG["detection"]
        self.confidence = confidence or cfg["confidence"]
        self.iou        = iou        or cfg["iou"]
        self.imgsz      = cfg.get("imgsz", 640)
        self.half       = cfg.get("half", False)

        # Device selection
        raw_device = device or cfg.get("device", "auto")
        if raw_device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = raw_device

        # Load YOLO model
        if model_path is None:
            model_path = PROJECT_ROOT / "models" / "detection" / cfg["model"]
        model_path = str(model_path)

        logger.info(f"Loading YOLO model: {model_path} on {self.device}")
        try:
            self.model = YOLO(model_path)
        except Exception:
            # Fallback: download from ultralytics hub
            logger.warning("Local weights not found — downloading yolo11n.pt")
            self.model = YOLO("yolo11n.pt")
            # Save to models directory for reuse
            dest = PROJECT_ROOT / "models" / "detection" / "yolo11n.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            src = Path(self.model.ckpt_path)
            if src.exists() and not dest.exists():
                shutil.copy(str(src), str(dest))

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
def process_video_frames(video_path: str,
                          detector: VehicleDetector,
                          sample_rate: int = 5,
                          max_frames: Optional[int] = None,
                          output_path: Optional[str] = None) -> List[Dict]:
    """
    Process a video file frame-by-frame, returning detection results.

    Args:
        video_path:   Path to input video.
        detector:     Initialised VehicleDetector.
        sample_rate:  Analyse every Nth frame (reduces compute).
        max_frames:   Stop after this many analysed frames (None = all).
        output_path:  If provided, write annotated video here.

    Returns:
        List of per-frame detection result dicts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps / sample_rate,
                                 (width, height))

    results_list: List[Dict] = []
    frame_idx, analysed = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % sample_rate != 0:
            continue

        result = detector.detect(frame)
        result["frame_index"] = frame_idx
        result["timestamp_s"] = frame_idx / fps
        results_list.append(result)

        if writer:
            writer.write(result["frame_annot"])

        analysed += 1
        if max_frames and analysed >= max_frames:
            break

    cap.release()
    if writer:
        writer.release()

    logger.info(f"Processed {analysed} frames from {Path(video_path).name} "
                f"(avg latency {detector.avg_latency_ms:.1f} ms)")
    return results_list
