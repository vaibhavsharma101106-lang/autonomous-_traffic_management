"""
Module 2: Vehicle Tracking — ByteTrack Integration
IT 207 Human Intelligence Project

Implements persistent vehicle tracking using Ultralytics ByteTrack,
providing continuous vehicle IDs and trajectory management.

Human Intelligence Mapping — Working Memory:
    Tracking maintains a 'mental model' of each vehicle across frames,
    analogous to how humans keep track of moving objects in a scene using
    working memory (Baddeley, 2000). The track history (trajectory buffer)
    mirrors the phonological loop — retaining a short sequence of positions.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

from src.utils import CONFIG, PROJECT_ROOT, is_point_in_roi, put_text_bg


# ── Colour palette for track IDs ──────────────────────────────────────────────
def _id_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a deterministic BGR colour for a track ID."""
    np.random.seed(int(track_id) % 2048)
    return tuple(np.random.randint(60, 220, 3).tolist())


class VehicleTracker:
    """
    ByteTrack-based multi-object vehicle tracker.

    Key behaviours:
        - Assigns persistent IDs to vehicles across frames
        - Maintains trajectory history (up to `max_trail` points)
        - Counts unique vehicles per configured ROI lane
        - Prevents duplicate counting via the seen_ids set
        - Provides entry / exit event detection
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 roi_config: Optional[Dict[str, List[List[int]]]] = None,
                 max_trail: int = 30,
                 confidence: Optional[float] = None):

        cfg     = CONFIG["detection"]
        tr_cfg  = CONFIG["tracking"]

        self.confidence = confidence or cfg["confidence"]
        self.max_trail  = max_trail

        # Device
        import torch
        device_raw = cfg.get("device", "auto")
        self.device = "cuda" if (device_raw == "auto" and torch.cuda.is_available()) else (
            device_raw if device_raw != "auto" else "cpu"
        )

        # Load YOLO model (reused for tracking)
        if model_path is None:
            model_path = PROJECT_ROOT / "models" / "detection" / cfg["model"]
        try:
            self.model = YOLO(str(model_path))
        except Exception:
            logger.warning("Tracker: local weights missing — downloading yolo11n.pt")
            self.model = YOLO("yolo11n.pt")
        self.model.to(self.device)

        # ROI configuration (selective attention filter)
        self.roi_config  = roi_config or CONFIG["analysis"].get("roi_default", {})
        self._roi_arrays = {
            lane: np.array(pts, np.int32)
            for lane, pts in self.roi_config.items()
        }

        # Working memory structures
        self.track_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=max_trail)
        )
        self.seen_ids: Dict[str, set] = {lane: set() for lane in self.roi_config}
        self.active_tracks: Dict[int, Dict] = {}   # track_id → latest info

        # Class mapping
        from src.detection.detector import VEHICLE_CLASSES
        self.vehicle_classes = VEHICLE_CLASSES

        logger.success(f"VehicleTracker ready — device={self.device}, "
                       f"trails={max_trail}, lanes={list(self.roi_config)}")

    # ── Core tracking step ─────────────────────────────────────────────────
    def track_frame(self, frame: np.ndarray) -> Dict:
        """
        Run ByteTrack on one frame.

        Returns:
            tracks        — list of active track dicts
            lane_counts   — current vehicles per lane (in-frame)
            unique_counts — cumulative unique vehicles per lane (seen_ids)
            class_totals  — current vehicles by class
            frame_annot   — annotated frame
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            iou=CONFIG["detection"]["iou"],
            imgsz=CONFIG["detection"].get("imgsz", 640),
            classes=list(self.vehicle_classes.keys()),
            tracker="bytetrack.yaml",
            verbose=False,
            device=self.device,
        )

        frame_annot = frame.copy()
        tracks: List[Dict] = []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes    = results[0].boxes.xyxy.cpu().numpy()
            confs    = results[0].boxes.conf.cpu().numpy()
            cls_ids  = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, conf, cls_id, tid in zip(boxes, confs, cls_ids, track_ids):
                if cls_id not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cls_name = self.vehicle_classes[cls_id]
                color    = _id_color(tid)

                # Update trajectory (working memory)
                self.track_history[tid].append((cx, cy))

                # Lane assignment
                lanes = [ln for ln, arr in self._roi_arrays.items()
                         if cv2.pointPolygonTest(arr, (float(cx), float(cy)), False) >= 0]
                for ln in lanes:
                    self.seen_ids[ln].add(tid)

                track_info = {
                    "track_id":   int(tid),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": cx,  "cy": cy,
                    "confidence": float(conf),
                    "class_id":   int(cls_id),
                    "class_name": cls_name,
                    "lanes":      lanes,
                    "trail":      list(self.track_history[tid]),
                }
                tracks.append(track_info)
                self.active_tracks[int(tid)] = track_info

                # Draw bounding box & ID
                cv2.rectangle(frame_annot, (x1, y1), (x2, y2), color, 2)
                label = f"#{tid} {cls_name} {conf:.2f}"
                put_text_bg(frame_annot, label, (x1, y1 - 7),
                            color=color, bg_color=(20, 20, 20))

                # Draw trajectory trail
                trail = list(self.track_history[tid])
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    c = tuple(int(ch * alpha) for ch in color)
                    cv2.line(frame_annot, trail[i - 1], trail[i], c, 2)

        # Draw ROI lanes
        for lane_name, pts in self.roi_config.items():
            cv2.polylines(frame_annot,
                          [np.array(pts, np.int32)], True, (0, 255, 100), 2)
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            cv2.putText(frame_annot, lane_name.upper(),
                        (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 100), 2, cv2.LINE_AA)

        # Aggregate
        lane_counts   = self._count_in_lanes(tracks)
        class_totals  = self._count_by_class(tracks)
        unique_counts = {ln: len(ids) for ln, ids in self.seen_ids.items()}

        # HUD
        frame_annot = self._draw_tracker_hud(frame_annot, lane_counts,
                                              unique_counts, len(tracks))

        return {
            "tracks":        tracks,
            "lane_counts":   lane_counts,
            "unique_counts": unique_counts,
            "class_totals":  class_totals,
            "frame_annot":   frame_annot,
            "active_count":  len(tracks),
        }

    # ── Aggregation helpers ────────────────────────────────────────────────
    def _count_in_lanes(self, tracks: List[Dict]) -> Dict[str, int]:
        counts = {lane: 0 for lane in self.roi_config}
        for t in tracks:
            for ln in t["lanes"]:
                counts[ln] = counts.get(ln, 0) + 1
        return counts

    def _count_by_class(self, tracks: List[Dict]) -> Dict[str, int]:
        totals: Dict[str, int] = {}
        for t in tracks:
            cls = t["class_name"]
            totals[cls] = totals.get(cls, 0) + 1
        return totals

    # ── HUD ───────────────────────────────────────────────────────────────
    def _draw_tracker_hud(self, frame: np.ndarray,
                           lane_counts: Dict, unique_counts: Dict,
                           active: int) -> np.ndarray:
        cv2.rectangle(frame, (0, 0), (300, 20 + 20 * (len(lane_counts) * 2 + 2)),
                      (10, 10, 10), -1)
        y = 18
        cv2.putText(frame, f"Active tracks: {active}",
                    (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20
        for lane in lane_counts:
            cv2.putText(frame, f"  {lane} in-frame: {lane_counts[lane]}",
                        (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 100), 1)
            y += 18
            cv2.putText(frame, f"  {lane} unique:   {unique_counts.get(lane, 0)}",
                        (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)
            y += 16
        return frame

    # ── State management ──────────────────────────────────────────────────
    def reset_counts(self):
        """Reset cumulative unique-vehicle counters (e.g. new session)."""
        for lane in self.seen_ids:
            self.seen_ids[lane].clear()
        self.active_tracks.clear()
        logger.info("Tracker counters reset.")

    def get_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """Return the trajectory history for a given track ID."""
        return list(self.track_history.get(track_id, []))

    def update_roi(self, roi_config: Dict[str, List[List[int]]]):
        """Update ROI configuration at runtime."""
        self.roi_config  = roi_config
        self._roi_arrays = {ln: np.array(pts, np.int32)
                            for ln, pts in roi_config.items()}
        self.seen_ids    = {ln: set() for ln in roi_config}
        logger.info(f"Tracker ROI updated: {list(roi_config)}")
