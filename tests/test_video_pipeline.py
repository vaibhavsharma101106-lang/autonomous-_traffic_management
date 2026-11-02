"""Focused tests for recorded-video validation, tracking flow and job persistence."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.detection.detector import process_video_frames, validate_video_file
from src.utils import create_analysis_job, get_analysis_job, get_analysis_jobs, init_database, update_analysis_job


class FakeDetector:
    def detect(self, frame):
        return {"detections": [{"class_name": "car"}], "lane_counts": {"lane1": 1},
                "class_totals": {"car": 1}, "frame_annot": frame.copy(), "latency_ms": 1.0}


class FakeTracker:
    def __init__(self):
        self.frames = 0

    def track_frame(self, frame):
        self.frames += 1
        return {"tracks": [{"track_id": self.frames, "class_name": "car"}],
                "lane_counts": {"lane1": 1}, "class_totals": {"car": 1},
                "frame_annot": frame.copy()}


class VideoPipelineTests(unittest.TestCase):
    def make_video(self, directory, name="clip.avi", count=5):
        path = Path(directory) / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
        for n in range(count):
            writer.write(np.full((360, 640, 3), n * 20, dtype=np.uint8))
        writer.release()
        return path

    def test_video_is_validated_and_first_sample_is_processed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_video(directory)
            metadata = validate_video_file(str(source))
            self.assertEqual((metadata["width"], metadata["height"]), (640, 360))
            progress = []
            results = process_video_frames(str(source), FakeDetector(), sample_rate=3,
                max_inference_dim=320, progress_callback=lambda done, expected: progress.append((done, expected)))
            self.assertEqual([r["frame_index"] for r in results], [1, 4])
            self.assertEqual(progress[-1], (2, 2))
            self.assertEqual(results[0]["source_width"], 640)

    def test_tracker_is_used_and_annotated_output_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_video(directory)
            output = Path(directory) / "annotated.mp4"
            tracker = FakeTracker()
            results = process_video_frames(str(source), FakeDetector(), sample_rate=1,
                max_frames=2, output_path=str(output), tracker=tracker)
            self.assertEqual(tracker.frames, 2)
            self.assertEqual(results[0]["detections"][0]["track_id"], 1)
            cap = cv2.VideoCapture(str(output))
            try:
                readable, frame = cap.read()
                self.assertTrue(readable)
                self.assertEqual(frame.shape[1], 640)
            finally:
                cap.release()

    def test_analysis_job_round_trip_and_validation_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            db = init_database(str(Path(directory) / "jobs.sqlite"))
            source = self.make_video(directory)
            metadata = validate_video_file(str(source))
            create_analysis_job(db, "job-1", source.name, str(source), source.stat().st_size, metadata)
            update_analysis_job(db, "job-1", status="completed", class_totals_json={"car": 3}, metrics_json={"frames": 3})
            job = get_analysis_job(db, "job-1")
            self.assertEqual(job["class_totals"], {"car": 3})
            self.assertEqual(job["metrics"]["frames"], 3)
            self.assertEqual(len(get_analysis_jobs(db)), 1)
            db.close()
            with self.assertRaises(ValueError):
                validate_video_file(str(source), max_size_bytes=1)
            with self.assertRaises(ValueError):
                validate_video_file(str(Path(directory) / "bad.txt"))


if __name__ == "__main__":
    unittest.main()
