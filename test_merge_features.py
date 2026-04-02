from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

import backend
from tracking_service import pipeline, storage
from tracking_service.schemas import (
    CalibrationEnvelope,
    CalibrationPreset,
    DetectionRecord,
    FieldLandmark,
    MatchArtifactSet,
    MatchRecord,
    TrackRecord,
    ViewCalibration,
)


def build_view_payload() -> dict:
    return {
        "view": "main",
        "roi": [0.0, 0.0, 100.0, 100.0],
        "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "landmarks": [
            {"name": "top_left", "image_point": [10.0, 10.0], "field_point": [-50.0, 50.0], "confidence": 0.9},
            {"name": "top_right", "image_point": [90.0, 10.0], "field_point": [50.0, 50.0], "confidence": 0.9},
            {"name": "bottom_left", "image_point": [10.0, 90.0], "field_point": [-50.0, -50.0], "confidence": 0.9},
            {"name": "bottom_right", "image_point": [90.0, 90.0], "field_point": [50.0, -50.0], "confidence": 0.9},
        ],
        "distortion_strength": 0.0,
        "distortion_x": 0.0,
        "distortion_y": 0.0,
        "reprojection_error": 0.0,
        "confidence": 1.0,
    }


def build_match_record() -> MatchRecord:
    view = ViewCalibration.model_validate(build_view_payload())
    now = time.time()
    return MatchRecord(
        id="match-123",
        created_at=now,
        updated_at=now,
        metadata={"display_name": "Test Match", "status": "awaiting_calibration", "processing": False},
        source={"source_name": "test.mp4", "source_url": "https://example.com/video", "stored_path": None},
        calibration=CalibrationEnvelope(
            mode="manual_override",
            created_at=now,
            updated_at=now,
            quality_checks={},
            views=[view],
        ),
        artifacts=MatchArtifactSet(),
        debug={},
    )


def build_preset(preset_id: str = "preset-1") -> CalibrationPreset:
    match = build_match_record()
    return CalibrationPreset(
        id=preset_id,
        name=f"Preset {preset_id}",
        created_at=match.created_at,
        updated_at=match.updated_at,
        calibration=match.calibration,
    )


def test_create_source_with_calibrate_first_uses_preset_and_saves_draft(monkeypatch):
    client = TestClient(backend.app)
    saved_matches: list[MatchRecord] = []
    preset = build_preset()

    class FakeStore:
        def load_calibration_preset(self, preset_id: str) -> CalibrationPreset:
            assert preset_id == preset.id
            return preset

        def save_match(self, match: MatchRecord) -> None:
            saved_matches.append(match)

        def copy_into_artifacts(self, source_path: str | None, match_id: str, name: str) -> str | None:
            raise AssertionError("copy_into_artifacts should not be called for URL sources")

    monkeypatch.setattr(backend, "store", FakeStore())
    monkeypatch.setattr(
        backend,
        "resolve_source",
        lambda source: pipeline.ResolvedSource(
            video_path="https://cdn.example.com/match.mp4",
            display_name="Resolved Match",
            source_url=source.source_url,
        ),
    )

    response = client.post(
        "/sources",
        json={
            "youtube_url": "https://youtu.be/example",
            "match_name": "Week 1 Final",
            "calibration_preset_id": preset.id,
            "calibrate_first": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()["match"]
    assert payload["metadata"]["status"] == "awaiting_calibration"
    assert payload["metadata"]["calibration_preset_id"] == preset.id
    assert payload["artifacts"]["source_video"] == "https://cdn.example.com/match.mp4"
    assert len(saved_matches) == 1
    assert saved_matches[0].calibration == preset.calibration


def test_update_match_calibration_rebuilds_existing_tracking(monkeypatch):
    client = TestClient(backend.app)
    match = build_match_record()
    match.detections.append(
        DetectionRecord(
            frame=1,
            time=0.1,
            view="main",
            source_confidence=0.9,
            image_anchor=[20.0, 20.0],
            field_point=[0.0, 0.0],
            bbox=[10.0, 10.0, 30.0, 30.0],
        )
    )
    match.tracks.append(
        TrackRecord(
            frame=1,
            time=0.1,
            track_id=7,
            x=0.0,
            y=0.0,
            confidence=0.9,
            source_views=["main"],
            source_detection_indices=[0],
        )
    )

    rebuild_calls: list[str] = []

    class FakeStore:
        def load_match(self, match_id: str) -> MatchRecord:
            assert match_id == match.id
            return match

        def save_match(self, updated_match: MatchRecord) -> None:
            raise AssertionError("save_match should not be called when rebuild_match_tracking handles persistence")

    def fake_rebuild(updated_match: MatchRecord, store_obj) -> MatchRecord:
        rebuild_calls.append(updated_match.id)
        return updated_match

    monkeypatch.setattr(backend, "store", FakeStore())
    monkeypatch.setattr(backend, "rebuild_match_tracking", fake_rebuild)

    response = client.put(
        f"/matches/{match.id}/calibration",
        json={"mode": "manual_override", "quality_checks": {"ok": True}, "views": [build_view_payload()]},
    )

    assert response.status_code == 200
    assert rebuild_calls == [match.id]


def test_update_match_calibration_saves_draft_without_rebuild(monkeypatch):
    client = TestClient(backend.app)
    match = build_match_record()
    saved_matches: list[MatchRecord] = []

    class FakeStore:
        def load_match(self, match_id: str) -> MatchRecord:
            assert match_id == match.id
            return match

        def save_match(self, updated_match: MatchRecord) -> None:
            saved_matches.append(updated_match)

    monkeypatch.setattr(backend, "store", FakeStore())
    monkeypatch.setattr(
        backend,
        "rebuild_match_tracking",
        lambda updated_match, store_obj: (_ for _ in ()).throw(AssertionError("rebuild should not run for drafts")),
    )

    response = client.put(
        f"/matches/{match.id}/calibration",
        json={"mode": "manual_override", "quality_checks": {"draft": True}, "views": [build_view_payload()]},
    )

    assert response.status_code == 200
    assert len(saved_matches) == 1
    assert saved_matches[0].calibration.quality_checks == {"draft": True}


def test_fuse_field_detections_merges_transitive_neighbors_with_weighting():
    detections = [
        {"field_point": [0.0, 0.0], "confidence": 0.2, "view": "left"},
        {"field_point": [20.0, 0.0], "confidence": 0.3, "view": "main"},
        {"field_point": [40.0, 0.0], "confidence": 0.5, "view": "right"},
    ]

    fused = pipeline.fuse_field_detections(detections)

    assert len(fused) == 1
    assert fused[0]["source_views"] == ["left", "main", "right"]
    assert fused[0]["source_detection_indices"] == [0, 1, 2]
    assert fused[0]["view"] == "right"
    assert fused[0]["field_point"] == [26.0, 0.0]


def test_tracking_store_calibration_preset_round_trip_and_sorting(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "CALIBRATION_PRESET_ROOT", tmp_path / "calibrations")
    storage.CALIBRATION_PRESET_ROOT.mkdir(parents=True, exist_ok=True)
    store = storage.TrackingStore()

    first = build_preset("preset-a")
    second = build_preset("preset-b")

    store.save_calibration_preset(first)
    time.sleep(0.01)
    store.save_calibration_preset(second)

    loaded = store.load_calibration_preset(first.id)
    presets = store.list_calibration_presets()

    assert loaded.id == first.id
    assert loaded.calibration.views[0].view == "main"
    assert [preset.id for preset in presets[:2]] == [second.id, first.id]
    assert Path(storage.CALIBRATION_PRESET_ROOT, f"{first.id}.json").exists()
