from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from .config import ARTIFACT_ROOT, CALIBRATION_PRESET_ROOT, JOB_ROOT, MATCH_ROOT, UPLOAD_ROOT, WATCHBOT_ROOT, ensure_data_dirs
from .schemas import CalibrationPreset, JobLogEntry, JobRecord, MatchRecord, SourceSubmission, WatchbotState


class TrackingStore:
    def __init__(self) -> None:
        ensure_data_dirs()

    def _job_path(self, job_id: str) -> Path:
        return JOB_ROOT / f"{job_id}.json"

    def _match_path(self, match_id: str) -> Path:
        return MATCH_ROOT / f"{match_id}.json"

    def _watchbot_path(self) -> Path:
        return WATCHBOT_ROOT / "state.json"

    def _calibration_preset_path(self, preset_id: str) -> Path:
        return CALIBRATION_PRESET_ROOT / f"{preset_id}.json"

    def save_upload(self, source_name: str, data: bytes) -> tuple[str, str]:
        ext = Path(source_name).suffix or ".mp4"
        upload_id = uuid.uuid4().hex
        target = UPLOAD_ROOT / f"{upload_id}{ext}"
        target.write_bytes(data)
        return upload_id, str(target)

    def create_job(self, source: SourceSubmission) -> JobRecord:
        now = time.time()
        job = JobRecord(
            id=uuid.uuid4().hex,
            created_at=now,
            updated_at=now,
            source=source,
        )
        self.save_job(job)
        return job

    def save_job(self, job: JobRecord) -> None:
        job.updated_at = time.time()
        self._job_path(job.id).write_text(job.model_dump_json(indent=2))

    def load_job(self, job_id: str) -> JobRecord:
        return JobRecord.model_validate_json(self._job_path(job_id).read_text())

    def list_jobs(self) -> list[JobRecord]:
        jobs = [JobRecord.model_validate_json(path.read_text()) for path in JOB_ROOT.glob("*.json")]
        jobs.sort(key=lambda item: item.created_at, reverse=True)
        return jobs

    def delete_job(self, job_id: str) -> None:
        path = self._job_path(job_id)
        if not path.exists():
            raise FileNotFoundError(job_id)
        job = JobRecord.model_validate_json(path.read_text())
        path.unlink()
        if job.source.stored_path:
            self._delete_if_unreferenced(Path(job.source.stored_path))

    def append_job_log(self, job_id: str, message: str, level: str = "info") -> JobRecord:
        job = self.load_job(job_id)
        job.logs.append(JobLogEntry(timestamp=time.time(), level=level, message=message))
        self.save_job(job)
        return job

    def save_match(self, match: MatchRecord) -> None:
        match.updated_at = time.time()
        self._match_path(match.id).write_text(match.model_dump_json(indent=2))

    def load_match(self, match_id: str) -> MatchRecord:
        return MatchRecord.model_validate_json(self._match_path(match_id).read_text())

    def list_matches(self) -> list[MatchRecord]:
        matches = [MatchRecord.model_validate_json(path.read_text()) for path in MATCH_ROOT.glob("*.json")]
        matches.sort(key=lambda item: item.updated_at, reverse=True)
        return matches

    def delete_match(self, match_id: str) -> None:
        match = self.load_match(match_id)
        match_path = self._match_path(match_id)
        if match_path.exists():
            match_path.unlink()

        artifact_dir = ARTIFACT_ROOT / match_id
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)

        stored_path = match.source.get("stored_path") if isinstance(match.source, dict) else None
        if stored_path:
            self._delete_if_unreferenced(Path(stored_path), excluded_match_id=match_id)

    def create_match_artifact_dir(self, match_id: str) -> Path:
        path = ARTIFACT_ROOT / match_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_calibration_preset(self, preset: CalibrationPreset) -> None:
        preset.updated_at = time.time()
        self._calibration_preset_path(preset.id).write_text(preset.model_dump_json(indent=2))

    def load_calibration_preset(self, preset_id: str) -> CalibrationPreset:
        return CalibrationPreset.model_validate_json(self._calibration_preset_path(preset_id).read_text())

    def list_calibration_presets(self) -> list[CalibrationPreset]:
        presets = [CalibrationPreset.model_validate_json(path.read_text()) for path in CALIBRATION_PRESET_ROOT.glob("*.json")]
        presets.sort(key=lambda item: item.updated_at, reverse=True)
        return presets

    def save_watchbot_state(self, state: WatchbotState) -> None:
        self._watchbot_path().write_text(state.model_dump_json(indent=2))

    def load_watchbot_state(self) -> WatchbotState:
        path = self._watchbot_path()
        if not path.exists():
            state = WatchbotState()
            self.save_watchbot_state(state)
            return state
        return WatchbotState.model_validate_json(path.read_text())

    def copy_into_artifacts(self, source_path: Optional[str], match_id: str, name: str) -> Optional[str]:
        if not source_path:
            return None
        source = Path(source_path)
        if not source.exists():
            return None
        artifact_dir = self.create_match_artifact_dir(match_id)
        target = artifact_dir / f"{name}{source.suffix or '.mp4'}"
        shutil.copy2(source, target)
        return f"/artifacts/{match_id}/{target.name}"

    def _delete_if_unreferenced(self, path: Path, excluded_match_id: Optional[str] = None) -> None:
        if not path.exists():
            return

        path_str = str(path)
        for match_path in MATCH_ROOT.glob("*.json"):
            if excluded_match_id and match_path.stem == excluded_match_id:
                continue
            match = MatchRecord.model_validate_json(match_path.read_text())
            source_path = match.source.get("stored_path") if isinstance(match.source, dict) else None
            if source_path == path_str:
                return

        for job_path in JOB_ROOT.glob("*.json"):
            job = JobRecord.model_validate_json(job_path.read_text())
            if job.source.stored_path == path_str:
                return

        path.unlink(missing_ok=True)
