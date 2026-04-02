from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import threading
import time
import urllib.parse
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests

from tracking_service.calibration import (
    field_points_form_valid_quad,
    image_landmarks_form_valid_quad,
    load_default_calibration,
    solve_view_homography,
)
from tracking_service.config import ARTIFACT_ROOT, DATA_ROOT, TARGET_FPS, ensure_data_dirs
from tracking_service.fuel import (
    invalidate_fuel_analysis,
    normalize_quad,
    normalize_rgb_color,
    run_fuel_analysis,
    sample_match_video_color,
)
from tracking_service.pipeline import process_job, rebuild_match_tracking, resolve_source
from tracking_service.schemas import (
    CalibrationEnvelope,
    CalibrationPreset,
    FieldLandmark,
    MatchArtifactSet,
    MatchLabelUpdate,
    MatchRecord,
    SourceSubmission,
    ViewCalibration,
)
from tracking_service.storage import TrackingStore
from tracking_service.watchbot import WatchbotManager

ensure_data_dirs()

app = FastAPI(title="FRC Tracking Platform", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/artifacts", StaticFiles(directory=ARTIFACT_ROOT), name="artifacts")
app.mount("/data", StaticFiles(directory=DATA_ROOT), name="data")

store = TrackingStore()
watchbot = WatchbotManager(store)
DEFAULT_TBA_API_KEY = "vAcmnornjIGHIHS52Zm9kKvrwjXJ3jzsbVaiwfTRQQgKP0lbsM36sRrtTjNRV4aB"


def _run_job(job_id: str) -> None:
    try:
        process_job(store.load_job(job_id), store)
    except Exception as exc:
        job = store.load_job(job_id)
        job.status = "failed"
        job.error = str(exc)
        store.save_job(job)
        job = store.append_job_log(job.id, f"Job failed: {exc}", level="error")


def _mark_orphaned_running_jobs() -> None:
    for job in store.list_jobs():
        if job.status != "running":
            continue

        job.status = "failed"
        job.error = (
            "Processing was interrupted because the backend server restarted while this job was running. "
            "Please queue the source again."
        )
        store.save_job(job)
        store.append_job_log(
            job.id,
            "Marked as failed on startup because the previous backend process exited mid-run.",
            level="error",
        )


def _resolve_preset_calibration(calibration_preset_id: str | None):
    if not calibration_preset_id:
        return None, load_default_calibration()

    try:
        preset = store.load_calibration_preset(calibration_preset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Calibration preset not found.") from exc
    return preset, preset.calibration


def _create_match_draft(source: SourceSubmission) -> dict:
    resolved = resolve_source(source)
    preset, calibration = _resolve_preset_calibration(source.calibration_preset_id)

    match = MatchRecord(
        id=uuid.uuid4().hex,
        created_at=time.time(),
        updated_at=time.time(),
        metadata={
            "display_name": source.requested_match_name or Path(source.source_name).stem,
            "status": "awaiting_calibration",
            "fps": TARGET_FPS,
            "source_kind": source.source_kind,
            "start_mode": "ocr_gated" if source.source_kind == "watchbot" else "immediate",
            "calibration_preset_id": preset.id if preset is not None else None,
            "calibration_preset_name": preset.name if preset is not None else None,
            "processing": False,
        },
        source={
            "source_name": source.source_name,
            "source_url": resolved.source_url,
            "stored_path": source.stored_path,
        },
        calibration=calibration,
        artifacts=MatchArtifactSet(),
        debug={"stages": ["draft_created"]},
    )

    if source.stored_path:
        copied_source = store.copy_into_artifacts(source.stored_path, match.id, "source")
        if copied_source:
            match.artifacts.source_video = copied_source
    elif resolved.video_path.startswith("http"):
        match.artifacts.source_video = resolved.video_path

    store.save_match(match)
    return match.model_dump()


@app.on_event("startup")
async def reconcile_jobs_on_startup() -> None:
    _mark_orphaned_running_jobs()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/jobs")
async def list_jobs() -> list[dict]:
    return [job.model_dump() for job in store.list_jobs()]


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    try:
        return store.load_job(job_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str) -> dict:
    try:
        job = store.load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc

    linked_match_id = job.match_id
    if linked_match_id:
        try:
            store.delete_match(linked_match_id)
        except FileNotFoundError:
            pass
        for sibling_job in store.list_jobs():
            if sibling_job.match_id == linked_match_id:
                try:
                    store.delete_job(sibling_job.id)
                except FileNotFoundError:
                    pass
    else:
        store.delete_job(job_id)
    return {"deleted": True, "job_id": job_id}


@app.post("/sources")
async def create_source(payload: dict) -> dict:
    youtube_url = payload.get("youtube_url")
    match_name = payload.get("match_name")
    calibration_preset_id = payload.get("calibration_preset_id")
    calibrate_first = bool(payload.get("calibrate_first"))

    if not youtube_url:
        raise HTTPException(status_code=400, detail="Provide a youtube_url or use /sources/upload for local files.")
    if calibration_preset_id:
        try:
            store.load_calibration_preset(str(calibration_preset_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration preset not found.") from exc

    source = SourceSubmission(
        source_kind="youtube",
        source_name=match_name or "youtube-source",
        source_url=youtube_url,
        requested_match_name=match_name,
        calibration_preset_id=calibration_preset_id,
    )

    if calibrate_first:
        return {"match": _create_match_draft(source)}

    job = store.create_job(source)
    store.append_job_log(job.id, "Job queued.")
    threading.Thread(target=_run_job, args=(job.id,), daemon=True).start()
    return {"job": job.model_dump()}


@app.post("/sources/upload")
async def create_upload_source(request: Request) -> dict:
    upload_name = request.query_params.get("upload_name") or "upload.mp4"
    match_name = request.query_params.get("match_name")
    calibration_preset_id = request.query_params.get("calibration_preset_id")
    calibrate_first = request.query_params.get("calibrate_first") in {"1", "true", "yes"}
    upload_bytes = await request.body()

    if not upload_bytes:
        raise HTTPException(status_code=400, detail="Upload body is empty.")
    if calibration_preset_id:
        try:
            store.load_calibration_preset(calibration_preset_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration preset not found.") from exc

    upload_id, stored_path = store.save_upload(upload_name, upload_bytes)
    source = SourceSubmission(
        source_kind="upload",
        source_name=upload_name or f"upload-{upload_id}.mp4",
        stored_path=stored_path,
        requested_match_name=match_name,
        calibration_preset_id=calibration_preset_id,
    )

    if calibrate_first:
        return {"match": _create_match_draft(source)}

    job = store.create_job(source)
    store.append_job_log(job.id, "Upload received and job queued.")
    threading.Thread(target=_run_job, args=(job.id,), daemon=True).start()
    return {"job": job.model_dump()}


@app.post("/watchbot/start")
async def start_watchbot(payload: dict) -> dict:
    stream_url = payload.get("stream_url")
    if not stream_url:
        raise HTTPException(status_code=400, detail="stream_url is required.")
    state = watchbot.start(stream_url)
    return {"watchbot": state.model_dump()}


@app.post("/watchbot/stop")
async def stop_watchbot() -> dict:
    state = watchbot.stop()
    return {"watchbot": state.model_dump()}


@app.get("/watchbot")
async def get_watchbot() -> dict:
    return {"watchbot": store.load_watchbot_state().model_dump()}


@app.get("/matches")
async def list_matches() -> list[dict]:
    matches = store.list_matches()
    return [
        {
            "id": match.id,
            "created_at": match.created_at,
            "updated_at": match.updated_at,
            "metadata": match.metadata,
            "artifacts": match.artifacts.model_dump(),
            "labels": match.labels,
        }
        for match in matches
    ]


@app.get("/matches/{match_id}")
async def get_match(match_id: str) -> dict:
    try:
        return store.load_match(match_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Match not found.") from exc


@app.delete("/matches/{match_id}")
async def delete_match(match_id: str) -> dict:
    try:
        store.delete_match(match_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Match not found.") from exc
    for job in store.list_jobs():
        if job.match_id == match_id:
            try:
                store.delete_job(job.id)
            except FileNotFoundError:
                pass
    return {"deleted": True, "match_id": match_id}


@app.get("/matches/{match_id}/tracks")
async def get_match_tracks(match_id: str) -> dict:
    try:
        match = store.load_match(match_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Match not found.") from exc
    return {
        "match_id": match.id,
        "metadata": match.metadata,
        "tracks": [track.model_dump() for track in match.tracks],
        "detections": [detection.model_dump() for detection in match.detections],
        "labels": match.labels,
        "artifacts": match.artifacts.model_dump(),
    }


@app.get("/matches/{match_id}/calibration")
async def get_match_calibration(match_id: str) -> dict:
    try:
        match = store.load_match(match_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Match not found.") from exc

    return match.calibration.model_dump()


@app.put("/matches/{match_id}/calibration")
async def update_match_calibration(match_id: str, payload: dict) -> dict:
    try:
        match = store.load_match(match_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Match not found.") from exc

    views = payload.get("views")
    if not isinstance(views, list):
        raise HTTPException(status_code=400, detail="Calibration payload requires a list of views.")

    for view in views:
        landmarks = view.get("landmarks", [])
        roi = view.get("roi")
        if isinstance(roi, list) and len(landmarks) >= 4:
            parsed_landmarks = [FieldLandmark.model_validate(landmark) for landmark in landmarks[:4]]
            if not image_landmarks_form_valid_quad(roi, parsed_landmarks):
                raise HTTPException(
                    status_code=400,
                    detail="Each view needs four unique landmark image points inside the ROI before calibration can be saved.",
                )
            if not field_points_form_valid_quad([list(landmark.field_point) for landmark in parsed_landmarks]):
                raise HTTPException(
                    status_code=400,
                    detail="The selected field landmark targets must form a valid quadrilateral before calibration can be saved.",
                )
            homography, error = solve_view_homography(
                view.get("view", "main"),
                roi,
                parsed_landmarks,
                float(view.get("distortion_x", 0.0)),
                float(view.get("distortion_y", view.get("distortion_strength", 0.0))),
            )
            view["homography"] = homography
            view["reprojection_error"] = error

    match.calibration.mode = payload.get("mode", "manual_override")
    match.calibration.updated_at = time.time()
    match.calibration.quality_checks = payload.get("quality_checks", match.calibration.quality_checks)
    match.calibration.views = [ViewCalibration.model_validate(view) for view in payload["views"]]
    if match.detections or match.tracks:
        match = rebuild_match_tracking(match, store)
    else:
        store.save_match(match)
    return match.calibration.model_dump()


@app.post("/matches/{match_id}/start-processing")
async def start_match_processing(match_id: str) -> dict:
    try:
        match = store.load_match(match_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Match not found.") from exc

    if match.metadata.get("processing") is True:
        raise HTTPException(status_code=409, detail="Match is already processing.")

    source = SourceSubmission(
        source_kind=str(match.metadata.get("source_kind") or "upload"),  # type: ignore[arg-type]
        source_url=match.source.get("source_url") if isinstance(match.source, dict) else None,
        source_name=str(match.source.get("source_name") if isinstance(match.source, dict) else match.id),
        stored_path=match.source.get("stored_path") if isinstance(match.source, dict) else None,
        requested_match_name=str(match.metadata.get("display_name") or match.id),
        calibration_preset_id=str(match.metadata.get("calibration_preset_id") or "") or None,
    )
    job = store.create_job(source)
    job.match_id = match.id
    store.save_job(job)

    match.metadata["status"] = "queued"
    match.metadata["processing"] = False
    store.save_match(match)

    store.append_job_log(job.id, "Draft match queued for processing.")
    threading.Thread(target=_run_job, args=(job.id,), daemon=True).start()
    return {"job": job.model_dump(), "match": match.model_dump()}


@app.get("/calibration-presets")
async def list_calibration_presets() -> list[dict]:
    return [preset.model_dump() for preset in store.list_calibration_presets()]


@app.post("/calibration-presets")
async def create_calibration_preset(payload: dict) -> dict:
    name = str(payload.get("name") or "").strip()
    calibration_payload = payload.get("calibration")
    if not name:
        raise HTTPException(status_code=400, detail="Calibration preset name is required.")
    if not isinstance(calibration_payload, dict):
        raise HTTPException(status_code=400, detail="Calibration preset requires a calibration payload.")

    preset = CalibrationPreset(
        id=uuid.uuid4().hex,
        name=name,
        created_at=time.time(),
        updated_at=time.time(),
        calibration=CalibrationEnvelope.model_validate(calibration_payload),
    )
    store.save_calibration_preset(preset)
    return {"preset": preset.model_dump()}


@app.put("/matches/{match_id}/labels")
async def update_match_labels(match_id: str, payload: MatchLabelUpdate) -> dict:
    try:
        match = store.load_match(match_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Match not found.") from exc
    match.labels = payload.labels
    store.save_match(match)
    return {"labels": match.labels}


@app.get("/matches/{match_id}/fuel")
async def get_match_fuel(match_id: str) -> dict:
    try:
        match = store.load_match(match_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Match not found.") from exc
    return {
        "match_id": match.id,
        "fuel_calibration": match.fuel_calibration.model_dump(),
        "fuel_analysis": match.fuel_analysis.model_dump(),
    }


@app.get("/matches/{match_id}/fuel-calibration")
async def get_match_fuel_calibration(match_id: str) -> dict:
    try:
        match = store.load_match(match_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Match not found.") from exc
    return match.fuel_calibration.model_dump()


@app.put("/matches/{match_id}/fuel-calibration")
async def update_match_fuel_calibration(match_id: str, payload: dict) -> dict:
    try:
        match = store.load_match(match_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Match not found.") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Fuel calibration payload must be a JSON object.")

    quad_labels = {
        "ground_quad": "ground corners",
        "left_wall_quad": "left wall",
        "right_wall_quad": "right wall",
    }
    for field_name, label in quad_labels.items():
        if field_name not in payload:
            continue
        value = payload[field_name]
        if value is None:
            setattr(match.fuel_calibration, field_name, None)
            continue
        normalized = normalize_quad(value)
        if normalized is None:
            raise HTTPException(
                status_code=400,
                detail=f"Fuel calibration for {label} must be four normalized points inside the video frame that form a valid quadrilateral.",
            )
        setattr(match.fuel_calibration, field_name, normalized)

    if "fuel_base_color" in payload:
        normalized_color = normalize_rgb_color(payload.get("fuel_base_color"))
        if normalized_color is None:
            raise HTTPException(status_code=400, detail="fuel_base_color must be a list like [255, 255, 0].")
        match.fuel_calibration.fuel_base_color = normalized_color

    match.fuel_calibration.updated_at = time.time()
    invalidate_fuel_analysis(match)
    store.save_match(match)
    return match.fuel_calibration.model_dump()


@app.put("/matches/{match_id}/fuel/base-color")
async def update_match_fuel_base_color(match_id: str, payload: dict) -> dict:
    try:
        match = store.load_match(match_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Match not found.") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Fuel base color payload must be a JSON object.")

    normalized_color = normalize_rgb_color(payload.get("fuel_base_color"))
    if normalized_color is None:
        color_object = payload.get("fuelBaseColor")
        if isinstance(color_object, dict):
            normalized_color = normalize_rgb_color(
                [color_object.get("r"), color_object.get("g"), color_object.get("b")]
            )
    if normalized_color is None:
        raise HTTPException(
            status_code=400,
            detail="Provide fuel_base_color as [r, g, b] or fuelBaseColor as { r, g, b }.",
        )

    match.fuel_calibration.fuel_base_color = normalized_color
    match.fuel_calibration.updated_at = time.time()
    invalidate_fuel_analysis(match)
    store.save_match(match)
    return {
        "fuel_base_color": match.fuel_calibration.fuel_base_color,
        "fuel_calibration": match.fuel_calibration.model_dump(),
    }


@app.post("/matches/{match_id}/fuel/base-color/sample")
async def sample_match_fuel_base_color(match_id: str, payload: dict) -> dict:
    try:
        match = store.load_match(match_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Match not found.") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Fuel color sample payload must be a JSON object.")

    x = payload.get("x")
    y = payload.get("y")
    time_sec = payload.get("time_sec", payload.get("timeSec", 0.0))
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)) or not isinstance(time_sec, (int, float)):
        raise HTTPException(status_code=400, detail="x, y, and time_sec must be numeric.")

    try:
        sampled_color = sample_match_video_color(match, float(x), float(y), float(time_sec))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    match.fuel_calibration.fuel_base_color = sampled_color
    match.fuel_calibration.updated_at = time.time()
    invalidate_fuel_analysis(match)
    store.save_match(match)
    return {
        "fuel_base_color": sampled_color,
        "fuel_calibration": match.fuel_calibration.model_dump(),
    }


@app.post("/matches/{match_id}/fuel/process")
async def process_match_fuel(match_id: str) -> dict:
    try:
        match = store.load_match(match_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Match not found.") from exc

    try:
        match = await asyncio.to_thread(run_fuel_analysis, match, store)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "match_id": match.id,
        "fuel_calibration": match.fuel_calibration.model_dump(),
        "fuel_analysis": match.fuel_analysis.model_dump(),
    }


@app.get("/tba/team/{team_key}/matches")
async def get_team_matches(team_key: str, year: int) -> dict:
    api_key = os.getenv("TBA_API_KEY", DEFAULT_TBA_API_KEY)
    if not api_key:
        raise HTTPException(status_code=503, detail="TBA_API_KEY is not configured on the backend.")

    normalized = team_key if team_key.startswith("frc") else f"frc{team_key}"
    url = f"https://www.thebluealliance.com/api/v3/team/{urllib.parse.quote(normalized)}/matches/{year}"

    try:
        response = requests.get(
            url,
            headers={
                "X-TBA-Auth-Key": api_key,
                "User-Agent": "FRC-AI-Obj-Tracker/1.0",
                "Accept": "application/json",
            },
            timeout=20,
        )
        if response.status_code == 403:
            raise HTTPException(
                status_code=502,
                detail="TBA rejected the request with 403 Forbidden. Check that the backend is using a valid TBA API key.",
            )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("Error"):
            raise HTTPException(status_code=502, detail=f"TBA error: {payload['Error']}")
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=502, detail=f"Failed to fetch TBA data: {exc}") from exc

    matches = []
    for match in payload:
      matches.append(
          {
              "key": match.get("key"),
              "event_key": match.get("event_key"),
              "comp_level": match.get("comp_level"),
              "match_number": match.get("match_number"),
              "set_number": match.get("set_number"),
              "alliances": match.get("alliances"),
              "videos": match.get("videos", []),
          }
      )

    return {"team_key": normalized, "year": year, "matches": matches}
