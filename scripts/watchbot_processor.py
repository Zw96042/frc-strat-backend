#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracking_service.config import WATCHBOT_ROOT, ensure_data_dirs
from tracking_service.fuel import invalidate_fuel_analysis, run_fuel_analysis
from tracking_service.pipeline import process_job
from tracking_service.schemas import SourceSubmission
from tracking_service.storage import TrackingStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch the watchbot capture directory and process completed videos.",
    )
    parser.add_argument(
        "--watch-dir",
        type=Path,
        default=WATCHBOT_ROOT,
        help=f"Directory to watch. Default: {WATCHBOT_ROOT}",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively watch nested event folders under the watch directory.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="How often to rescan the directory.",
    )
    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=15.0,
        help="How long an mp4 must stay unchanged before it is processed.",
    )
    parser.add_argument(
        "--with-fuel",
        action="store_true",
        help="Also run fuel analysis after match processing completes.",
    )
    parser.add_argument(
        "--calibration-preset-id",
        help="Robot calibration preset id to apply before match processing.",
    )
    parser.add_argument(
        "--calibration-preset-name",
        help="Robot calibration preset name to apply before match processing.",
    )
    parser.add_argument(
        "--fuel-preset-id",
        help="Fuel calibration preset id to apply before fuel processing.",
    )
    parser.add_argument(
        "--fuel-preset-name",
        help="Fuel calibration preset name to apply before fuel processing.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process all currently stable videos once, then exit.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional limit on how many videos to process in this run. Default: unlimited.",
    )
    return parser.parse_args()


def iter_mp4s(watch_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.mp4" if recursive else "*.mp4"
    return sorted(
        (
            path
            for path in watch_dir.glob(pattern)
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
    )


def find_existing_job(store: TrackingStore, source_path: Path):
    target = str(source_path.resolve())
    for job in store.list_jobs():
        if job.source.stored_path and str(Path(job.source.stored_path).resolve()) == target:
            return job
    return None


def find_existing_match(store: TrackingStore, source_path: Path):
    target = str(source_path.resolve())
    for match in store.list_matches():
        stored_path = match.source.get("stored_path") if isinstance(match.source, dict) else None
        if stored_path and str(Path(stored_path).resolve()) == target:
            return match
    return None


def make_requested_match_name(source_path: Path, watch_dir: Path) -> str:
    try:
        relative_parent = source_path.resolve().parent.relative_to(watch_dir.resolve())
    except ValueError:
        relative_parent = source_path.parent

    if str(relative_parent) in {"", "."}:
        return f"Watchbot Capture {source_path.stem}"
    return f"{relative_parent.as_posix()} {source_path.stem}"


def resolve_robot_calibration_preset_id(store: TrackingStore, preset_id: str | None, preset_name: str | None) -> str | None:
    if preset_id:
        store.load_calibration_preset(preset_id)
        return preset_id
    if not preset_name:
        return None

    normalized = preset_name.strip().lower()
    for preset in store.list_calibration_presets():
        if preset.name.strip().lower() == normalized:
            return preset.id
    raise ValueError(f'Robot calibration preset not found: "{preset_name}"')


def resolve_fuel_calibration_preset(store: TrackingStore, preset_id: str | None, preset_name: str | None):
    if preset_id:
        return store.load_fuel_calibration_preset(preset_id)
    if not preset_name:
        return None

    normalized = preset_name.strip().lower()
    for preset in store.list_fuel_calibration_presets():
        if preset.name.strip().lower() == normalized:
            return preset
    raise ValueError(f'Fuel calibration preset not found: "{preset_name}"')


def process_capture(
    store: TrackingStore,
    source_path: Path,
    watch_dir: Path,
    with_fuel: bool,
    calibration_preset_id: str | None,
    fuel_preset,
) -> bool:
    existing_job = find_existing_job(store, source_path)
    if existing_job is not None:
        if existing_job.status in {"queued", "running"}:
            print(f"[skip] {source_path} already has active job {existing_job.id}", flush=True)
            return False
        if existing_job.status == "completed" and existing_job.match_id:
            print(f"[skip] {source_path} already completed as match {existing_job.match_id}", flush=True)
            return False

    existing_match = find_existing_match(store, source_path)
    if existing_match is not None:
        print(f"[skip] {source_path} already has match {existing_match.id}", flush=True)
        return False

    source = SourceSubmission(
        source_kind="upload",
        source_name=source_path.name,
        stored_path=str(source_path),
        requested_match_name=make_requested_match_name(source_path, watch_dir),
        calibration_preset_id=calibration_preset_id,
    )
    job = store.create_job(source)
    store.append_job_log(job.id, "Queued by watchbot processor worker.")
    print(f"[{job.id}] processing {source_path}", flush=True)

    try:
        match = process_job(store.load_job(job.id), store)
        if fuel_preset is not None:
            match.fuel_calibration = fuel_preset.fuel_calibration.model_copy(deep=True)
            match.fuel_calibration.updated_at = time.time()
            invalidate_fuel_analysis(match)
            store.save_match(match)
        print(f"[{job.id}] match complete: {match.id}", flush=True)
        if with_fuel:
            print(f"[{job.id}] fuel processing starting: {match.id}", flush=True)
            match = run_fuel_analysis(match, store)
            print(f"[{job.id}] fuel complete: {match.id}", flush=True)
    except Exception as exc:
        failed_job = store.load_job(job.id)
        failed_job.status = "failed"
        failed_job.error = str(exc)
        store.save_job(failed_job)
        store.append_job_log(job.id, f"Worker failed: {exc}", level="error")
        print(f"[{job.id}] failed: {exc}", flush=True)
    return True


def main() -> int:
    args = parse_args()
    ensure_data_dirs()
    watch_dir = args.watch_dir.expanduser().resolve()
    watch_dir.mkdir(parents=True, exist_ok=True)

    store = TrackingStore()
    observed: dict[str, tuple[int, float]] = {}
    processed_count = 0
    calibration_preset_id = resolve_robot_calibration_preset_id(
        store,
        args.calibration_preset_id,
        args.calibration_preset_name,
    )
    fuel_preset = resolve_fuel_calibration_preset(
        store,
        args.fuel_preset_id,
        args.fuel_preset_name,
    )

    print(
        f"Watching {watch_dir} for stable mp4 captures every {args.poll_seconds:.1f}s "
        f"(stable after {args.stable_seconds:.1f}s).",
        flush=True,
    )

    while True:
        for path in iter_mp4s(watch_dir, recursive=args.recursive):
            resolved = path.resolve()
            stat = resolved.stat()
            key = str(resolved)

            if args.once:
                if process_capture(store, resolved, watch_dir, args.with_fuel, calibration_preset_id, fuel_preset):
                    processed_count += 1
                if args.max_files > 0 and processed_count >= args.max_files:
                    print(f"Reached max-files={args.max_files}; exiting.", flush=True)
                    return 0
                continue

            previous = observed.get(key)
            if previous is None or previous[0] != stat.st_size:
                observed[key] = (stat.st_size, time.time())
                continue

            size_bytes, stable_since = previous
            if time.time() - stable_since < args.stable_seconds:
                continue

            if process_capture(store, resolved, watch_dir, args.with_fuel, calibration_preset_id, fuel_preset):
                processed_count += 1
            observed.pop(key, None)

            if args.max_files > 0 and processed_count >= args.max_files:
                print(f"Reached max-files={args.max_files}; exiting.", flush=True)
                return 0
        if args.once:
            print(f"One-shot scan complete. Processed {processed_count} file(s).", flush=True)
            return 0

        time.sleep(max(args.poll_seconds, 0.5))


if __name__ == "__main__":
    raise SystemExit(main())
