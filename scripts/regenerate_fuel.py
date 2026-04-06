#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracking_service.fuel import invalidate_fuel_analysis, run_fuel_analysis
from tracking_service.storage import TrackingStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate fuel artifacts for already-processed matches.",
    )
    parser.add_argument(
        "--source-prefix",
        type=Path,
        default=REPO_ROOT / "data" / "watchbot" / "clips",
        help="Only include matches whose stored source path is under this directory.",
    )
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=["ready", "error"],
        help="Fuel statuses to regenerate. Default: ready error",
    )
    parser.add_argument(
        "--include-completed",
        action="store_true",
        help="Also rerun matches whose fuel already completed.",
    )
    parser.add_argument(
        "--builtin-only",
        action="store_true",
        help="Skip the external fuel-density-map processor and use the built-in generator only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of matches to regenerate.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List eligible matches without running regeneration.",
    )
    return parser.parse_args()


def is_processed_match(match) -> bool:
    return bool(match.tracks) or bool(match.artifacts.annotated_video) or bool(match.artifacts.source_video)


def stored_path_for(match) -> str | None:
    if not isinstance(match.source, dict):
        return None
    value = match.source.get("stored_path")
    return str(value) if value else None


def display_name_for(match) -> str:
    return str(match.metadata.get("display_name") or match.source.get("source_name") or match.id)


def iter_candidates(store: TrackingStore, source_prefix: Path, statuses: set[str]):
    resolved_prefix = source_prefix.expanduser().resolve()
    for match in store.list_matches():
        stored_path = stored_path_for(match)
        if not stored_path:
            continue
        source_path = Path(stored_path).expanduser().resolve()
        try:
            source_path.relative_to(resolved_prefix)
        except ValueError:
            continue
        if not source_path.exists():
            continue
        if not match.fuel_calibration.ground_quad:
            continue
        if not is_processed_match(match):
            continue
        if match.fuel_analysis.status not in statuses:
            continue
        yield match


def main() -> int:
    args = parse_args()
    if args.builtin_only:
        os.environ["FUEL_PROCESSOR_DISABLE_EXTERNAL"] = "1"

    statuses = {status.strip().lower() for status in args.statuses if status.strip()}
    if args.include_completed:
        statuses.add("completed")

    store = TrackingStore()
    candidates = list(iter_candidates(store, args.source_prefix, statuses))
    if args.limit > 0:
        candidates = candidates[: args.limit]

    print(
        f"Eligible matches: {len(candidates)} under {args.source_prefix.expanduser().resolve()} "
        f"for statuses={sorted(statuses)}",
        flush=True,
    )
    for match in candidates:
        print(
            f" - {match.id} | {display_name_for(match)} | fuel={match.fuel_analysis.status}",
            flush=True,
        )

    if args.list_only:
        return 0

    completed = 0
    failed = 0
    started_at = time.time()

    for match in candidates:
        print(f"[{match.id}] regenerating fuel for {display_name_for(match)}", flush=True)
        try:
            invalidate_fuel_analysis(match)
            store.save_match(match)
            refreshed = store.load_match(match.id)
            refreshed = run_fuel_analysis(refreshed, store)
            field_map = refreshed.fuel_analysis.artifacts.field_map or "missing"
            print(f"[{match.id}] fuel complete -> {field_map}", flush=True)
            completed += 1
        except Exception as exc:
            print(f"[{match.id}] fuel failed -> {exc}", flush=True)
            failed += 1

    elapsed = time.time() - started_at
    print(
        f"Finished fuel regeneration in {elapsed:.1f}s | completed={completed} failed={failed}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
