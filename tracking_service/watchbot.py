from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

from .pipeline import capture_stream_segment, process_job
from .schemas import SourceSubmission, WatchbotState
from .storage import TrackingStore


class WatchbotManager:
    def __init__(self, store: TrackingStore) -> None:
        self.store = store
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self, stream_url: str, calibration_preset_id: str | None = None) -> WatchbotState:
        state = WatchbotState(
            active=True,
            stream_url=stream_url,
            started_at=time.time(),
            updated_at=time.time(),
            last_message="Watchbot armed with template start detection.",
            capture_directory=str((Path(__file__).resolve().parent.parent / "data" / "watchbot").resolve()),
        )
        self.store.save_watchbot_state(state)
        self._stop_event.clear()

        def runner() -> None:
            capture_path = Path(state.capture_directory or ".") / f"watchbot-{int(time.time())}.mp4"
            ok, result = capture_stream_segment(stream_url, str(capture_path), self._stop_event.is_set)
            latest_state = self.store.load_watchbot_state()
            if not latest_state.active:
                return
            if not ok:
                latest_state.last_message = result
                latest_state.updated_at = time.time()
                latest_state.active = False
                self.store.save_watchbot_state(latest_state)
                return

            source = SourceSubmission(
                source_kind="watchbot",
                source_url=stream_url,
                source_name=capture_path.name,
                stored_path=str(capture_path),
                requested_match_name=f"Watchbot Capture {capture_path.stem}",
                calibration_preset_id=calibration_preset_id,
            )
            job = self.store.create_job(source)
            self.store.append_job_log(job.id, "Watchbot captured a match segment and enqueued processing.")
            try:
                process_job(self.store.load_job(job.id), self.store)
                latest_state.last_message = f"Processed match from {capture_path.name}"
            except Exception as exc:
                failed_job = self.store.load_job(job.id)
                failed_job.status = "failed"
                failed_job.error = str(exc)
                self.store.save_job(failed_job)
                latest_state.last_message = f"Watchbot processing failed: {exc}"
            latest_state.updated_at = time.time()
            latest_state.active = False
            self.store.save_watchbot_state(latest_state)

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        return state

    def stop(self) -> WatchbotState:
        self._stop_event.set()
        state = self.store.load_watchbot_state()
        state.active = False
        state.updated_at = time.time()
        state.last_message = "Watchbot stopped."
        self.store.save_watchbot_state(state)
        return state
