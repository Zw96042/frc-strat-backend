from __future__ import annotations

from pathlib import Path

from tracking_service import fuel


def test_fuel_processor_backend_defaults_to_cpu_when_cuda_runtime_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.delenv("FUEL_PROCESSOR_BACKEND", raising=False)
    monkeypatch.delenv("FUEL_PROCESSOR_PYTHON_BIN", raising=False)

    root = tmp_path / "fuel-density-map"
    cuda_python = root / ".venv-opencv-cuda" / "bin" / "python"
    cuda_python.parent.mkdir(parents=True, exist_ok=True)
    cuda_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(fuel, "FUEL_DENSITY_MAP_ROOT", root)
    monkeypatch.setattr(fuel, "_processor_python_supports_cuda", lambda _python_bin: False)

    assert fuel.fuel_processor_backend() == "cpu"


def test_fuel_processor_backend_prefers_override_python_when_it_reports_cuda(monkeypatch):
    monkeypatch.delenv("FUEL_PROCESSOR_BACKEND", raising=False)
    monkeypatch.setenv("FUEL_PROCESSOR_PYTHON_BIN", "/tmp/custom-fuel-python")
    monkeypatch.setattr(fuel, "_processor_python_supports_cuda", lambda python_bin: python_bin == "/tmp/custom-fuel-python")

    assert fuel.fuel_processor_backend() == "cuda"


def test_processor_python_bin_prefers_cpu_virtualenv_when_backend_is_cpu(monkeypatch, tmp_path):
    monkeypatch.delenv("FUEL_PROCESSOR_PYTHON_BIN", raising=False)

    root = tmp_path / "fuel-density-map"
    cpu_python = root / ".venv" / "bin" / "python"
    cuda_python = root / ".venv-opencv-cuda" / "bin" / "python"
    cpu_python.parent.mkdir(parents=True, exist_ok=True)
    cuda_python.parent.mkdir(parents=True, exist_ok=True)
    cpu_python.write_text("", encoding="utf-8")
    cuda_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(fuel, "FUEL_DENSITY_MAP_ROOT", root)
    monkeypatch.setattr(fuel, "fuel_processor_backend", lambda: "cpu")

    assert Path(fuel.processor_python_bin()) == cpu_python
