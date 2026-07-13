from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
app = importlib.import_module("ai_data_visualisation_agent")


class SessionState(dict):
    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: object) -> None:
        self[name] = value


def test_configured_secrets_are_not_copied_into_session_defaults(monkeypatch) -> None:
    configured = {
        "OLLAMA_API_KEY": "server-ollama-secret",
        "E2B_API_KEY": "server-e2b-secret",
    }
    session_state = SessionState()
    monkeypatch.setattr(app.st, "session_state", session_state)
    monkeypatch.setattr(app, "_is_demo_mode", lambda: False)
    monkeypatch.setattr(app, "_get_secret", lambda name: configured.get(name, ""))

    app._init_state()

    assert session_state["ollama_api_key"] == ""
    assert session_state["e2b_api_key"] == ""
    assert app._current_secrets() == app.Secrets(
        ollama_api_key="server-ollama-secret",
        e2b_api_key="server-e2b-secret",
    )

    session_state["ollama_api_key"] = "session-ollama-key"
    session_state["e2b_api_key"] = "session-e2b-key"
    assert app._current_secrets() == app.Secrets(
        ollama_api_key="session-ollama-key",
        e2b_api_key="session-e2b-key",
    )


def test_dataset_state_stays_in_session_without_creating_files(monkeypatch, tmp_path) -> None:
    session_state = SessionState()
    monkeypatch.setattr(app.st, "session_state", session_state)
    monkeypatch.setattr(app, "_is_demo_mode", lambda: False)
    monkeypatch.chdir(tmp_path)
    app._init_state()

    app._set_active_dataset("private.csv", b"value\n42\n")

    assert session_state["dataset_name"] == "private.csv"
    assert session_state["dataset_bytes"] == b"value\n42\n"
    assert list(tmp_path.iterdir()) == []


def test_demo_mode_disables_all_credentials(monkeypatch) -> None:
    monkeypatch.setattr(
        app.st,
        "session_state",
        SessionState(
            ollama_api_key="session-ollama-key",
            e2b_api_key="session-e2b-key",
        ),
    )
    monkeypatch.setattr(app, "_is_demo_mode", lambda: True)

    assert app._current_secrets() == app.Secrets(ollama_api_key="", e2b_api_key="")
    assert app._current_model() == app.DEMO_MODEL


def test_demo_results_are_deterministic() -> None:
    demo_df = app._load_csv(app.DEMO_DATASET_BYTES)

    first = app._build_demo_run_record("Show the monthly revenue trend", demo_df)
    second = app._build_demo_run_record("Show the monthly revenue trend", demo_df)
    quality = app._build_demo_run_record("Audit missing values and duplicates", demo_df)

    assert first == second
    assert first["results"][0]["chart"]["type"] == "line"
    assert "increase" in first["assistant_text"]
    assert quality["results"][0]["kind"] == "table"
    assert "0 missing cells and 0 duplicate rows" in quality["assistant_text"]
