import importlib

import pytest


@pytest.fixture(scope="module")
def app_module():
    return importlib.import_module("ai_data_visualisation_agent")


def test_module_imports(app_module):
    assert app_module.APP_TITLE == "AI Data Visualization Agent"
    assert callable(app_module.main)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("sales report.csv", "sales_report.csv"),
        ("../../private.csv", "private.csv"),
        ("...", "data.csv"),
    ],
)
def test_uploaded_filename_is_sanitized(app_module, source, expected):
    assert app_module._sanitize_filename(source) == expected


def test_chart_prompt_detection(app_module):
    assert app_module._request_needs_chart("Plot revenue by month")
    assert not app_module._request_needs_chart("List the missing columns")


def test_demo_mode_uses_bundled_data_and_local_results(app_module, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")

    assert app_module._is_demo_mode()
    assert app_module._current_secrets() == app_module.Secrets("", "")

    frame = app_module._load_csv(app_module.DEMO_DATASET_BYTES)
    run_record = app_module._build_demo_run_record("Show the monthly revenue trend", frame)

    assert not frame.empty
    assert run_record["assistant_text"]
    assert run_record["results"]
    assert run_record["python_code"] == ""
