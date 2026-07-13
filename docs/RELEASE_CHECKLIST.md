# Public Demo Release Checklist

This checklist describes the release boundary for the current Streamlit application. It does not
indicate that a hosted deployment exists.

## Before every release

- Confirm the working tree contains only intended release changes.
- Create a fresh Python 3.11 or 3.12 virtual environment.
- Install `requirements-dev.txt` and run the verification commands below.
- Review dependency updates before regenerating `requirements-lock.txt`.
- Confirm `.streamlit/secrets.toml` remains untracked.
- Set `DEMO_MODE=true` for any deployment reachable by untrusted users.
- Confirm public demo instances do not contain Ollama Cloud or E2B credentials.
- Disable unnecessary network egress and Streamlit usage telemetry for the public demo.
- If live mode is used in a controlled setting, use approved data and dedicated provider accounts
  with appropriate quotas and spending limits.

## Verification commands

```bash
python -m pip install --disable-pip-version-check -r requirements-dev.txt
python -m pip check
python -m ruff check .
python -m pytest
python -c "import ai_data_visualisation_agent as app; assert callable(app.main)"
```

For an interactive smoke test:

```bash
DEMO_MODE=true streamlit run ai_data_visualisation_agent.py \
  --server.headless true \
  --browser.gatherUsageStats false
```

Confirm the bundled synthetic dataset appears in AI Workspace and Dataset Lab, then try a trend,
regional comparison, and data-quality prompt. The demo must not request API keys or accept uploads.

## Public exposure gate

Only deterministic `DEMO_MODE=true` is intended for untrusted public traffic. Do not expose live
mode until the deployment has authentication, reviewed session isolation, request and file-size
limits, retention controls, provider credential isolation, and abuse/cost controls.
