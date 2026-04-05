.PHONY: venv install install-dev run lint

venv:
	python3 -m venv .venv

install: venv
	. .venv/bin/activate && pip install -r requirements.txt

install-dev: install
	. .venv/bin/activate && pip install -r requirements-dev.txt

run:
	. .venv/bin/activate && streamlit run ai_data_visualisation_agent.py

lint:
	. .venv/bin/activate && ruff check .

