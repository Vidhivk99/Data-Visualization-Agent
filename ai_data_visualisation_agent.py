import base64
import hashlib
import io
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from e2b_code_interpreter import Sandbox
from PIL import Image


APP_TITLE = "AI Data Visualization Agent"
APP_SUBTITLE = (
    "Upload a CSV, explore the data, and run analysis from one place."
)
APP_STATE_DIR = os.path.join(".streamlit", "app_state")
WORKSPACE_STATE_PATH = os.path.join(APP_STATE_DIR, "workspace_state.json")
DATASET_STATE_PATH = os.path.join(APP_STATE_DIR, "active_dataset.csv")
INTRO_MESSAGE = (
    "Upload a CSV and tell me what you want to explore. I can help with trends, comparisons, outliers, "
    "and quick data-quality checks."
)


@dataclass(frozen=True)
class ModelOption:
    label: str
    id: str
    api_candidates: tuple[str, ...]
    summary: str
    best_for: str


MODEL_CATALOG: tuple[ModelOption, ...] = (
    ModelOption(
        label="qwen3-coder:480b-cloud",
        id="qwen3-coder:480b-cloud",
        api_candidates=("qwen3-coder:480b-cloud", "qwen3-coder:480b"),
        summary="Large coding-oriented cloud model for the most complex analysis tasks.",
        best_for="Complex analysis plans, code generation, and harder dataset requests.",
    ),
    ModelOption(
        label="gpt-oss:120b-cloud",
        id="gpt-oss:120b-cloud",
        api_candidates=("gpt-oss:120b-cloud", "gpt-oss:120b"),
        summary="Large general-purpose cloud model with strong reasoning depth.",
        best_for="Deep reasoning, multi-step chart requests, and broader analysis prompts.",
    ),
    ModelOption(
        label="gpt-oss:20b-cloud",
        id="gpt-oss:20b-cloud",
        api_candidates=("gpt-oss:20b-cloud", "gpt-oss:20b"),
        summary="Faster cloud model for lighter prompts and quicker iterations.",
        best_for="Shorter analysis loops, focused questions, and faster responses.",
    ),
    ModelOption(
        label="deepseek-v3.1:671b-cloud",
        id="deepseek-v3.1:671b-cloud",
        api_candidates=("deepseek-v3.1:671b-cloud", "deepseek-v3.1:671b"),
        summary="Very large reasoning model for detailed explanations and heavier prompts.",
        best_for="Detailed audits, insight-heavy reporting, and broad reasoning tasks.",
    ),
)

MODEL_BY_ID = {model.id: model for model in MODEL_CATALOG}

CODE_BLOCK_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\n(?P<code>.*?)\n```", re.DOTALL)
CHART_REQUEST_RE = re.compile(
    r"\b(plot|chart|graph|visuali[sz]e|visuali[sz]ation|histogram|scatter|bar chart|line chart|box plot|distribution|trend)\b",
    re.IGNORECASE,
)


try:
    from urllib3.exceptions import NotOpenSSLWarning

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass


@dataclass(frozen=True)
class Secrets:
    ollama_api_key: str
    e2b_api_key: str


@dataclass(frozen=True)
class DatasetProfile:
    rows: int
    columns: int
    missing_cells: int
    duplicate_rows: int
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    datetime_like_columns: tuple[str, ...]
    memory_usage_mb: float
    completeness_ratio: float
    top_missing_columns: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ActiveDataset:
    name: str
    file_bytes: bytes
    df: pd.DataFrame
    profile: DatasetProfile


def _get_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, "")


def _default_chat_history() -> list[dict[str, str]]:
    return [{"role": "assistant", "content": INTRO_MESSAGE}]


def _ensure_app_state_dir() -> None:
    os.makedirs(APP_STATE_DIR, exist_ok=True)


def _persist_workspace_state() -> None:
    _ensure_app_state_dir()
    payload = {
        "ollama_api_key": str(st.session_state.get("ollama_api_key") or ""),
        "e2b_api_key": str(st.session_state.get("e2b_api_key") or ""),
        "model_id": str(st.session_state.get("model_id") or MODEL_CATALOG[0].id),
        "dataset_name": st.session_state.get("dataset_name"),
        "dataset_token": st.session_state.get("dataset_token"),
        "chat": st.session_state.get("chat") or _default_chat_history(),
        "analysis_runs": st.session_state.get("analysis_runs") or [],
        "pending_prompt": str(st.session_state.get("pending_prompt") or ""),
    }
    with open(WORKSPACE_STATE_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)

    dataset_bytes = st.session_state.get("dataset_bytes")
    if st.session_state.get("dataset_name") and isinstance(dataset_bytes, (bytes, bytearray)):
        with open(DATASET_STATE_PATH, "wb") as handle:
            handle.write(bytes(dataset_bytes))
    elif os.path.exists(DATASET_STATE_PATH):
        os.remove(DATASET_STATE_PATH)


def _load_persisted_workspace_state() -> None:
    if not os.path.exists(WORKSPACE_STATE_PATH):
        return

    try:
        with open(WORKSPACE_STATE_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return

    model_id = str(payload.get("model_id") or "")
    if model_id in MODEL_BY_ID:
        st.session_state.model_id = model_id

    st.session_state.ollama_api_key = str(payload.get("ollama_api_key") or st.session_state.get("ollama_api_key") or "")
    st.session_state.e2b_api_key = str(payload.get("e2b_api_key") or st.session_state.get("e2b_api_key") or "")
    st.session_state.dataset_name = payload.get("dataset_name")
    st.session_state.dataset_token = payload.get("dataset_token")
    st.session_state.chat = payload.get("chat") or _default_chat_history()
    st.session_state.analysis_runs = payload.get("analysis_runs") or []
    st.session_state.pending_prompt = str(payload.get("pending_prompt") or "")

    if os.path.exists(DATASET_STATE_PATH) and st.session_state.dataset_name:
        try:
            with open(DATASET_STATE_PATH, "rb") as handle:
                st.session_state.dataset_bytes = handle.read()
        except Exception:
            st.session_state.dataset_bytes = None
    else:
        st.session_state.dataset_bytes = None


def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
    return name or "data.csv"


def _dataset_runtime_path(filename: str) -> str:
    return f"./{_sanitize_filename(filename)}"


def _file_token(filename: str, file_bytes: bytes) -> str:
    digest = hashlib.md5(file_bytes).hexdigest()
    return f"{filename}:{digest}"


def _extract_code_blocks(markdown: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for match in CODE_BLOCK_RE.finditer(markdown or ""):
        lang = (match.group("lang") or "").strip().lower()
        code = (match.group("code") or "").strip()
        if code:
            blocks.append((lang, code))
    return blocks


def _pick_python_code(markdown: str) -> str:
    blocks = _extract_code_blocks(markdown)
    for lang, code in blocks:
        if lang in {"python", "py"}:
            return code
    if blocks:
        _, code = blocks[0]
        return code
    return ""


def _strip_code_blocks(markdown: str) -> str:
    cleaned = CODE_BLOCK_RE.sub("", markdown or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _request_needs_chart(user_text: str) -> bool:
    return bool(CHART_REQUEST_RE.search(user_text or ""))


def _augment_user_text_for_model(user_text: str) -> str:
    if not _request_needs_chart(user_text):
        return user_text
    return (
        f"{user_text}\n\n"
        "Important: create at least one visible matplotlib chart for this request. "
        "Use clear labels and a title, call plt.tight_layout(), and call plt.show()."
    )


def _ensure_chart_display(code: str, user_text: str) -> str:
    lowered = (code or "").lower()
    if not _request_needs_chart(user_text):
        return code
    if "matplotlib" not in lowered and "plt." not in lowered:
        return code
    if "plt.show(" in lowered or ".show(" in lowered:
        return code
    return (
        f"{code.rstrip()}\n\n"
        "import matplotlib.pyplot as plt\n"
        "if plt.get_fignums():\n"
        "    try:\n"
        "        plt.tight_layout()\n"
        "    except Exception:\n"
        "        pass\n"
        "    plt.show()\n"
    )


@st.cache_data(show_spinner=False)
def _load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)


@st.cache_data(show_spinner=False)
def _profile_dataset(file_bytes: bytes) -> DatasetProfile:
    df = _load_csv(file_bytes)
    missing_cells = int(df.isna().sum().sum())
    missing_by_col = df.isna().sum().sort_values(ascending=False)
    top_missing_columns = tuple(
        (str(column), int(count)) for column, count in missing_by_col[missing_by_col > 0].head(6).items()
    )
    numeric_columns = tuple(str(column) for column in df.select_dtypes(include="number").columns.tolist())
    categorical_columns = tuple(str(column) for column in df.select_dtypes(exclude="number").columns.tolist())
    datetime_like_columns = tuple(
        str(column)
        for column in df.columns
        if re.search(r"(date|time|year|month|day)", str(column), re.IGNORECASE)
    )
    total_cells = max(int(df.shape[0] * max(df.shape[1], 1)), 1)
    completeness_ratio = 1 - (missing_cells / total_cells)
    memory_usage_mb = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)
    return DatasetProfile(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        missing_cells=missing_cells,
        duplicate_rows=int(df.duplicated().sum()),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_like_columns=datetime_like_columns,
        memory_usage_mb=memory_usage_mb,
        completeness_ratio=completeness_ratio,
        top_missing_columns=top_missing_columns,
    )


@st.cache_data(show_spinner=False)
def _column_metadata(file_bytes: bytes) -> pd.DataFrame:
    df = _load_csv(file_bytes)
    meta = pd.DataFrame(
        {
            "column": [str(column) for column in df.columns],
            "dtype": df.dtypes.astype(str).tolist(),
            "missing": df.isna().sum().tolist(),
            "missing_pct": (df.isna().mean().mul(100).round(1)).tolist(),
            "unique_values": df.nunique(dropna=True).tolist(),
        }
    )
    return meta.sort_values(["missing", "unique_values"], ascending=[False, False]).reset_index(drop=True)


def _dataset_brief(df: pd.DataFrame, profile: DatasetProfile, max_cols: int = 28) -> str:
    columns = list(df.columns)[:max_cols]
    dtypes = {str(column): str(df[column].dtype) for column in columns}
    head = df[columns].head(5).to_dict(orient="records")
    return (
        f"rows={profile.rows}, columns={profile.columns}\n"
        f"column_names={[str(column) for column in columns]}\n"
        f"dtypes={dtypes}\n"
        f"missing_cells={profile.missing_cells}, duplicate_rows={profile.duplicate_rows}\n"
        f"numeric_columns={list(profile.numeric_columns[:10])}\n"
        f"categorical_columns={list(profile.categorical_columns[:10])}\n"
        f"datetime_like_columns={list(profile.datetime_like_columns[:8])}\n"
        f"head_5={head}\n"
    )


def _build_system_prompt(dataset_path: str, df: pd.DataFrame, profile: DatasetProfile) -> str:
    return f"""You are a Python data analyst and visualization engineer.

You have access to a CSV dataset at path: {dataset_path}

Operating rules:
- Read the CSV only from that exact path using pandas.
- Use only pandas, numpy, matplotlib, and Python standard library modules.
- Never call the network and never write files.
- If the request needs cleaning, handle it directly in code and mention the assumption briefly.
- Prefer readable, production-quality code with clear labels, titles, and tight layout for plots.
- If the user asks for a chart or plot, you must create at least one matplotlib chart.
- Always call plt.tight_layout() and plt.show() for any figure you create.
- Return a concise answer that explains what you are doing or what you found.
- Then include exactly one Python code block in triple backticks.
- The code block must be executable as-is in the sandbox.

Dataset context:
{_dataset_brief(df, profile)}
"""


class OllamaAPIError(Exception):
    def __init__(self, message: str, http_status: Optional[int] = None) -> None:
        super().__init__(message)
        self.http_status = http_status


def _extract_ollama_error_message(raw: str) -> str:
    text = raw.strip()
    if not text:
        return "Ollama Cloud request failed."
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text

    if isinstance(payload, dict):
        if isinstance(payload.get("error"), str):
            return payload["error"]
        if isinstance(payload.get("message"), str):
            return payload["message"]
        error_payload = payload.get("error")
        if isinstance(error_payload, dict) and isinstance(error_payload.get("message"), str):
            return error_payload["message"]

    return text


def _ollama_chat(
    *,
    api_key: str,
    model: ModelOption,
    messages: list[dict[str, str]],
) -> str:
    last_model_error: Optional[OllamaAPIError] = None

    for candidate in model.api_candidates:
        request_payload = json.dumps(
            {
                "model": candidate,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8")
        request = urllib_request.Request(
            "https://ollama.com/api/chat",
            data=request_payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib_request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            raw_error = exc.read().decode("utf-8", errors="replace")
            message = _extract_ollama_error_message(raw_error)
            lowered = message.lower()
            if exc.code in {400, 404} and ("not found" in lowered or "model" in lowered):
                last_model_error = OllamaAPIError(message, http_status=exc.code)
                continue
            raise OllamaAPIError(message, http_status=exc.code) from exc
        except urllib_error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise OllamaAPIError(f"Could not reach Ollama Cloud: {reason}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaAPIError("Ollama Cloud returned an unreadable response.") from exc

        content = str(payload.get("message", {}).get("content", "") or "").strip()
        if content:
            return content
        raise OllamaAPIError("Ollama Cloud returned an empty response.")

    if last_model_error is not None:
        raise last_model_error
    raise OllamaAPIError("Ollama Cloud request failed.")


def _run_code(
    sandbox: Sandbox,
    code: str,
) -> tuple[Optional[list[Any]], Optional[str], Optional[str], Optional[str]]:
    with st.spinner("Executing Python in the secure E2B sandbox..."):
        execution = sandbox.run_code(code)

    stdout = getattr(getattr(execution, "logs", None), "stdout", None)
    stderr = getattr(getattr(execution, "logs", None), "stderr", None)
    error = getattr(execution, "error", None)
    results = getattr(execution, "results", None)
    return results, stdout, stderr, str(error) if error else None


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_safe_value(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _figure_to_png_b64(figure: Any) -> Optional[str]:
    try:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        return None


def _coerce_chart_axis(values: list[Any], scale_type: str) -> list[Any]:
    if scale_type == "datetime":
        try:
            return pd.to_datetime(values).tolist()
        except Exception:
            return values
    return values


def _render_chart_payload(chart_payload: dict[str, Any]) -> bool:
    chart_type = str(chart_payload.get("type") or "unknown")
    title = str(chart_payload.get("title") or "")

    if chart_type == "superchart":
        rendered = False
        for child in chart_payload.get("elements") or []:
            if isinstance(child, dict):
                rendered = _render_chart_payload(child) or rendered
        return rendered

    if chart_type in {"line", "scatter"}:
        rows: list[dict[str, Any]] = []
        for series in chart_payload.get("elements") or []:
            label = str((series or {}).get("label") or "Series")
            points = (series or {}).get("points") or []
            x_values = _coerce_chart_axis(
                [point[0] for point in points if isinstance(point, (list, tuple)) and len(point) == 2],
                str(chart_payload.get("x_scale") or ""),
            )
            y_values = [point[1] for point in points if isinstance(point, (list, tuple)) and len(point) == 2]
            for x_value, y_value in zip(x_values, y_values):
                rows.append({"series": label, "x": x_value, "y": y_value})

        if not rows:
            return False

        chart_frame = pd.DataFrame(rows)
        x_title = str(chart_payload.get("x_label") or "x")
        y_title = str(chart_payload.get("y_label") or "y")
        mark = alt.Chart(chart_frame).mark_line(point=True) if chart_type == "line" else alt.Chart(chart_frame).mark_circle(size=90)
        interactive_chart = (
            mark.encode(
                x=alt.X("x", title=x_title),
                y=alt.Y("y:Q", title=y_title),
                color=alt.Color("series:N", title="Series"),
                tooltip=["series:N", "x", "y:Q"],
            )
            .properties(title=title or None)
            .interactive()
        )
        st.altair_chart(interactive_chart, use_container_width=True)
        return True

    if chart_type == "bar":
        rows = [
            {
                "label": item.get("label"),
                "group": item.get("group"),
                "value": item.get("value"),
            }
            for item in (chart_payload.get("elements") or [])
            if isinstance(item, dict)
        ]
        if not rows:
            return False

        chart_frame = pd.DataFrame(rows)
        chart_frame["value"] = pd.to_numeric(chart_frame["value"], errors="coerce")
        chart_frame = chart_frame.dropna(subset=["value"])
        if chart_frame.empty:
            return False

        encodings: dict[str, Any] = {
            "x": alt.X("label:N", title=str(chart_payload.get("x_label") or "Category")),
            "y": alt.Y("value:Q", title=str(chart_payload.get("y_label") or "Value")),
            "tooltip": ["label:N", "group:N", "value:Q"],
        }
        if chart_frame["group"].fillna("").nunique() > 1:
            encodings["color"] = alt.Color("group:N", title="Group")
            encodings["xOffset"] = alt.XOffset("group:N")

        interactive_chart = (
            alt.Chart(chart_frame)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(**encodings)
            .properties(title=title or None)
            .interactive()
        )
        st.altair_chart(interactive_chart, use_container_width=True)
        return True

    if chart_type == "pie":
        rows = [
            {
                "label": item.get("label"),
                "angle": item.get("angle"),
                "radius": item.get("radius"),
            }
            for item in (chart_payload.get("elements") or [])
            if isinstance(item, dict)
        ]
        if not rows:
            return False

        chart_frame = pd.DataFrame(rows)
        chart_frame["angle"] = pd.to_numeric(chart_frame["angle"], errors="coerce")
        chart_frame = chart_frame.dropna(subset=["angle"])
        if chart_frame.empty:
            return False

        interactive_chart = (
            alt.Chart(chart_frame)
            .mark_arc()
            .encode(
                theta=alt.Theta("angle:Q"),
                color=alt.Color("label:N", title="Category"),
                tooltip=["label:N", "angle:Q"],
            )
            .properties(title=title or None)
            .interactive()
        )
        st.altair_chart(interactive_chart, use_container_width=True)
        return True

    if chart_type == "box_and_whisker":
        rows = []
        for item in chart_payload.get("elements") or []:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "label": item.get("label"),
                    "min": item.get("min"),
                    "q1": item.get("first_quartile"),
                    "median": item.get("median"),
                    "q3": item.get("third_quartile"),
                    "max": item.get("max"),
                }
            )
        if not rows:
            return False
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        return True

    return False


def _serialize_results(results: Iterable[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for result in results:
        chart_payload = getattr(result, "chart", None)
        if chart_payload is not None:
            serialized.append({"kind": "chart", "chart": _json_safe_value(chart_payload)})
            continue

        html = getattr(result, "html", None)
        if html:
            serialized.append({"kind": "html", "html": str(html)})
            continue

        png_b64 = getattr(result, "png", None)
        if png_b64:
            serialized.append({"kind": "image", "png": str(png_b64)})
            continue

        figure = getattr(result, "figure", None)
        if figure is not None:
            figure_png = _figure_to_png_b64(figure)
            if figure_png:
                serialized.append({"kind": "image", "png": figure_png})
            else:
                serialized.append({"kind": "text", "value": str(result)})
            continue

        if isinstance(result, pd.Series):
            serialized.append(
                {
                    "kind": "table",
                    "data": result.to_frame(name=result.name or "value").to_json(
                        orient="split", date_format="iso"
                    ),
                }
            )
            continue

        if isinstance(result, pd.DataFrame):
            serialized.append({"kind": "table", "data": result.to_json(orient="split", date_format="iso")})
            continue

        result_json = getattr(result, "json", None)
        if result_json is not None:
            serialized.append({"kind": "json", "value": _json_safe_value(result_json)})
            continue

        result_data = getattr(result, "data", None)
        if result_data is not None:
            serialized.append({"kind": "json", "value": _json_safe_value(result_data)})
            continue

        result_markdown = getattr(result, "markdown", None)
        if result_markdown:
            serialized.append({"kind": "markdown", "value": str(result_markdown)})
            continue

        serialized.append({"kind": "text", "value": str(result)})

    return serialized


def _render_serialized_results(results: Iterable[dict[str, Any]]) -> None:
    for result in results:
        kind = str(result.get("kind") or "")
        if kind == "chart" and isinstance(result.get("chart"), dict):
            if _render_chart_payload(result["chart"]):
                continue

        if kind == "html" and result.get("html"):
            components.html(str(result["html"]), height=620, scrolling=True)
            continue

        if kind == "image" and result.get("png"):
            try:
                image = Image.open(io.BytesIO(base64.b64decode(str(result["png"]))))
                st.image(image, caption="Generated visualization", use_container_width=True)
                continue
            except Exception:
                st.write("Generated visualization")
                continue

        if kind == "table" and result.get("data"):
            try:
                frame = pd.read_json(io.StringIO(str(result["data"])), orient="split")
                st.dataframe(frame, use_container_width=True)
                continue
            except Exception:
                st.code(str(result["data"]))
                continue

        if kind == "json":
            st.json(result.get("value"))
            continue

        if kind == "markdown":
            st.markdown(str(result.get("value") or ""))
            continue

        st.write(result.get("value", ""))


def _render_results(results: Iterable[Any]) -> list[dict[str, Any]]:
    serialized_results = _serialize_results(results)
    _render_serialized_results(serialized_results)
    return serialized_results


def _quality_score(profile: DatasetProfile) -> int:
    missing_penalty = (1 - profile.completeness_ratio) * 50
    duplicate_ratio = profile.duplicate_rows / max(profile.rows, 1)
    duplicate_penalty = min(duplicate_ratio, 0.2) * 30
    type_bonus = 8 if profile.numeric_columns and profile.categorical_columns else 0
    score = round(88 - missing_penalty - duplicate_penalty + type_bonus)
    return max(38, min(98, score))


def _quality_label(score: int) -> tuple[str, str]:
    if score >= 84:
        return "Strong", "good"
    if score >= 68:
        return "Usable", "warn"
    return "Needs cleanup", "risk"


def _suggested_prompts(profile: DatasetProfile) -> list[str]:
    prompts: list[str] = []
    numeric = list(profile.numeric_columns)
    categorical = list(profile.categorical_columns)
    datetime_cols = list(profile.datetime_like_columns)

    if numeric and categorical:
        prompts.append(
            f"Compare the average {numeric[0]} across {categorical[0]} and explain the biggest differences."
        )
    if len(numeric) >= 2:
        prompts.append(
            f"Create a scatter plot of {numeric[0]} versus {numeric[1]} and comment on correlation or clusters."
        )
    if numeric:
        prompts.append(
            f"Show the distribution of {numeric[0]} and call out any outliers or unusual spread in the data."
        )
    if datetime_cols and numeric:
        prompts.append(
            f"Plot {numeric[0]} over {datetime_cols[0]} and summarize the main trend, peaks, and drops."
        )
    if categorical:
        prompts.append(
            f"Rank the top categories in {categorical[0]} and visualize where the biggest concentration appears."
        )

    fallback_prompts = [
        "Audit this dataset for missing values, duplicates, and suspicious columns, then show a chart if useful.",
        "Pick the most decision-ready visualization for this dataset and explain why it is the right choice.",
    ]

    seen: set[str] = set()
    ordered_prompts: list[str] = []
    for prompt in prompts + fallback_prompts:
        if prompt not in seen:
            ordered_prompts.append(prompt)
            seen.add(prompt)
    return ordered_prompts[:4]


def _format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_section_header(title: str, subtitle: str, kicker: str) -> None:
    st.caption(kicker.upper())
    st.subheader(title)
    st.write(subtitle)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg-1: #fbf7f2;
                --bg-2: #f6efe5;
                --card: rgba(255, 255, 255, 0.78);
                --border: rgba(16, 32, 51, 0.10);
                --ink: #102033;
                --muted: #546274;
                --navy: #0f172a;
                --orange: #f97316;
                --teal: #0f766e;
                --shadow: 0 20px 45px rgba(15, 23, 42, 0.10);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 30%),
                    radial-gradient(circle at top right, rgba(249, 115, 22, 0.16), transparent 28%),
                    linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
                color: var(--ink);
            }

            .block-container {
                padding-top: 1.8rem;
                padding-bottom: 3rem;
                max-width: 1280px;
            }

            h1, h2, h3 {
                font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
                color: var(--navy);
                letter-spacing: -0.02em;
            }

            p, li, label, span, div {
                font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.98) 0%, rgba(20, 42, 59, 0.98) 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }

            [data-testid="stSidebar"] * {
                color: #f8fafc;
            }

            [data-testid="stSidebar"] input,
            [data-testid="stSidebar"] textarea,
            [data-testid="stSidebar"] [data-baseweb="select"] > div {
                background: rgba(255, 255, 255, 0.92);
                color: #111827 !important;
                -webkit-text-fill-color: #111827 !important;
                caret-color: #111827;
            }

            [data-testid="stSidebar"] input::placeholder,
            [data-testid="stSidebar"] textarea::placeholder {
                color: #6b7280 !important;
            }

            [data-testid="stTabs"] button[data-baseweb="tab"] {
                border-radius: 999px;
                padding: 0.55rem 1.05rem;
                font-weight: 700;
            }

            [data-testid="stFileUploader"] section {
                background: rgba(255, 255, 255, 0.74);
                border: 1.5px dashed rgba(15, 23, 42, 0.18);
                border-radius: 24px;
                padding: 1rem;
            }

            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 0.8rem;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            }

            div[data-testid="stChatMessage"] {
                background: rgba(255, 255, 255, 0.68);
                border: 1px solid rgba(16, 32, 51, 0.08);
                border-radius: 22px;
                padding: 0.75rem 1rem;
                box-shadow: 0 14px 32px rgba(15, 23, 42, 0.06);
            }

            [data-testid="stChatInput"] {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(15, 23, 42, 0.10);
                border-radius: 0 0 24px 24px;
                border-top: 0;
                padding: 0.35rem 0.35rem 0.15rem 0.35rem;
                box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
                position: sticky;
                bottom: 1rem;
                z-index: 30;
            }

            [data-testid="stChatInput"] textarea {
                color: #0f172a !important;
            }

            [data-testid="stChatInput"] textarea::placeholder {
                color: #64748b !important;
            }

            [data-testid="stPopover"] button,
            button[data-testid="stBaseButton-secondary"] {
                border-radius: 999px;
            }

            div[data-testid="stPopover"] > div > button {
                background: rgba(255, 255, 255, 0.84);
                border: 1px solid rgba(15, 23, 42, 0.10);
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
                font-weight: 600;
            }

            div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPopover"]) {
                position: sticky;
                bottom: 4.95rem;
                z-index: 24;
                background: rgba(251, 247, 242, 0.94);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 24px 24px 0 0;
                border-bottom: 0;
                padding: 0.75rem 0.85rem;
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
                backdrop-filter: blur(10px);
            }

            button[kind="secondary"],
            button[kind="primary"] {
                border-radius: 999px;
            }

            [data-testid="stSidebar"] .stButton > button {
                width: 100%;
                background: linear-gradient(135deg, #f97316, #fb923c);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.18);
                box-shadow: 0 14px 28px rgba(249, 115, 22, 0.24);
            }

            [data-testid="stSidebar"] .stButton > button:hover {
                background: linear-gradient(135deg, #f97316, #fb923c);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.18);
                box-shadow: 0 16px 30px rgba(249, 115, 22, 0.26);
            }

            [data-testid="stSidebar"] .stButton > button:focus,
            [data-testid="stSidebar"] .stButton > button:active {
                background: linear-gradient(135deg, #f97316, #fb923c);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.18);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    st.session_state.setdefault("ollama_api_key", _get_secret("OLLAMA_API_KEY"))
    st.session_state.setdefault("e2b_api_key", _get_secret("E2B_API_KEY"))
    st.session_state.setdefault("model_id", MODEL_CATALOG[0].id)
    st.session_state.setdefault("chat", _default_chat_history())
    st.session_state.setdefault("analysis_runs", [])
    st.session_state.setdefault("dataset_name", None)
    st.session_state.setdefault("dataset_token", None)
    st.session_state.setdefault("dataset_bytes", None)
    st.session_state.setdefault("pending_prompt", "")
    st.session_state.setdefault("_workspace_state_loaded", False)
    if not st.session_state._workspace_state_loaded:
        _load_persisted_workspace_state()
        st.session_state._workspace_state_loaded = True
    if not st.session_state.chat:
        st.session_state.chat = _default_chat_history()


def _reset_conversation() -> None:
    st.session_state.ollama_api_key = ""
    st.session_state.e2b_api_key = ""
    st.session_state.chat = _default_chat_history()
    st.session_state.analysis_runs = []
    st.session_state.pending_prompt = ""
    _persist_workspace_state()


def _clear_active_dataset() -> None:
    st.session_state.dataset_name = None
    st.session_state.dataset_token = None
    st.session_state.dataset_bytes = None
    _reset_conversation()


def _set_active_dataset(filename: str, file_bytes: bytes) -> None:
    token = _file_token(filename, file_bytes)
    dataset_changed = st.session_state.dataset_token != token
    st.session_state.dataset_name = filename
    st.session_state.dataset_token = token
    st.session_state.dataset_bytes = file_bytes
    if dataset_changed:
        st.session_state.chat = _default_chat_history()
        st.session_state.analysis_runs = []
        st.session_state.pending_prompt = ""
    _persist_workspace_state()


def _current_model() -> ModelOption:
    selected_id = str(st.session_state.get("model_id") or MODEL_CATALOG[0].id)
    return MODEL_BY_ID.get(selected_id, MODEL_CATALOG[0])


def _current_secrets() -> Secrets:
    return Secrets(
        ollama_api_key=str(st.session_state.get("ollama_api_key") or ""),
        e2b_api_key=str(st.session_state.get("e2b_api_key") or ""),
    )


def _sidebar() -> tuple[Secrets, ModelOption]:
    with st.sidebar:
        st.markdown("## Control Room")
        st.caption("Wire in your keys, choose the model profile, and steer the analysis run.")

        st.session_state.ollama_api_key = st.text_input(
            "Ollama Cloud API key",
            value=st.session_state.ollama_api_key,
            type="password",
            help="Saved locally for this app until you use Reset conversation.",
        )
        st.caption("Get a key: https://ollama.com/settings/keys")

        st.session_state.e2b_api_key = st.text_input(
            "E2B API key",
            value=st.session_state.e2b_api_key,
            type="password",
            help="Saved locally for this app until you use Reset conversation.",
        )
        st.caption("Get a key: https://e2b.dev/")

        selected_id = st.selectbox(
            "Inference model",
            options=[model.id for model in MODEL_CATALOG],
            format_func=lambda model_id: MODEL_BY_ID[model_id].label,
            index=[model.id for model in MODEL_CATALOG].index(st.session_state.model_id),
        )
        st.session_state.model_id = selected_id
        selected_model = MODEL_BY_ID[selected_id]

        st.markdown("### Model profile")
        st.markdown(f"**{selected_model.label}**")
        st.caption(selected_model.summary)
        st.caption(f"Best for: {selected_model.best_for}")

        st.divider()
        if st.session_state.dataset_name:
            st.markdown("### Active dataset")
            st.caption(st.session_state.dataset_name)

        st.markdown("### Workflow tips")
        st.write("- Ask for one outcome at a time.")
        st.write("- Mention exact columns when you can.")
        st.write("- Start with a cleanup audit for messy files.")

        if st.button("Reset conversation", use_container_width=True):
            _reset_conversation()
            st.rerun()

    _persist_workspace_state()
    secrets = Secrets(
        ollama_api_key=str(st.session_state.ollama_api_key or ""),
        e2b_api_key=str(st.session_state.e2b_api_key or ""),
    )
    return secrets, selected_model


def _upload_dataset(sandbox: Sandbox, filename: str, file_bytes: bytes) -> str:
    dataset_path = _dataset_runtime_path(filename)
    sandbox.files.write(dataset_path, file_bytes)
    return dataset_path


def _render_ollama_exception(exc: OllamaAPIError) -> str:
    message = str(exc)
    http_status = getattr(exc, "http_status", None)
    lowered = message.lower()

    if http_status == 402 or "credit_limit" in lowered:
        user_message = (
            "Your Ollama Cloud account has run out of credits. Add credits or review your account billing, "
            "wait a few minutes for the balance to refresh, and try again. Your CSV upload is fine."
        )
        st.error("Ollama Cloud credit limit reached.")
        st.info(user_message)
    elif http_status == 401:
        user_message = (
            "Your Ollama Cloud API key was rejected. Check the key in the sidebar and try again."
        )
        st.error("Ollama Cloud authentication failed.")
        st.info(user_message)
    elif http_status == 429:
        user_message = (
            "Ollama Cloud rate limits were hit. Wait a moment and retry, or switch to a lighter model."
        )
        st.error("Ollama Cloud rate limit reached.")
        st.info(user_message)
    elif "could not reach ollama cloud" in lowered:
        user_message = (
            "The app could not reach Ollama Cloud. Check your connection and try again."
        )
        st.error("Ollama Cloud connection failed.")
        st.info(user_message)
    elif http_status in {400, 404} and ("not found" in lowered or "model" in lowered):
        user_message = (
            "The selected Ollama Cloud model could not be reached with the current API mapping. "
            "Switch models and retry."
        )
        st.error("Ollama Cloud model lookup failed.")
        st.info(user_message)
    else:
        user_message = (
            "Ollama Cloud returned an error before analysis could run. Please retry or check your model and account settings."
        )
        st.error("Ollama Cloud request failed.")
        st.info(user_message)

    with st.expander("Technical details", expanded=False):
        st.code(message)

    return user_message


def _render_dataset_stats(profile: DatasetProfile) -> None:
    quality_score = _quality_score(profile)
    quality_label, _ = _quality_label(quality_score)
    metric_left, metric_mid_left, metric_mid_right, metric_right = st.columns(4, gap="large")
    with metric_left:
        st.metric("Rows", f"{profile.rows:,}")
    with metric_mid_left:
        st.metric("Columns", f"{profile.columns:,}")
    with metric_mid_right:
        st.metric("Completeness", _format_percentage(profile.completeness_ratio))
    with metric_right:
        st.metric("Memory", f"{profile.memory_usage_mb:.2f} MB")
    st.caption(f"Dataset health: {quality_label} ({quality_score}/100)")


def _render_prompt_gallery(profile: DatasetProfile) -> None:
    st.caption("STARTER PROMPT")
    st.write("Pick a question and send it to the chat.")
    prompt_options = _suggested_prompts(profile)
    selected_prompt = st.selectbox(
        "Starter prompt",
        options=prompt_options,
        key="workspace_starter_prompt_select",
    )
    if st.button("Use selected prompt", use_container_width=True, key="workspace_starter_prompt_button"):
        st.session_state.pending_prompt = selected_prompt
        _persist_workspace_state()


def _render_workspace_overview(profile: DatasetProfile, model: ModelOption) -> None:
    quality_score = _quality_score(profile)
    quality_label, _ = _quality_label(quality_score)
    st.write(f"**Model:** {model.label}")
    st.caption(model.summary)
    st.write(f"**Best for:** {model.best_for}")
    metric_left, metric_right = st.columns(2)
    with metric_left:
        st.metric("Numeric columns", str(len(profile.numeric_columns)))
        st.metric("Duplicates", f"{profile.duplicate_rows:,}")
    with metric_right:
        st.metric("Categorical columns", str(len(profile.categorical_columns)))
        st.metric("Dataset health", f"{quality_score}/100")
    st.caption(f"Dataset quality: {quality_label}")
    if profile.top_missing_columns:
        top_missing = ", ".join(
            f"{column} ({count})" for column, count in profile.top_missing_columns[:3]
        )
        st.write(f"**Columns to watch:** {top_missing}")


def _render_dataset_quick_view(uploaded_name: str, profile: DatasetProfile) -> None:
    st.write(f"**File:** {uploaded_name}")
    st.caption("A quick summary of the current dataset.")

    metric_left, metric_right = st.columns(2)
    with metric_left:
        st.metric("Rows", f"{profile.rows:,}")
        st.metric("Numeric", str(len(profile.numeric_columns)))
    with metric_right:
        st.metric("Columns", f"{profile.columns:,}")
        st.metric("Categorical", str(len(profile.categorical_columns)))

    st.write(f"**Completeness:** {_format_percentage(profile.completeness_ratio)}")
    if profile.top_missing_columns:
        st.write(
            "**Highest missingness:** "
            + ", ".join(f"{column} ({count})" for column, count in profile.top_missing_columns[:3])
        )

    if profile.numeric_columns:
        st.write("**Numeric columns:** " + ", ".join(profile.numeric_columns[:6]))
    if profile.categorical_columns:
        st.write("**Categorical columns:** " + ", ".join(profile.categorical_columns[:6]))


def _render_dataset_lab(file_bytes: bytes, df: pd.DataFrame, profile: DatasetProfile) -> None:
    _render_section_header(
        "Dataset overview",
        "A clear look at the file you uploaded, from quality checks to column details.",
        "Dataset Lab",
    )
    _render_dataset_stats(profile)
    top_left, top_right = st.columns([1.0, 1.0], gap="large")
    with top_left:
        st.markdown("#### Missing values")
        top_missing = pd.DataFrame(profile.top_missing_columns, columns=["column", "missing"])
        if not top_missing.empty:
            st.bar_chart(top_missing.set_index("column"), use_container_width=True)
        else:
            st.success("No major missing-value issues stand out in the leading columns.")
    with top_right:
        st.markdown("#### Quick summary")
        metric_left, metric_right = st.columns(2)
        with metric_left:
            st.metric("Numeric", str(len(profile.numeric_columns)))
            st.metric("Datetime-like", str(len(profile.datetime_like_columns)))
        with metric_right:
            st.metric("Categorical", str(len(profile.categorical_columns)))
            st.metric("Duplicates", f"{profile.duplicate_rows:,}")
        if profile.top_missing_columns:
            st.write(
                "**Highest missingness:** "
                + ", ".join(f"{column} ({count})" for column, count in profile.top_missing_columns[:4])
            )

    st.markdown("#### Data preview")
    st.dataframe(df.head(40), use_container_width=True, height=420)

    schema_left, schema_right = st.columns([0.95, 1.05], gap="large")
    with schema_left:
        st.markdown("#### Column groups")
        if profile.numeric_columns:
            st.caption("Numeric columns")
            st.code(", ".join(profile.numeric_columns[:10]), language="text")
        if profile.categorical_columns:
            st.caption("Categorical columns")
            st.code(", ".join(profile.categorical_columns[:10]), language="text")
        if profile.datetime_like_columns:
            st.caption("Datetime-like columns")
            st.code(", ".join(profile.datetime_like_columns[:10]), language="text")
    with schema_right:
        st.markdown("#### Column metadata")
        st.dataframe(_column_metadata(file_bytes), use_container_width=True, height=360)

    numeric_preview = df.select_dtypes(include="number")
    if not numeric_preview.empty:
        with st.expander("Numeric summary", expanded=False):
            st.dataframe(numeric_preview.describe().T, use_container_width=True)


def _render_project_details() -> None:
    _render_section_header(
        "How the app works",
        "A short overview of what this app does and how each step fits together.",
        "Project Details",
    )
    st.markdown("#### What you can do here")
    st.write(
        "The app keeps file upload, analysis, chart generation, and result review in one place so the workflow stays easy to follow."
    )
    st.write("- Upload a CSV and inspect it")
    st.write("- Ask questions in plain language")
    st.write("- Review charts, code, and runtime logs")

    st.markdown("#### Behind the scenes")
    st.write("1. **Upload**: the CSV is loaded and checked inside the app.")
    st.write("2. **Analysis**: the selected Ollama Cloud model writes the Python needed for the request.")
    st.write("3. **Execution**: E2B runs that code in an isolated environment.")
    st.write("4. **Results**: the app shows the charts, tables, explanation, and logs in the workspace.")


def _render_information_page() -> None:
    st.caption("INFORMATION")
    st.title(APP_TITLE)
    st.write(APP_SUBTITLE)
    st.write(
        "This app helps you upload a CSV, ask questions about it, and review the charts and code that come back."
    )

    overview_col, workflow_col = st.columns([1.1, 0.9], gap="large")
    with overview_col:
        st.markdown("#### What you can do")
        st.write("- Upload a CSV in the AI Workspace page")
        st.write("- Check the file structure and data quality")
        st.write("- Ask for trends, comparisons, charts, or quick audits")
        st.write("- Review the generated Python, outputs, and logs")

    with workflow_col:
        st.markdown("#### Where to start")
        st.write("1. **Information**: start here for a quick overview.")
        st.write("2. **AI Workspace**: upload your file and run prompts.")
        st.write("3. **Dataset Lab**: inspect the uploaded dataset in more detail.")
        st.write("4. **Project Details**: see how the app is set up.")

    detail_col, model_col = st.columns([1.0, 1.0], gap="large")
    with detail_col:
        st.markdown("#### What it includes")
        st.write("- CSV profiling and column checks")
        st.write("- Prompt-to-Python analysis")
        st.write("- E2B sandbox execution")
        st.write("- Restored workspace state after refresh")

    with model_col:
        st.markdown("#### Available models")
        for model in MODEL_CATALOG:
            st.write(f"- **{model.label}**: {model.best_for}")


def _render_analysis_history() -> None:
    with st.chat_message("assistant"):
        st.markdown(INTRO_MESSAGE)

    for run in st.session_state.analysis_runs:
        with st.chat_message("user"):
            st.markdown(str(run.get("user_text") or ""))

        with st.chat_message("assistant"):
            assistant_text = str(run.get("assistant_text") or "")
            if assistant_text:
                st.markdown(assistant_text)

            python_code = str(run.get("python_code") or "")
            if python_code:
                with st.expander("Python code", expanded=False):
                    st.code(python_code, language="python")

            sandbox_error = str(run.get("sandbox_error") or "")
            if sandbox_error:
                st.error("Sandbox execution failed. Refine the prompt or mention the target columns explicitly.")
                st.code(sandbox_error)

            stdout = str(run.get("stdout") or "")
            if stdout:
                with st.expander("stdout", expanded=False):
                    st.text(stdout)

            stderr = str(run.get("stderr") or "")
            if stderr:
                with st.expander("stderr", expanded=False):
                    st.text(stderr)

            serialized_results = run.get("results") or []
            if isinstance(serialized_results, list) and serialized_results:
                _render_serialized_results(serialized_results)

            runtime_seconds = run.get("runtime_seconds")
            if isinstance(runtime_seconds, (int, float)):
                st.caption(f"Current project run time: {runtime_seconds:.2f} seconds")


def _run_assistant_query(
    *,
    secrets: Secrets,
    model: ModelOption,
    uploaded_name: str,
    file_bytes: bytes,
    df: pd.DataFrame,
    profile: DatasetProfile,
    user_text: str,
) -> None:
    run_record: dict[str, Any] = {
        "user_text": user_text,
        "assistant_text": "",
        "python_code": "",
        "results": [],
        "stdout": "",
        "stderr": "",
        "sandbox_error": "",
        "runtime_seconds": None,
    }

    st.session_state.chat.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    if not secrets.ollama_api_key or not secrets.e2b_api_key:
        message = "Missing API keys. Add Ollama Cloud and E2B keys in the sidebar, then try again."
        run_record["assistant_text"] = message
        st.session_state.chat.append({"role": "assistant", "content": message})
        with st.chat_message("assistant"):
            st.markdown(message)
        st.session_state.analysis_runs.append(run_record)
        _persist_workspace_state()
        return

    assistant_text = ""
    display_text = ""
    run_started_at = time.perf_counter()
    with st.chat_message("assistant"):
        try:
            dataset_path = _dataset_runtime_path(uploaded_name)

            with st.spinner("Putting together the analysis and Python..."):
                system_prompt = _build_system_prompt(dataset_path, df, profile)
                messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
                messages.extend(st.session_state.chat[-10:])
                if messages and messages[-1]["role"] == "user":
                    messages[-1] = {
                        "role": "user",
                        "content": _augment_user_text_for_model(user_text),
                    }
                assistant_text = _ollama_chat(
                    api_key=secrets.ollama_api_key,
                    model=model,
                    messages=messages,
                )

            python_code = _pick_python_code(assistant_text)
            display_text = _strip_code_blocks(assistant_text)
            if python_code and not display_text:
                display_text = "Analysis complete. Open `Python code` to review the script."
            elif not display_text:
                display_text = assistant_text.strip()

            if display_text:
                st.markdown(display_text)
            run_record["assistant_text"] = display_text

            if not python_code:
                st.info("The model response did not include runnable Python. Try a more explicit request.")
                st.session_state.chat.append({"role": "assistant", "content": display_text})
                st.session_state.analysis_runs.append(run_record)
                _persist_workspace_state()
                return

            python_code = _ensure_chart_display(python_code, user_text)
            run_record["python_code"] = python_code
            with st.expander("Python code", expanded=False):
                st.code(python_code, language="python")

            with Sandbox(api_key=secrets.e2b_api_key) as sandbox:
                _upload_dataset(sandbox, uploaded_name, file_bytes)
                results, stdout, stderr, error = _run_code(sandbox, python_code)

            if error:
                run_record["sandbox_error"] = error
                st.error("Sandbox execution failed. Refine the prompt or mention the target columns explicitly.")
                st.code(error)
            if stdout:
                run_record["stdout"] = stdout
                with st.expander("stdout", expanded=False):
                    st.text(stdout)
            if stderr:
                run_record["stderr"] = stderr
                with st.expander("stderr", expanded=False):
                    st.text(stderr)
            if results:
                run_record["results"] = _render_results(results)
                if _request_needs_chart(user_text) and not any(
                    item.get("kind") in {"chart", "image", "html"} for item in run_record["results"]
                ):
                    st.warning("The analysis ran, but no chart was returned. Try naming the columns or chart type you want.")
            elif _request_needs_chart(user_text):
                st.warning("The analysis finished, but it did not return a chart. Try naming the columns or chart type you want.")
            runtime_seconds = time.perf_counter() - run_started_at
            run_record["runtime_seconds"] = runtime_seconds
            st.caption(f"Current project run time: {runtime_seconds:.2f} seconds")
        except OllamaAPIError as exc:
            assistant_text = _render_ollama_exception(exc)
            run_record["assistant_text"] = assistant_text
            runtime_seconds = time.perf_counter() - run_started_at
            run_record["runtime_seconds"] = runtime_seconds
            st.caption(f"Current project run time: {runtime_seconds:.2f} seconds")
        except Exception as exc:
            assistant_text = (
                "I hit an issue while generating or running the analysis. "
                "Please retry with a more specific prompt or confirm the uploaded CSV is valid."
            )
            run_record["assistant_text"] = assistant_text
            st.error(assistant_text)
            st.code(str(exc))
            runtime_seconds = time.perf_counter() - run_started_at
            run_record["runtime_seconds"] = runtime_seconds
            st.caption(f"Current project run time: {runtime_seconds:.2f} seconds")

    st.session_state.chat.append({"role": "assistant", "content": run_record["assistant_text"] or assistant_text})
    st.session_state.analysis_runs.append(run_record)
    _persist_workspace_state()


def _render_workspace(
    *,
    secrets: Secrets,
    model: ModelOption,
    uploaded_name: str,
    file_bytes: bytes,
    df: pd.DataFrame,
    profile: DatasetProfile,
) -> None:
    _render_section_header(
        "Ask about your data",
        "Type a question and the app will generate code, run it, and show the result here.",
        "AI Workspace",
    )

    if not secrets.ollama_api_key or not secrets.e2b_api_key:
        st.warning("Add both API keys in the sidebar to run analysis.")

    with st.container(border=True):
        _render_analysis_history()

        pending_prompt = str(st.session_state.pop("pending_prompt", ""))
        if pending_prompt:
            _persist_workspace_state()

    workspace_tool_col, dataset_tool_col, starter_tool_col = st.columns(3, gap="small")
    with workspace_tool_col:
        with st.popover("Workspace", use_container_width=True):
            _render_workspace_overview(profile, model)
    with dataset_tool_col:
        with st.popover("Dataset", use_container_width=True):
            _render_dataset_quick_view(uploaded_name, profile)
    with starter_tool_col:
        with st.popover("Starter Prompt", use_container_width=True):
            _render_prompt_gallery(profile)

    user_text = st.chat_input(
        "Ask about trends, comparisons, anomalies, segmentation, or the best visualization for this data..."
    )
    if not user_text and pending_prompt:
        user_text = pending_prompt
    if not user_text:
        return

    _run_assistant_query(
        secrets=secrets,
        model=model,
        uploaded_name=uploaded_name,
        file_bytes=file_bytes,
        df=df,
        profile=profile,
        user_text=user_text,
    )


def _build_active_dataset(filename: str, file_bytes: bytes) -> Optional[ActiveDataset]:
    try:
        df = _load_csv(file_bytes)
        profile = _profile_dataset(file_bytes)
    except Exception as exc:
        st.error("This CSV could not be parsed cleanly. Please verify the file encoding and delimiter.")
        st.code(str(exc))
        return None

    return ActiveDataset(name=filename, file_bytes=file_bytes, df=df, profile=profile)


def _restore_active_dataset() -> Optional[ActiveDataset]:
    dataset_name = st.session_state.get("dataset_name")
    dataset_bytes = st.session_state.get("dataset_bytes")
    if not dataset_name or not isinstance(dataset_bytes, (bytes, bytearray)):
        return None

    active_dataset = _build_active_dataset(str(dataset_name), bytes(dataset_bytes))
    if active_dataset is None:
        st.warning("The saved dataset could not be restored cleanly. Upload the CSV again.")
        _clear_active_dataset()
        return None

    return active_dataset


def _render_workspace_input() -> Optional[ActiveDataset]:
    _render_section_header(
        "Load a dataset",
        "Upload a CSV here. Once it is loaded, the same file is available across the other pages.",
        "Input",
    )
    uploaded_file = st.file_uploader(
        "Upload a CSV",
        type=["csv"],
        help="The file is profiled in-app and uploaded to E2B only for sandbox execution.",
    )

    if uploaded_file is not None:
        active_dataset = _build_active_dataset(uploaded_file.name, uploaded_file.getvalue())
        if active_dataset is not None:
            _set_active_dataset(active_dataset.name, active_dataset.file_bytes)
        return active_dataset

    active_dataset = _restore_active_dataset()
    if active_dataset is None:
        return None
    st.caption(f"Using saved dataset: {active_dataset.name}")
    return active_dataset


def _workspace_page() -> None:
    active_dataset = _render_workspace_input()
    if active_dataset is None:
        st.info(
            "Upload a CSV to get started. Your prompts, code, charts, and results will show up here."
        )
        return

    _render_workspace(
        secrets=_current_secrets(),
        model=_current_model(),
        uploaded_name=active_dataset.name,
        file_bytes=active_dataset.file_bytes,
        df=active_dataset.df,
        profile=active_dataset.profile,
    )


def _dataset_lab_page() -> None:
    active_dataset = _restore_active_dataset()
    if active_dataset is None:
        st.info(
            "Upload a CSV from the AI Workspace page first. This page will then show the structure, preview, and quality checks."
        )
        return

    st.caption(f"Current dataset: {active_dataset.name}")
    _render_dataset_lab(active_dataset.file_bytes, active_dataset.df, active_dataset.profile)


def _project_details_page() -> None:
    active_dataset = _restore_active_dataset()
    if active_dataset is not None:
        st.caption(f"Current dataset: {active_dataset.name}")
    _render_project_details()


def _is_running_with_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _init_state()
    navigation = st.navigation(
        [
            st.Page(_render_information_page, title="Information", default=True),
            st.Page(_workspace_page, title="AI Workspace", url_path="ai-workspace"),
            st.Page(_dataset_lab_page, title="Dataset Lab", url_path="dataset-lab"),
            st.Page(_project_details_page, title="Project Details", url_path="project-details"),
        ],
        position="top",
    )
    _inject_styles()

    _sidebar()
    navigation.run()


if __name__ == "__main__":
    if not _is_running_with_streamlit():
        print(
            "This is a Streamlit app. Run it with:\n"
            "  streamlit run ai_data_visualisation_agent.py",
            file=sys.stderr,
        )
        raise SystemExit(1)
    main()
