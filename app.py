from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import numpy as np
from datetime import date

# =============================================================================
# SUPABASE INTEGRATION — plug in real queries here when ready
# Expected Supabase table schemas are documented in each function.
# =============================================================================

def fetch_nowcast_data(quarter: str) -> dict[str, list[float]]:
    """
    Fetch nowcast model predictions for a given quarter.

    Supabase table: nowcast_predictions
      - quarter     TEXT   e.g. '2026:Q1'
      - model       TEXT   e.g. 'Combined model', 'Model 1', 'Model 2'
      - month_label TEXT   e.g. 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'
      - month_order INT    1–6 (for sorting)
      - value       FLOAT  % annual GDP growth prediction

    Returns: {model_name: [val_month1, ..., val_month6]}, [month_labels]
    """
    raise NotImplementedError("Replace with Supabase query")


def fetch_nowcast_x_labels(quarter: str) -> list[str]:
    """
    Fetch the ordered month labels for the x-axis of a given quarter.
    (Could be derived from fetch_nowcast_data, kept separate for flexibility.)
    """
    raise NotImplementedError("Replace with Supabase query")


def fetch_confidence_intervals(
    quarter: str, model: str
) -> tuple[list[str], list[float], list[float]]:
    """
    Fetch confidence interval bounds for a model/quarter.

    Supabase table: confidence_intervals
      - quarter   TEXT
      - model     TEXT
      - month_label TEXT
      - month_order INT
      - lower     FLOAT
      - upper     FLOAT

    Returns: (month_labels, lower_bounds, upper_bounds)
    """
    raise NotImplementedError("Replace with Supabase query")


def fetch_historical_data(
    start_date, end_date, flash_month: int
) -> tuple[list[str], list[float], dict[str, list[float]]]:
    """
    Fetch historical actuals and model predictions.

    flash_month: which monthly flash estimate to use (1, 2, or 3)

    Supabase tables:
      historical_actuals
        - quarter  TEXT
        - value    FLOAT  (actual % annual GDP growth)

      historical_predictions
        - quarter     TEXT
        - model       TEXT
        - flash_month INT   (1, 2, or 3 — month within the quarter)
        - value       FLOAT

    Returns: (quarter_labels, actual_values, {model: [values]})
    """
    raise NotImplementedError("Replace with Supabase query")


def fetch_evaluation_metrics(models: list[str]) -> dict[str, dict]:
    """
    Fetch model evaluation metrics.

    Supabase table: evaluation_metrics
      - model        TEXT
      - rmse         FLOAT
      - dm_statistic FLOAT

    Returns: {model_name: {'rmse': float, 'dm_statistic': float}}
    """
    raise NotImplementedError("Replace with Supabase query")


# =============================================================================
# DUMMY DATA — delete this entire block when Supabase is connected,
# and replace each get_dummy_* call in the server with the fetch_* equivalent.
# =============================================================================

QUARTERS = ["2026:Q1", "2025:Q4"]
MODELS = ["Combined model", "Model 1", "Model 2"]
MODEL_COLORS = {
    "Combined model": "#1f77b4",
    "Model 1": "#2ca02c",
    "Model 2": "#d62728",
}

_NOWCAST_X = ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03"]

_NOWCAST_Y = {
    "2026:Q1": {
        "Combined model": [None, None, None, 2.5, 2.4, 2.6],
        "Model 1":        [None, None, None, 2.3, 2.2, 2.4],
        "Model 2":        [None, None, None, 2.7, 2.6, 2.8],
    },
    "2025:Q4": {
        "Combined model": [1.9, 2.0, 2.1, 2.2, 2.3, 2.5],
        "Model 1":        [1.7, 1.8, 1.9, 2.0, 2.1, 2.3],
        "Model 2":        [2.0, 2.2, 2.3, 2.4, 2.5, 2.7],
    },
}
# Default fallback for quarters without specific dummy data
for _q in ["2025:Q3", "2025:Q2"]:
    _NOWCAST_Y[_q] = _NOWCAST_Y["2025:Q4"]


def get_dummy_nowcast_data(quarter: str):
    """Dummy nowcast data — replace with fetch_nowcast_data(quarter)."""
    return _NOWCAST_Y.get(quarter, _NOWCAST_Y["2026:Q1"]), _NOWCAST_X


def get_dummy_confidence_intervals(quarter: str, model: str):
    """Dummy CI data — replace with fetch_confidence_intervals(quarter, model)."""
    base = _NOWCAST_Y.get(quarter, _NOWCAST_Y["2026:Q1"]).get(model, [2.0] * 6)
    labels, lower, upper = [], [], []
    for x, v in zip(_NOWCAST_X, base):
        if v is not None:
            labels.append(x)
            lower.append(v - 0.35)
            upper.append(v + 0.35)
    return labels, lower, upper


_HIST_QUARTERS = [
    "2020:Q1", "2020:Q2", "2020:Q3", "2020:Q4", "2021:Q1", "2021:Q2"
]
_HIST_ACTUAL = [2.3, -5.0, -31.4, 33.4, 6.3, 6.7]
_HIST_PREDS = {
    "Combined model": [2.5, -4.8, -30.9, 32.8, 6.1, 6.5],
    "Model 1":        [2.2, -4.5, -31.0, 33.0, 6.0, 6.4],
    "Model 2":        [2.7, -5.2, -31.8, 33.8, 6.5, 6.9],
}
# Simulate slight differences per flash month
_HIST_PREDS_BY_MONTH = {
    1: _HIST_PREDS,
    2: {m: [v + 0.1 for v in vals] for m, vals in _HIST_PREDS.items()},
    3: {m: [v + 0.2 for v in vals] for m, vals in _HIST_PREDS.items()},
}

_DUMMY_METRICS = {
    "Combined model": {"rmse": 1.2, "dm_statistic": 0.8},
    "Model 1":        {"rmse": 1.5, "dm_statistic": 1.1},
    "Model 2":        {"rmse": 1.4, "dm_statistic": 0.9},
}


def get_dummy_historical_data(start_date, end_date, flash_month: int):
    """Dummy historical data — replace with fetch_historical_data(...)."""
    return _HIST_QUARTERS, _HIST_ACTUAL, _HIST_PREDS_BY_MONTH.get(flash_month, _HIST_PREDS)


def get_dummy_metrics(models: list[str]):
    """Dummy metrics — replace with fetch_evaluation_metrics(models)."""
    return {m: _DUMMY_METRICS[m] for m in models if m in _DUMMY_METRICS}


# =============================================================================
# END DUMMY DATA
# =============================================================================


# ── UI ────────────────────────────────────────────────────────────────────────

nowcast_controls = ui.div(
    ui.input_radio_buttons(
        "quarter",
        None,
        choices=QUARTERS,
        selected="2026:Q1",
        inline=True,
    ),
    ui.card(
        ui.card_header("Model Selection"),
        ui.input_checkbox_group(
            "nowcast_models",
            None,
            choices=MODELS,
            selected=["Combined model"],
        ),
    ),
    ui.card(
        ui.card_header("Confidence Interval"),
        ui.input_select(
            "ci_model",
            None,
            choices={"None": "None"},
            selected="None",
        ),
    ),
)

historical_controls = ui.div(
    ui.card(
        ui.card_header("Input date range"),
        ui.input_date_range(
            "hist_date_range",
            None,
            start="2020-01-01",
            end="2022-01-01",
        ),
    ),
    ui.card(
        ui.card_header("Display Options"),
        ui.strong("MODEL SELECTION"),
        ui.input_checkbox_group(
            "hist_models",
            None,
            choices=MODELS,
            selected=["Combined model"],
        ),
        ui.strong("FLASH ESTIMATE USED"),
        ui.input_select(
            "flash_month",
            None,
            choices={"1": "1st month", "2": "2nd month", "3": "3rd month"},
            selected="1",
        ),
    ),
    ui.card(
        ui.card_header("Evaluation Metrics"),
        ui.output_ui("eval_metrics"),
    ),
)

app_ui = ui.page_fluid(
    ui.h1("US GDP Nowcast"),
    ui.navset_tab(
        ui.nav_panel(
            "Nowcast",
            ui.layout_columns(
                ui.card(output_widget("nowcast_plot")),
                nowcast_controls,
                col_widths=[8, 4],
            ),
        ),
        ui.nav_panel(
            "Historical Data",
            ui.layout_columns(
                ui.card(output_widget("historical_plot")),
                historical_controls,
                col_widths=[8, 4],
            ),
        ),
        id="main_tabs",
        selected="Nowcast",
    ),
)


# ── Server ────────────────────────────────────────────────────────────────────

def server(input, output, session):

    # Keep the CI dropdown choices in sync with selected models
    @reactive.effect
    def _sync_ci_choices():
        selected = input.nowcast_models()
        choices = {"None": "None"}
        for m in (selected or []):
            choices[m] = m
        ui.update_select("ci_model", choices=choices, selected="None")

    # ── Nowcast plot ──────────────────────────────────────────────────────────

    @render_widget
    def nowcast_plot():
        quarter = input.quarter()
        selected_models = input.nowcast_models() or []
        ci_model = input.ci_model()

        # TODO: swap get_dummy_nowcast_data → fetch_nowcast_data when Supabase ready
        data, x_labels = get_dummy_nowcast_data(quarter)

        fig = go.Figure()

        for model in selected_models:
            if model in data:
                fig.add_trace(
                    go.Scatter(
                        x=x_labels,
                        y=data[model],
                        mode="lines+markers",
                        name=model,
                        line=dict(color=MODEL_COLORS.get(model, "#888"), width=2),
                    )
                )

        # Shaded confidence interval
        if ci_model and ci_model != "None" and ci_model in selected_models:
            # TODO: swap get_dummy_confidence_intervals → fetch_confidence_intervals
            x_ci, lower, upper = get_dummy_confidence_intervals(quarter, ci_model)
            ci_color = MODEL_COLORS.get(ci_model, "#888")
            r, g, b = int(ci_color[1:3], 16), int(ci_color[3:5], 16), int(ci_color[5:7], 16)
            fig.add_trace(
                go.Scatter(
                    x=x_ci + x_ci[::-1],
                    y=upper + lower[::-1],
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{ci_model} 95% CI",
                    showlegend=True,
                )
            )

        fig.update_layout(
            yaxis_title="% annual GDP growth",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=50, r=20, t=60, b=40),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        )
        return fig

    # ── Historical plot ───────────────────────────────────────────────────────

    @render_widget
    def historical_plot():
        date_range = input.hist_date_range()
        start_date = date_range[0] if date_range else date(2020, 1, 1)
        end_date   = date_range[1] if date_range else date(2022, 1, 1)
        selected_models = input.hist_models() or []
        flash_month = int(input.flash_month())

        # TODO: swap get_dummy_historical_data → fetch_historical_data when Supabase ready
        quarters, actual, predictions = get_dummy_historical_data(
            start_date, end_date, flash_month
        )

        fig = go.Figure()

        # Actual GDP — dotted line
        fig.add_trace(
            go.Scatter(
                x=quarters,
                y=actual,
                mode="lines+markers",
                name="Actual",
                line=dict(color="#000000", width=2, dash="dot"),
            )
        )

        # Model predictions — solid lines
        for model in selected_models:
            if model in predictions:
                fig.add_trace(
                    go.Scatter(
                        x=quarters,
                        y=predictions[model],
                        mode="lines+markers",
                        name=model,
                        line=dict(color=MODEL_COLORS.get(model, "#888"), width=2),
                    )
                )

        fig.update_layout(
            yaxis_title="% annual GDP growth",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=50, r=20, t=60, b=40),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        )
        return fig

    # ── Evaluation metrics ────────────────────────────────────────────────────

    @render.ui
    def eval_metrics():
        selected_models = input.hist_models() or []
        if not selected_models:
            return ui.p("No models selected.")

        # TODO: swap get_dummy_metrics → fetch_evaluation_metrics when Supabase ready
        metrics = get_dummy_metrics(selected_models)

        rows = []
        for model in selected_models:
            if model not in metrics:
                continue
            m = metrics[model]
            rows.append(
                ui.div(
                    ui.tags.u(ui.strong(model.upper())),
                    ui.p(f"RMSE: {m['rmse']:.1f}"),
                    ui.p(f"DM Test Statistic: {m['dm_statistic']:.1f}"),
                )
            )
        return ui.div(*rows) if rows else ui.p("No metrics available.")


app = App(app_ui, server)
