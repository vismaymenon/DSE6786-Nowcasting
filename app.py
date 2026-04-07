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


def fetch_dm_matrix(models: list[str]) -> dict[tuple[str, str], float | None]:
    """
    Fetch pairwise DM test statistics.

    Supabase table: dm_statistics
      - model_a  TEXT
      - model_b  TEXT
      - dm_stat  FLOAT

    Returns: {(model_a, model_b): dm_stat} for all i≠j pairs; diagonal not included.
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


_DUMMY_DM_MATRIX = {
    ("Combined model", "Model 1"): 0.12,
    ("Combined model", "Model 2"): 0.24,
    ("Model 1", "Model 2"):        0.35,
}


def get_dummy_dm_matrix(models: list[str]):
    """Dummy DM matrix — replace with fetch_dm_matrix(models)."""
    result = {}
    for m1 in models:
        for m2 in models:
            if m1 == m2:
                result[(m1, m2)] = None
            else:
                key = (m1, m2) if (m1, m2) in _DUMMY_DM_MATRIX else (m2, m1)
                result[(m1, m2)] = _DUMMY_DM_MATRIX.get(key)
    return result


# =============================================================================
# END DUMMY DATA
# =============================================================================


# =============================================================================
# THEME — edit colours and fonts here
# =============================================================================
#
# Each mode has the following keys:
#   bg_page         — page / outermost background
#   bg_card         — card body background
#   bg_card_header  — card header strip background
#   text_primary    — main body text
#   text_secondary  — muted / label text
#   accent          — buttons, active tabs, highlights (secondary accent colour)
#   border          — card / input borders
#   grid            — plot gridlines
#   plot_bg         — plot area background (passed directly to Plotly)
#   plot_paper      — plot paper background (passed directly to Plotly)
#   plot_text       — axis labels / tick text colour in Plotly
#
# FONTS
#   font_body       — applied to <body>; controls all UI text
#   font_heading    — applied to h1–h3
#
# To load a Google Font, add a ui.tags.link() in app_ui and reference it here,
# e.g. font_body = "'Inter', sans-serif"

THEME = {
    "light": {
        # ── Backgrounds ──────────────────────────────────────────────────────
        "bg_page":        "#f8f9fa",   # TODO: swap for your light page colour
        "bg_card":        "#ffffff",   # TODO: swap for your light card colour
        "bg_card_header": "#f1f3f5",   # TODO: swap for your light card header
        # ── Text ─────────────────────────────────────────────────────────────
        "text_primary":   "#212529",   # TODO: swap for your light primary text
        "text_secondary": "#6c757d",   # TODO: swap for your light muted text
        # ── Accent ───────────────────────────────────────────────────────────
        "accent":         "#0d6efd",   # TODO: swap for your light accent colour
        # ── Borders & grids ──────────────────────────────────────────────────
        "border":         "#dee2e6",   # TODO: swap for your light border colour
        "grid":           "#e9ecef",   # TODO: swap for your light plot grid
        # ── Plotly surface colours ────────────────────────────────────────────
        "plot_bg":        "#ffffff",
        "plot_paper":     "#ffffff",
        "plot_text":      "#212529",
        # ── Fonts ─────────────────────────────────────────────────────────────
        # TODO: replace with your chosen font stack, e.g. "'Inter', sans-serif"
        "font_body":      "'Hanken Grotesk', sans-serif",
        "font_heading":   "'Hanken Grotesk', sans-serif",
    },
    "dark": {
        # ── Backgrounds ──────────────────────────────────────────────────────
        "bg_page":        "#1a1d21",   # TODO: swap for your dark page colour
        "bg_card":        "#2b2f35",   # TODO: swap for your dark card colour
        "bg_card_header": "#22262c",   # TODO: swap for your dark card header
        # ── Text ─────────────────────────────────────────────────────────────
        "text_primary":   "#e9ecef",   # TODO: swap for your dark primary text
        "text_secondary": "#adb5bd",   # TODO: swap for your dark muted text
        # ── Accent ───────────────────────────────────────────────────────────
        "accent":         "#4dabf7",   # TODO: swap for your dark accent colour
        # ── Borders & grids ──────────────────────────────────────────────────
        "border":         "#3d4249",   # TODO: swap for your dark border colour
        "grid":           "#3d4249",   # TODO: swap for your dark plot grid
        # ── Plotly surface colours ────────────────────────────────────────────
        "plot_bg":        "#2b2f35",
        "plot_paper":     "#2b2f35",
        "plot_text":      "#e9ecef",
        # ── Fonts ─────────────────────────────────────────────────────────────
        # TODO: replace with your chosen font stack (can differ from light mode)
        "font_body":      "'Hanken Grotesk', sans-serif",
        "font_heading":   "'Hanken Grotesk', sans-serif",
    },
}

# =============================================================================
# END THEME
# =============================================================================


# ── Onboarding wizard helpers ─────────────────────────────────────────────────


_ABOUT_NOWCASTING = "Lorem ipsum dolor sit amet"
_QUARTER_SELECTION = "Lorem ipsum dolor sit amet"
_MODEL_SELECTION = "Lorem ipsum dolor sit amet"
_CONFIDENCE_INTERVAL = "Lorem ipsum dolor sit amet"
_HISTORICAL_DATA = "Lorem ipsum dolor sit amet"
_DATE_RANGE_SELECTION = "Lorem ipsum dolor sit amet"
_FLASH_ESTIMATE = "Lorem ipsum dolor sit amet"
_EVALUATION_METRICS = "Lorem ipsum dolor sit amet"

_TOOLTIP_BASE = (
    "position: fixed; background: white; padding: 1.2rem 1.5rem; "
    "border-radius: 8px; z-index: 1001; min-width: 240px; max-width: 320px; "
    "box-shadow: 0 4px 20px rgba(0,0,0,0.4);"
)
_BTN_MARGIN = "margin-right: 8px;"


def _btn_row(step: int):
    buttons = []
    if step >= 2:
        buttons.append(
            ui.input_action_button("wizard_prev", "\u2190", style=_BTN_MARGIN)
        )
    if step == 1:
        buttons.append(
            ui.input_action_button("wizard_skip", "Skip tutorial", style=_BTN_MARGIN)
        )
        buttons.append(ui.input_action_button("wizard_next", "Show me around"))
    elif step == 6:
        pass  # no Next — user must click the Historical Data tab to advance
    elif step == 10:
        buttons.append(ui.input_action_button("wizard_finish", "Finish tutorial"))
    else:
        buttons.append(ui.input_action_button("wizard_next", "\u2192"))
    return ui.div(*buttons, style="margin-top: 1rem;")


def _close_btn():
    return ui.input_action_button(
        "wizard_close", "×",
        style=(
            "position: absolute; top: 0.75rem; right: 0.75rem; "
            "background: none; border: none; font-size: 1.25rem; "
            "color: #666; cursor: pointer; padding: 0; line-height: 1;"
        ),
    )


def _centered_modal(header: str, body: str | None, step: int):
    content = [ui.h3(header, style="margin-bottom: 1rem;")]
    if body:
        content.append(ui.p(body))
    content.append(_btn_row(step))
    return ui.div(
        _close_btn(),
        *content,
        style=(
            "position: fixed; top: 50%; left: 50%; "
            "transform: translate(-50%, -50%); "
            "background: white; padding: 2.5rem; border-radius: 10px; "
            "min-width: 360px; max-width: 540px; "
            "box-shadow: 0 0 0 9999px rgba(0,0,0,0.7), "
            "0 4px 30px rgba(0,0,0,0.4); "
            "z-index: 1001; pointer-events: auto; position: fixed;"
        ),
    )


def _spotlight(selector: str, tooltip_pos: str, description: str, step: int):
    """
    Spotlight overlay: the target element gets a massive box-shadow that
    darkens everything else. A floating tooltip sits next to it.
    """
    css = f"""
        {selector} {{
            position: relative !important;
            z-index: 1000 !important;
            box-shadow: 0 0 0 9999px rgba(0,0,0,0.7) !important;
            border-radius: 6px;
        }}
    """
    hint = ""
    if step == 6:
        hint = ui.p(
            ui.tags.em("Click the 'Historical Data' tab to continue."),
            style="margin-top: 0.5rem; font-size: 0.85rem; color: #555;",
        )
    return ui.div(
        ui.tags.style(css),
        ui.div(
            _close_btn(),
            ui.p(description, style="margin-bottom: 0.25rem;"),
            hint,
            _btn_row(step),
            style=f"{_TOOLTIP_BASE} {tooltip_pos} position: fixed;",
        ),
    )


# ── UI ────────────────────────────────────────────────────────────────────────

nowcast_controls = ui.div(
    ui.card(
        ui.card_header("Quarter Selection"),
        ui.input_radio_buttons(
            "quarter",
            None,
            choices=QUARTERS,
            selected="2026:Q1",
            inline=True,
        ),
        id="card_quarter",
    ),
    ui.card(
        ui.card_header("Model Selection"),
        ui.input_checkbox_group(
            "nowcast_models",
            None,
            choices=MODELS,
            selected=["Combined model"],
        ),
        id="card_nowcast_model",
    ),
    ui.card(
        ui.card_header("Confidence Interval"),
        ui.input_select(
            "ci_model",
            None,
            choices={"None": "None"},
            selected="None",
        ),
        id="card_ci",
    ),
)

historical_controls = ui.div(
    ui.card(
        ui.card_header("Date Range Selection"),
        ui.input_date_range(
            "hist_date_range",
            None,
            start="2020-01-01",
            end="2022-01-01",
        ),
        id="card_date_range",
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
        id="card_hist_display",
    ),
    ui.card(
        ui.card_header("Evaluation Metrics"),
        ui.output_ui("eval_metrics"),
        id="card_eval",
    ),
)

app_ui = ui.page_fluid(
    ui.tags.link(
        rel="stylesheet",
        href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:ital,wght@0,100..900;1,100..900&display=swap",
    ),
    ui.output_ui("theme_css"),
    ui.output_ui("wizard_ui"),
    ui.output_ui("dm_overlay"),
    ui.div(
        ui.h1("US GDP Nowcast", style="margin: 0;"),
        ui.div(
            ui.output_ui("dark_mode_btn"),
            ui.input_action_button("wizard_replay", "Play tutorial", style="margin-left: 1rem;"),
            style="margin-left: auto; display: flex; align-items: center;",
        ),
        style="display: flex; align-items: center; margin-bottom: 1rem;",
    ),
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
    style="padding: 2rem 3rem 0 3rem;",
)


# ── Server ────────────────────────────────────────────────────────────────────

def server(input, output, session):

    wizard_step = reactive.value(1)
    dm_overlay_visible = reactive.value(False)
    is_dark = reactive.value(False)

    # ── Theme CSS injection ───────────────────────────────────────────────────

    @render.ui
    def theme_css():
        t = THEME["dark"] if is_dark.get() else THEME["light"]
        css = f"""
            body {{
                background-color: {t['bg_page']} !important;
                color: {t['text_primary']} !important;
                font-family: {t['font_body']} !important;  /* ← body font */
            }}
            h1, h2, h3, h4, h5, h6 {{
                font-family: {t['font_heading']} !important;  /* ← heading font */
                font-weight: bold !important;
                color: {t['text_primary']} !important;
            }}
            .card {{
                background-color: {t['bg_card']} !important;
                border-color: {t['border']} !important;
                color: {t['text_primary']} !important;
            }}
            .card-header {{
                background-color: {t['bg_card_header']} !important;
                border-color: {t['border']} !important;
                color: {t['text_primary']} !important;
            }}
            .nav-tabs {{
                border-color: {t['border']} !important;
            }}
            .nav-tabs .nav-link {{
                color: {t['text_secondary']} !important;
            }}
            .nav-tabs .nav-link.active {{
                background-color: {t['bg_card']} !important;
                border-color: {t['border']} !important;
                color: {t['accent']} !important;
            }}
            label, .form-label, .shiny-input-container {{
                color: {t['text_primary']} !important;
            }}
            .form-control, .form-select {{
                background-color: {t['bg_card']} !important;
                border-color: {t['border']} !important;
                color: {t['text_primary']} !important;
            }}
            .btn-default, .btn-secondary {{
                background-color: {t['bg_card_header']} !important;
                border-color: {t['border']} !important;
                color: {t['text_primary']} !important;
            }}
            /* ── Secondary accent: active/focus highlights ── */
            .btn-primary, a {{
                color: {t['accent']} !important;
            }}
            /* ── Tab content padding ── */
            .tab-content > .tab-pane {{
                padding-top: 1.25rem;
            }}
        """
        return ui.tags.style(css)

    @render.ui
    def dark_mode_btn():
        label = "View in light mode" if is_dark.get() else "View in dark mode"
        return ui.input_action_button("toggle_dark_mode", label)

    @reactive.effect
    @reactive.event(input.toggle_dark_mode)
    def _on_toggle_dark():
        is_dark.set(not is_dark.get())

    # ── Wizard rendering ──────────────────────────────────────────────────────

    @render.ui
    def wizard_ui():
        step = wizard_step.get()
        if step == 0:
            return ui.div()
        if step == 1:
            return _centered_modal("US GDP Nowcast", None, step)
        if step == 2:
            return _centered_modal(
                "About Nowcasting", _ABOUT_NOWCASTING, step)
        if step == 3:
            return _spotlight(
                "#card_quarter",
                "right: 36%; top: 20%;",
                _QUARTER_SELECTION, step,
            )
        if step == 4:
            return _spotlight(
                "#card_nowcast_model",
                "right: 36%; top: 37%;",
                _MODEL_SELECTION, step,
            )
        if step == 5:
            return _spotlight(
                "#card_ci",
                "right: 36%; top: 56%;",
                _CONFIDENCE_INTERVAL, step,
            )
        if step == 6:
            return _spotlight(
                ".nav-tabs li:nth-child(2) .nav-link",
                "left: 35%; top: 7%;",
                _HISTORICAL_DATA, step,
            )
        if step == 7:
            return _spotlight(
                "#card_date_range",
                "right: 36%; top: 22%;",
                _DATE_RANGE_SELECTION, step,
            )
        if step == 8:
            return _spotlight(
                "#card_hist_display",
                "right: 36%; top: 38%;",
                _MODEL_SELECTION, step,
            )
        if step == 9:
            return _spotlight(
                "#card_hist_display",
                "right: 36%; top: 54%;",
                _FLASH_ESTIMATE, step,
            )
        if step == 10:
            return _spotlight(
                "#card_eval",
                "right: 36%; top: 68%;",
                _EVALUATION_METRICS, step,
            )
        return ui.div()

    # ── Wizard navigation ─────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.wizard_next)
    def _on_next():
        new = wizard_step.get() + 1
        if new == 7:
            ui.update_navs("main_tabs", selected="Historical Data")
        elif new <= 6:
            ui.update_navs("main_tabs", selected="Nowcast")
        wizard_step.set(new)

    @reactive.effect
    @reactive.event(input.wizard_prev)
    def _on_prev():
        new = wizard_step.get() - 1
        if new <= 6:
            ui.update_navs("main_tabs", selected="Nowcast")
        else:
            ui.update_navs("main_tabs", selected="Historical Data")
        wizard_step.set(new)

    @reactive.effect
    @reactive.event(input.wizard_skip)
    def _on_skip():
        wizard_step.set(0)

    @reactive.effect
    @reactive.event(input.wizard_close)
    def _on_close():
        wizard_step.set(0)

    @reactive.effect
    @reactive.event(input.wizard_finish)
    def _on_finish():
        wizard_step.set(0)

    @reactive.effect
    @reactive.event(input.wizard_replay)
    def _on_replay():
        ui.update_navs("main_tabs", selected="Nowcast")
        wizard_step.set(1)

    # Advance from step 6 when the user clicks the Historical Data tab
    @reactive.effect
    @reactive.event(input.main_tabs)
    def _tab_advance():
        if wizard_step.get() == 6 and input.main_tabs() == "Historical Data":
            wizard_step.set(7)

    # ── DM overlay show/hide ──────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.view_dm_stats)
    def _show_dm_overlay():
        dm_overlay_visible.set(True)

    @reactive.effect
    @reactive.event(input.close_dm_overlay)
    def _hide_dm_overlay():
        dm_overlay_visible.set(False)

    @render.ui
    def dm_overlay():
        if not dm_overlay_visible.get():
            return ui.div()

        selected_models = input.hist_models() or []
        t = THEME["dark"] if is_dark.get() else THEME["light"]

        # TODO: swap get_dummy_dm_matrix → fetch_dm_matrix when Supabase ready
        matrix = get_dummy_dm_matrix(selected_models)
        # TODO: swap get_dummy_metrics → fetch_evaluation_metrics when Supabase ready
        metrics = get_dummy_metrics(selected_models)

        # Build DM matrix table
        cell_style = (
            f"border: 1px solid {t['border']}; padding: 1rem 1.5rem; "
            "text-align: center; vertical-align: middle; min-width: 90px;"
        )
        diag_style = cell_style + f" background-color: {t['bg_card_header']};"
        header_style = (
            f"border: 1px solid {t['border']}; padding: 0.5rem 1rem; "
            f"text-align: center; font-weight: normal; color: {t['text_primary']};"
        )

        header_row_cells = [ui.tags.th("", style=header_style)]
        for m in selected_models:
            header_row_cells.append(ui.tags.th(m, style=header_style))

        data_rows = []
        for m1 in selected_models:
            row_cells = [ui.tags.th(m1, style=header_style)]
            for m2 in selected_models:
                val = matrix.get((m1, m2))
                if val is None:
                    row_cells.append(ui.tags.td("", style=diag_style))
                else:
                    row_cells.append(ui.tags.td(f"{val:.2f}", style=cell_style))
            data_rows.append(ui.tags.tr(*row_cells))

        dm_table = ui.tags.table(
            ui.tags.tr(*header_row_cells),
            *data_rows,
            style="border-collapse: collapse;",
        )

        # RMSE column
        rmse_lines = [ui.tags.u(ui.strong("RMSE"))]
        for model in selected_models:
            if model in metrics:
                rmse_lines.append(ui.p(f"{model}: {metrics[model]['rmse']:.1f}"))

        return ui.div(
            # Backdrop
            ui.div(style=(
                "position: fixed; inset: 0; background: rgba(0,0,0,0.6); "
                "z-index: 1100;"
            )),
            # Panel
            ui.div(
                # Header row
                ui.div(
                    ui.h3("DM Statistics", style="margin: 0;"),
                    ui.input_action_button(
                        "close_dm_overlay", "×",
                        style=(
                            "background: none; border: none; font-size: 1.5rem; "
                            f"color: {t['text_primary']}; cursor: pointer; "
                            "padding: 0; line-height: 1;"
                        ),
                    ),
                    style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;",
                ),
                ui.p(
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
                    style=f"color: {t['text_secondary']}; margin-bottom: 1.25rem;",
                ),
                # Two columns
                ui.div(
                    ui.div(dm_table, style="flex: 1; overflow-x: auto;"),
                    ui.div(
                        *rmse_lines,
                        style=(
                            f"min-width: 160px; padding-left: 2rem; "
                            f"border-left: 1px solid {t['border']}; margin-left: 1.5rem;"
                        ),
                    ),
                    style="display: flex; align-items: flex-start;",
                ),
                style=(
                    f"position: fixed; top: 50%; left: 50%; "
                    "transform: translate(-50%, -50%); "
                    f"background: {t['bg_card']}; color: {t['text_primary']}; "
                    "padding: 2rem; border-radius: 10px; "
                    "z-index: 1101; pointer-events: auto; "
                    "min-width: 480px; max-width: 85vw; max-height: 85vh; overflow-y: auto; "
                    f"box-shadow: 0 4px 30px rgba(0,0,0,0.4);"
                ),
            ),
        )

    # ── Keep CI dropdown in sync with selected models ─────────────────────────

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
        t = THEME["dark"] if is_dark.get() else THEME["light"]

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
            plot_bgcolor=t["plot_bg"],
            paper_bgcolor=t["plot_paper"],
            font=dict(color=t["plot_text"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=50, r=20, t=60, b=40),
            xaxis=dict(showgrid=True, gridcolor=t["grid"]),
            yaxis=dict(showgrid=True, gridcolor=t["grid"])
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
        t = THEME["dark"] if is_dark.get() else THEME["light"]

        # TODO: swap get_dummy_historical_data → fetch_historical_data when Supabase ready
        quarters, actual, predictions = get_dummy_historical_data(
            start_date, end_date, flash_month
        )

        fig = go.Figure()

        # Actual GDP — dotted line
        actual_line_color = t["text_primary"]
        fig.add_trace(
            go.Scatter(
                x=quarters,
                y=actual,
                mode="lines+markers",
                name="Actual",
                line=dict(color=actual_line_color, width=2, dash="dot"),
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
            plot_bgcolor=t["plot_bg"],
            paper_bgcolor=t["plot_paper"],
            font=dict(color=t["plot_text"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=50, r=20, t=60, b=40),
            xaxis=dict(showgrid=True, gridcolor=t["grid"]),
            yaxis=dict(showgrid=True, gridcolor=t["grid"])
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

        rmse_lines = []
        for model in selected_models:
            if model not in metrics:
                continue
            m = metrics[model]
            rmse_lines.append(ui.div(f"{model}: {m['rmse']:.1f}"))

        if not rmse_lines:
            return ui.p("No metrics available.")

        content = [ui.tags.u(ui.strong("RMSE")), *rmse_lines]
        if len(selected_models) > 1:
            content.append(
                ui.input_action_button(
                    "view_dm_stats", "View DM statistics",
                    style="margin-top: 0.75rem; width: 100%;",
                )
            )
        return ui.div(*content)


app = App(app_ui, server)
