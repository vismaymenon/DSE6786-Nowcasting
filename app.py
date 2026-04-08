from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import numpy as np
from datetime import date
from pipeline.fetch_functions import fetch_nowcast_data, fetch_confidence_intervals, fetch_historical_data, fetch_rmse, fetch_dm

QUARTERS = ["2026:Q1", "2025:Q4"]

# Display name -> database name mapping
MODEL_DB_NAMES = {
    "Ensemble":       "All_Model_Average",
    "RF Lags Avg":    "RF_Lags_Average",
    "RF Lags UMIDAS": "RF_Lags_UMIDAS",
    "LASSO UMIDAS":   "LASSO_UMIDAS",
}

MODELS = list(MODEL_DB_NAMES.keys())

MODEL_COLORS = {
    "Ensemble": "#1f77b4",
    "RF Lags Avg": "#2ca02c",
    "RF Lags UMIDAS": "#d62728",
    "LASSO UMIDAS": "#ff7f0e",
}
def to_db_names(display_names: list[str]) -> list[str]:
    """Translate a list of display names to DB names for fetch functions."""
    return [MODEL_DB_NAMES[m] for m in display_names if m in MODEL_DB_NAMES]

def from_db_name(db_name: str) -> str:
    """Translate a single DB name back to its display name."""
    return {v: k for k, v in MODEL_DB_NAMES.items()}.get(db_name, db_name)
 

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
        flash_month = int(input.flash_month() or "1")
        t = THEME["dark"] if is_dark.get() else THEME["light"]
        db_models = to_db_names(selected_models)

        matrix = fetch_dm(db_models,flash_month)
        metrics = fetch_rmse(db_models)

        # Build DM matrix table
        cell_style = (
            f"border: 1px solid {t['border']}; padding: 1rem 1.5rem; "
            "text-align: center; vertical-align: middle; min-width: 90px;"
        )
        diag_style = cell_style + f" background-color: {t['bg_card_header']};"
        header_style = (
            f"border: 1px solid {t['border']}; padding: 0.5rem 1rem; "
            f"text-align: center; font-weight: normal; color: {t['text_primary']};")

        header_row_cells = [ui.tags.th("", style=header_style)]
        for m in db_models:
            header_row_cells.append(ui.tags.th(m, style=header_style))

        data_rows = []
        for m1 in db_models:
            row_cells = [ui.tags.th(m1, style=header_style)]
            for m2 in db_models:
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
        data, x_labels = fetch_nowcast_data(quarter)

        fig = go.Figure()

        for model in selected_models:
            db_name = MODEL_DB_NAMES.get(model)
            if db_name is not None and db_name in data:
                fig.add_trace(
                    go.Scatter(
                        x=x_labels,
                        y=data[db_name],
                        mode="lines+markers",
                        name=model,
                        line=dict(color=MODEL_COLORS.get(model, "#888"), width=2),
                    )
                )

        # Shaded confidence interval
        if ci_model and ci_model != "None" and ci_model in selected_models:
            db_ci_model = MODEL_DB_NAMES.get(ci_model)
            x_ci, _ci50_lo, _ci50_hi, _ci80_lo, _ci80_hi = fetch_confidence_intervals(quarter, db_ci_model)
            ci_color = MODEL_COLORS.get(ci_model, "#888")
            r, g, b = int(ci_color[1:3], 16), int(ci_color[3:5], 16), int(ci_color[5:7], 16)
            fig.add_trace(
                go.Scatter(
                    x=x_ci + x_ci[::-1],
                    y=_ci80_hi + _ci80_lo[::-1],
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
        quarters, actual, predictions = fetch_historical_data(
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
        metrics = fetch_rmse(db_models)

        rmse_lines = []
        for model in selected_models:
            db_name = MODEL_DB_NAMES.get(model)
            if db_name not in metrics:
                continue
            m = metrics[db_name]
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

