"""Streamlit demo UI for transformer-circuit-visualizer."""

from __future__ import annotations

import streamlit as st

from transformer_circuit_visualizer.analysis import CircuitAnalyzer
from transformer_circuit_visualizer.config import settings
from transformer_circuit_visualizer.model_service import TransformerLensModelService
from transformer_circuit_visualizer.schemas import (
    AblateHeadRequest,
    AnalyzeRequest,
)
from transformer_circuit_visualizer.ui_helpers import (
    ablation_delta_dataframe,
    attention_heatmap_dataframe,
    attention_matrix_dataframe,
    head_summary_dataframe,
    logit_lens_dataframe,
    predictions_dataframe,
    tokens_dataframe,
)


st.set_page_config(
    page_title="Transformer Circuit Visualizer",
    layout="wide",
)


def main() -> None:
    """Render the demo workbench."""

    _apply_page_style()
    _render_header()

    with st.sidebar:
        st.header("Run")
        prompt = st.text_area(
            "Prompt",
            value="The capital of France is",
            height=120,
        )
        model_options = list(settings.available_models)
        default_model_index = (
            model_options.index(settings.default_model)
            if settings.default_model in model_options
            else 0
        )
        model_name = st.selectbox(
            "Model",
            options=model_options,
            index=default_model_index,
        )
        top_k = st.slider("Top-k predictions", min_value=1, max_value=20, value=5)
        run_button = st.button("Run analysis", type="primary", use_container_width=True)

    if run_button:
        _run_analysis(
            prompt=prompt,
            model_name=model_name,
            top_k=top_k,
        )

    if "analysis_result" not in st.session_state:
        _render_empty_state()
        st.stop()

    result = st.session_state["analysis_result"]
    service = st.session_state["model_service"]
    analyzer = st.session_state["analyzer"]
    selected_model_name = st.session_state["selected_model_name"]
    prompt = st.session_state["prompt"]
    top_k = st.session_state["top_k"]

    _render_run_summary(
        model_name=result.metadata.model_name,
        token_count=len(result.tokens),
        head_count=result.metadata.n_layers * result.metadata.n_heads,
    )

    tokens_tab, logits_tab, attention_tab, heads_tab, ablation_tab = st.tabs(
        ["Tokens", "Predictions", "Attention", "Heads", "Ablation"]
    )

    with tokens_tab:
        _render_tokens(result)

    with logits_tab:
        _render_predictions(result)

    with attention_tab:
        _render_attention(
            service=service,
            selected_model_name=selected_model_name,
            prompt=prompt,
            n_layers=result.metadata.n_layers,
            n_heads=result.metadata.n_heads,
        )

    with heads_tab:
        _render_head_summaries(result)

    with ablation_tab:
        _render_ablation(
            analyzer=analyzer,
            prompt=prompt,
            selected_model_name=selected_model_name,
            top_k=top_k,
            n_layers=result.metadata.n_layers,
            n_heads=result.metadata.n_heads,
        )


def _run_analysis(prompt: str, model_name: str, top_k: int) -> None:
    selected_model_name = model_name.strip()
    service = TransformerLensModelService(model_names=(selected_model_name,))
    analyzer = CircuitAnalyzer(service)

    try:
        result = analyzer.analyze(
            AnalyzeRequest(prompt=prompt, model_name=selected_model_name, top_k=top_k)
        )
    except Exception as exc:  # pragma: no cover - Streamlit UI surface only.
        st.error(str(exc))
        return

    st.session_state["analysis_result"] = result
    st.session_state["model_service"] = service
    st.session_state["analyzer"] = analyzer
    st.session_state["selected_model_name"] = selected_model_name
    st.session_state["prompt"] = prompt
    st.session_state["top_k"] = top_k


def _render_header() -> None:
    st.markdown(
        """
        <section class="tcv-header">
          <div>
            <p class="tcv-kicker">Mechanistic interpretability workbench</p>
            <h1>Transformer Circuit Visualizer</h1>
            <p class="tcv-deck">
              Inspect tokens, logits, attention heads, and ablations from one prompt.
            </p>
          </div>
          <div class="tcv-header-note">
            <span>Runs TransformerLens models.</span>
            <span>First load may download weights.</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="tcv-empty">
          <p class="tcv-empty-title">Ready for a prompt.</p>
          <p>
            Choose a TransformerLens-supported model, then run analysis to inspect
            cached activations and head behavior.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_run_summary(model_name: str, token_count: int, head_count: int) -> None:
    st.markdown(
        f"""
        <div class="tcv-summary-grid">
          <div class="tcv-summary-card">
            <span>Model</span>
            <strong>{model_name}</strong>
          </div>
          <div class="tcv-summary-card">
            <span>Tokens</span>
            <strong>{token_count}</strong>
          </div>
          <div class="tcv-summary-card">
            <span>Heads</span>
            <strong>{head_count}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_tokens(result: object) -> None:
    st.subheader("Tokens")
    st.dataframe(
        tokens_dataframe(result.tokens, result.token_ids),
        use_container_width=True,
        hide_index=True,
    )


def _render_predictions(result: object) -> None:
    predictions_col, summary_col = st.columns([1, 1])
    with predictions_col:
        st.subheader("Final Next-Token Predictions")
        st.dataframe(
            predictions_dataframe(result.final_token_predictions),
            use_container_width=True,
            hide_index=True,
        )

    with summary_col:
        st.subheader("Head Summary")
        st.dataframe(
            head_summary_dataframe(result.head_summaries),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Layer-by-Layer Logit Lens")
    st.dataframe(
        logit_lens_dataframe(result.logit_lens),
        use_container_width=True,
        hide_index=True,
    )


def _render_head_summaries(result: object) -> None:
    st.subheader("Head Summary")
    st.dataframe(
        head_summary_dataframe(result.head_summaries),
        use_container_width=True,
        hide_index=True,
    )


def _render_attention(
    service: TransformerLensModelService,
    selected_model_name: str,
    prompt: str,
    n_layers: int,
    n_heads: int,
) -> None:
    heatmap_layer, heatmap_head = _attention_controls(n_layers, n_heads)
    attention = service.attention_pattern(
        service.run_with_cache(selected_model_name, prompt),
        layer=heatmap_layer,
        head=heatmap_head,
    )

    heatmap_col, matrix_col = st.columns([1.2, 1])
    with heatmap_col:
        st.subheader("Attention Heatmap")
        st.vega_lite_chart(
            attention_heatmap_dataframe(attention),
            {
                "mark": "rect",
                "encoding": {
                    "x": {"field": "key_position", "type": "ordinal", "title": "Key token"},
                    "y": {"field": "query_position", "type": "ordinal", "title": "Query token"},
                    "color": {
                        "field": "attention",
                        "type": "quantitative",
                        "scale": {"scheme": "viridis"},
                    },
                    "tooltip": [
                        {"field": "query_position", "type": "ordinal"},
                        {"field": "query_token", "type": "nominal"},
                        {"field": "key_position", "type": "ordinal"},
                        {"field": "key_token", "type": "nominal"},
                        {"field": "attention", "type": "quantitative"},
                    ],
                },
                "height": 360,
            },
            use_container_width=True,
        )

    with matrix_col:
        st.subheader("Attention Matrix")
        st.dataframe(attention_matrix_dataframe(attention), use_container_width=True)


def _render_ablation(
    analyzer: CircuitAnalyzer,
    prompt: str,
    selected_model_name: str,
    top_k: int,
    n_layers: int,
    n_heads: int,
) -> None:
    st.subheader("Head Ablation")
    ablation_layer, ablation_head = _ablation_controls(n_layers, n_heads)

    try:
        ablation = analyzer.ablate_head(
            AblateHeadRequest(
                prompt=prompt,
                model_name=selected_model_name,
                layer=ablation_layer,
                head=ablation_head,
                top_k=top_k,
            )
        )
    except Exception as exc:  # pragma: no cover - Streamlit UI surface only.
        st.error(str(exc))
        st.stop()

    before_col, after_col, delta_col = st.columns(3)
    with before_col:
        st.caption("Before ablation")
        st.dataframe(predictions_dataframe(ablation.before), use_container_width=True, hide_index=True)
    with after_col:
        st.caption("After ablation")
        st.dataframe(predictions_dataframe(ablation.after), use_container_width=True, hide_index=True)
    with delta_col:
        st.caption("Deltas")
        st.dataframe(ablation_delta_dataframe(ablation.deltas), use_container_width=True, hide_index=True)


def _attention_controls(n_layers: int, n_heads: int) -> tuple[int, int]:
    layer_col, head_col = st.columns(2)
    with layer_col:
        layer = st.number_input("Attention layer", min_value=0, max_value=n_layers - 1, value=0)
    with head_col:
        head = st.number_input("Attention head", min_value=0, max_value=n_heads - 1, value=0)
    return int(layer), int(head)


def _ablation_controls(n_layers: int, n_heads: int) -> tuple[int, int]:
    layer_col, head_col = st.columns(2)
    with layer_col:
        layer = st.number_input("Ablation layer", min_value=0, max_value=n_layers - 1, value=0)
    with head_col:
        head = st.number_input("Ablation head", min_value=0, max_value=n_heads - 1, value=0)
    return int(layer), int(head)


def _apply_page_style() -> None:
    st.markdown(
        """
        <style>
        :root {
          --tcv-bg: #f5f8f4;
          --tcv-panel: #ffffff;
          --tcv-panel-soft: #eef5ee;
          --tcv-text: #17251b;
          --tcv-muted: #5c6f62;
          --tcv-accent: #166534;
          --tcv-accent-strong: #0f4d2a;
          --tcv-border: #d7e2d8;
          --tcv-shadow: 0 18px 48px rgba(31, 64, 39, 0.08);
        }
        html, body, .stApp, [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(circle at top left, rgba(22, 101, 52, 0.10), transparent 34rem),
            linear-gradient(180deg, #f9fbf7 0%, var(--tcv-bg) 100%);
          color: var(--tcv-text);
        }
        .stApp {
          font-variant-numeric: tabular-nums;
        }
        .block-container {
          max-width: 1180px;
          padding-top: 1.25rem;
          padding-bottom: 3rem;
        }
        [data-testid="stHeader"] {
          background: rgba(245, 248, 244, 0.82);
          backdrop-filter: blur(14px);
          border-bottom: 1px solid rgba(215, 226, 216, 0.7);
        }
        h1, h2, h3, p, label, span, div[data-testid="stMarkdownContainer"] {
          color: var(--tcv-text);
        }
        .tcv-header {
          display: grid;
          grid-template-columns: 1fr;
          gap: 1.5rem;
          align-items: end;
          margin: 0.5rem 0 1.25rem;
          padding: 1.35rem 0 0.5rem;
        }
        .tcv-kicker {
          margin: 0 0 0.55rem;
          color: var(--tcv-accent);
          font-size: 0.82rem;
          font-weight: 700;
          letter-spacing: 0.02em;
        }
        .tcv-header h1 {
          margin: 0;
          max-width: 880px;
          color: var(--tcv-text);
          font-size: clamp(2.25rem, 4.2vw, 3.9rem);
          line-height: 1;
          letter-spacing: 0;
          text-wrap: balance;
        }
        .tcv-deck {
          max-width: 58ch;
          margin: 0.85rem 0 0;
          color: var(--tcv-muted);
          font-size: 1rem;
          line-height: 1.55;
        }
        .tcv-header-note {
          display: grid;
          gap: 0.35rem;
          max-width: 440px;
          padding: 1rem;
          border: 1px solid var(--tcv-border);
          border-radius: 14px;
          background: rgba(255, 255, 255, 0.72);
          box-shadow: var(--tcv-shadow);
        }
        .tcv-header-note span {
          color: var(--tcv-muted);
          font-size: 0.9rem;
        }
        .tcv-empty {
          margin-top: 1.5rem;
          padding: 1.25rem;
          border: 1px solid var(--tcv-border);
          border-radius: 16px;
          background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(238, 245, 238, 0.88));
          box-shadow: var(--tcv-shadow);
        }
        .tcv-empty-title {
          margin: 0 0 0.35rem;
          color: var(--tcv-text);
          font-size: 1.05rem;
          font-weight: 700;
        }
        .tcv-empty p:last-child {
          margin: 0;
          color: var(--tcv-muted);
          line-height: 1.55;
        }
        .tcv-summary-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 1rem;
          margin: 1rem 0 1.2rem;
        }
        .tcv-summary-card {
          min-width: 0;
          padding: 1rem 1.1rem;
          border: 1px solid var(--tcv-border);
          border-radius: 14px;
          background: rgba(255, 255, 255, 0.86);
          box-shadow: var(--tcv-shadow);
        }
        .tcv-summary-card span {
          display: block;
          color: var(--tcv-muted);
          font-size: 0.84rem;
          font-weight: 600;
        }
        .tcv-summary-card strong {
          display: block;
          overflow: hidden;
          color: var(--tcv-text);
          font-size: clamp(1.35rem, 3vw, 2.3rem);
          font-weight: 700;
          line-height: 1.1;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        [data-testid="stSidebar"] {
          background:
            linear-gradient(180deg, #ffffff 0%, #eef5ee 100%);
          border-right: 1px solid var(--tcv-border);
        }
        [data-testid="stSidebar"] * {
          color: var(--tcv-text);
        }
        [data-testid="stSidebar"] h2 {
          color: var(--tcv-accent-strong);
          letter-spacing: 0;
        }
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] input {
          color: var(--tcv-text);
          background: #ffffff;
          border: 1px solid var(--tcv-border);
          border-radius: 10px;
        }
        [data-testid="stSidebar"] textarea:focus,
        [data-testid="stSidebar"] input:focus {
          border-color: var(--tcv-accent);
          box-shadow: 0 0 0 3px rgba(22, 101, 52, 0.16);
        }
        [data-testid="stSidebar"] button[kind="primary"] {
          min-height: 46px;
          border-radius: 10px;
          background: var(--tcv-accent);
          color: #ffffff !important;
          border: 1px solid var(--tcv-accent);
          font-weight: 700;
          transition: transform 160ms ease, background 160ms ease;
        }
        [data-testid="stSidebar"] button[kind="primary"] * {
          color: #ffffff !important;
        }
        [data-testid="stSidebar"] button[kind="primary"]:hover {
          background: var(--tcv-accent-strong);
          border-color: var(--tcv-accent-strong);
        }
        [data-testid="stSidebar"] button[kind="primary"]:active {
          transform: translateY(1px);
        }
        [data-testid="stAlert"] {
          border: 1px solid var(--tcv-border);
          border-radius: 14px;
          background: #ffffff;
          color: var(--tcv-text);
        }
        [data-testid="stMetric"] {
          background: rgba(255, 255, 255, 0.86);
          border: 1px solid var(--tcv-border);
          border-radius: 14px;
          padding: 0.8rem 1rem;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] [data-testid="stMetricLabel"] {
          color: var(--tcv-text);
        }
        div[data-testid="stTabs"] button {
          min-height: 44px;
          color: var(--tcv-muted);
          font-weight: 600;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
          color: var(--tcv-accent-strong);
        }
        [data-testid="stDataFrame"],
        [data-testid="stVegaLiteChart"] {
          border: 1px solid var(--tcv-border);
          border-radius: 14px;
          overflow: hidden;
          background: #ffffff;
          box-shadow: var(--tcv-shadow);
        }
        [data-testid="stNumberInput"] input {
          min-height: 44px;
          color: var(--tcv-text);
          background: #ffffff;
          border: 1px solid var(--tcv-border);
          border-radius: 10px;
        }
        [data-testid="stHorizontalBlock"] {
          align-items: stretch;
        }
        @media (max-width: 760px) {
          .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
          }
          .tcv-header,
          .tcv-summary-grid {
            grid-template-columns: 1fr;
          }
          .tcv-header h1 {
            font-size: 2.4rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
