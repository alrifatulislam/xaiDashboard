# app.py
import html

import streamlit as st

from pipeline import run_pipeline
from models import MODEL_CONFIGS  # just for listing model names


# ---------- Helper to color words by importance ---------- #

def render_colored_text(method_name: str, words, scores):
    """
    Render words as HTML spans with background color intensity
    proportional to their importance score (in [0,1]).

    High importance → stronger color.
    """
    if words is None or scores is None or len(words) == 0:
        st.info(f"No tokens available for {method_name}.")
        return

    spans = []
    for w, s in zip(words, scores):
        # clamp s to [0,1]
        s = max(0.0, min(1.0, float(s)))

        # simple white → red background
        # s=0 → rgb(255,255,255) (white)
        # s=1 → rgb(255, 200, 200) (light red)
        intensity = int(200 + (55 * (1 - s)))  # 200–255 range for green/blue
        bg_color = f"rgb(255,{intensity},{intensity})"

        # escape the word for safe HTML
        w_esc = html.escape(w)
        span = (
            f'<span style="background-color:{bg_color};'
            f' padding:2px 4px; margin:1px; border-radius:3px;'
            f' display:inline-block;">{w_esc}</span>'
        )
        spans.append(span)

    html_text = " ".join(spans)
    st.markdown(
        f"<div style='line-height:2.0;'>{html_text}</div>",
        unsafe_allow_html=True,
    )


# ---------- Streamlit UI ---------- #

st.set_page_config(
    page_title="Explainability Comparison Dashboard",
    layout="wide",
)

st.title("Explainability Comparison Dashboard for Generative AI Models")

st.markdown(
    """
This dashboard compares **Attention**, **SHAP**, **LIME**, and **Integrated Gradients (IG)** explanations for
summarization models, and visualizes each explanation as colored words
(high intensity = high importance).
"""
)

# ---------- Sidebar controls ---------- #

st.sidebar.header("Configuration")

dataset_name = st.sidebar.selectbox("Dataset", ["XSum", "SAMSum"])
sample_index = st.sidebar.number_input("Sample index", min_value=0, max_value=1000, value=0, step=1)

all_models = list(MODEL_CONFIGS.keys())
model_name = st.sidebar.selectbox("Summarization model", all_models)

max_new_tokens = st.sidebar.slider("Max new tokens for summary", 10, 150, 60, step=10)

st.sidebar.markdown("### Explainers to run")
use_attention = st.sidebar.checkbox("Attention", value=True)
use_shap = st.sidebar.checkbox("SHAP", value=True)
use_ig = st.sidebar.checkbox("Integrated Gradients (IG)", value=True)
use_lime = st.sidebar.checkbox("LIME", value=True)

st.sidebar.markdown("### Explanation evaluation by LLM")
use_llm_judge = st.sidebar.checkbox("LLM judge for explanations (experimental)", value=False)

faith_max_tokens = st.sidebar.slider(
    "Max tokens for faithfulness computation",
    5,
    40,
    20,
    step=5,
)

run_button = st.sidebar.button("Run explanation")


# ---------- Main logic using pipeline ---------- #

if run_button:
    with st.spinner("Running pipeline..."):
        result = run_pipeline(
            dataset_name=dataset_name,
            sample_index=int(sample_index),
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            use_attention=use_attention,
            use_shap=use_shap,
            use_ig=use_ig,
            use_lime=use_lime,
            faith_max_tokens=faith_max_tokens,
            use_llm_judge=use_llm_judge,
        )

    # ----- Texts ----- #
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Source Document / Dialogue")
        st.text_area("Source", result.source_text, height=300)

    with col2:
        st.subheader("Reference Summary")
        st.text_area("Reference", result.reference_summary, height=120)
        st.subheader("Model Summary")
        st.text_area("Generated", result.generated_summary, height=120)
        rouge = result.rougeL
        st.markdown(
            f"""
            <div style="
                margin-top: 0.5rem;
                padding: 0.6rem 0.9rem;
                border-radius: 0.5rem;
                background-color: #f0f5ff;
                border: 1px solid #4a90e2;
                display: inline-block;
                font-size: 1.05rem;
            ">
                <span style="font-weight: 600;">ROUGE-L (↑)</span>
                <span style="color: #555;">&nbsp;F1, reference vs generated:&nbsp;</span>
                <span style="font-weight: 700;">{rouge:.3f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ----- Explanations as colored words ----- #
    st.subheader("Explanations (colored words by importance)")

    cols = st.columns(len(result.explanations) or 1)

    for idx, (method, (words, scores)) in enumerate(result.explanations.items()):
        with cols[idx]:
            st.markdown(f"### {method}")
            render_colored_text(method, words, scores)

    st.markdown("---")

    # ----- Faithfulness & latency table ----- #
    st.subheader("Faithfulness via token deletion (ROUGE-L drop)")

    if result.explanations:
        rows = []
        for method in result.explanations.keys():
            rows.append(
                {
                    "Method": method,
                    "Latency (s)": result.latencies.get(method, float("nan")),
                    "Faithfulness (↑)": result.faithfulness.get(method, float("nan")),
                }
            )
        st.dataframe(
            rows,
            use_container_width=True,
        )
    else:
        st.info("No explainers were selected.")

    st.markdown("---")

    # ----- Cross-method consistency ----- #
    st.subheader("Cross-method Consistency (↑) (Spearman ρ on word importances)")

    if not result.consistency.empty:
        st.dataframe(result.consistency, use_container_width=True)
    else:
        st.info("Need at least two explanation methods to compute consistency.")
    
    
    # ----- LLM-as-judge evaluation ----- #
    if use_llm_judge:
        st.markdown("---")
        st.subheader("LLM-judged Explanation Quality")

        if "__error__" in result.llm_judgments:
            err = result.llm_judgments["__error__"]
            st.warning(f"LLM judge failed: {err.get('comment', 'Unknown error')}")
        elif not result.llm_judgments:
            st.info("No LLM judgments available.")
        else:
            rows = []
            for method in result.explanations.keys():
                jd = result.llm_judgments.get(method, {})
                rows.append(
                    {
                        "Method": method,
                        "LLM faithfulness (0–5)": jd.get("faithfulness"),
                        "LLM interpretability (0–5)": jd.get("interpretability"),
                        "LLM comment": jd.get("comment", ""),
                    }
                )

            import pandas as pd
            df_judge = pd.DataFrame(rows)[
                ["Method", "LLM faithfulness (0–5)", "LLM interpretability (0–5)"]
            ]
            st.dataframe(df_judge, use_container_width=True)

            st.markdown("**LLM comments on explanations:**")
            for row in rows:
                st.markdown(
                    f"- **{row['Method']}** – faithfulness={row['LLM faithfulness (0–5)']}, "
                    f"interpretability={row['LLM interpretability (0–5)']}  \n"
                    f"  {row['LLM comment']}"
                )
else:
    st.info("Configure options in the sidebar and click **Run explanation**.")
