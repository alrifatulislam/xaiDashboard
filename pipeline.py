# pipeline.py
"""
End-to-end pipeline for:
- loading a sample (XSum / SAMSum)
- running a summarization model
- computing Attention / SHAP / LIME explanations
- computing ROUGE-L, faithfulness, and consistency metrics

This can be called:
- from app.py (for Streamlit UI)
- directly via `python pipeline.py` for debugging.
"""

from typing import Dict, List, Tuple
import time
from itertools import combinations

import numpy as np
import pandas as pd

from models import SummarizationModel, load_sample
from explainers import (
    attention_token_importance,
    merge_bpe_to_words,
    shap_importance,
    ig_importance,
    lime_importance,
)
from metrics import (
    summary_rougeL,
    faithfulness_via_deletion,
    compute_spearman,
)


class PipelineResult:
    """
    Simple container for all results.
    """

    def __init__(
        self,
        source_text: str,
        reference_summary: str,
        generated_summary: str,
        rougeL: float,
        explanations: Dict[str, Tuple[List[str], np.ndarray]],
        latencies: Dict[str, float],
        faithfulness: Dict[str, float],
        consistency: pd.DataFrame,
        llm_judgments: Dict[str, Dict] | None = None,
    ):
        self.source_text = source_text
        self.reference_summary = reference_summary
        self.generated_summary = generated_summary
        self.rougeL = rougeL
        self.explanations = explanations          # method -> (words, scores)
        self.latencies = latencies                # method -> seconds
        self.faithfulness = faithfulness          # method -> rho
        self.consistency = consistency            # DataFrame
        self.llm_judgments = llm_judgments or {}  # method -> judgments


def run_pipeline(
    dataset_name: str,
    sample_index: int,
    model_name: str,
    max_new_tokens: int = 60,
    use_attention: bool = True,
    use_shap: bool = True,
    use_ig: bool = True,
    use_lime: bool = True,
    faith_max_tokens: int = 20,
    use_llm_judge: bool = False, 
) -> PipelineResult:
    """
    Run the complete pipeline for a single (dataset, sample, model) config.

    Returns:
        PipelineResult object with texts, explanations, and metrics.
    """
    # ---------- 1. Load data & model ----------
    sample = load_sample(dataset_name, split="validation", index=sample_index)
    source_text = sample["source"]
    reference_summary = sample["reference_summary"]

    model = SummarizationModel(model_name)
    generated_summary, info = model.summarize(
        source_text,
        max_new_tokens=max_new_tokens,
    )

    rougeL = summary_rougeL(reference_summary, generated_summary)

    explanations: Dict[str, Tuple[List[str], np.ndarray]] = {}
    latencies: Dict[str, float] = {}
    faithfulness: Dict[str, float] = {}

    # ========== 2. Compute explanations ========== #
    # ATTENTION
    if use_attention:
        t0 = time.perf_counter()
        att_tokens, att_scores = attention_token_importance(info, agg="mean_last")
        latencies["Attention"] = time.perf_counter() - t0

        att_words, att_word_scores = merge_bpe_to_words(att_tokens, att_scores)
        explanations["Attention"] = (att_words, att_word_scores)

    # SHAP
    if use_shap:
        t0 = time.perf_counter()
        shap_words, shap_scores = shap_importance(
            model,
            source_text,
        )
        latencies["SHAP"] = time.perf_counter() - t0
        explanations["SHAP"] = (shap_words, shap_scores)
    
    # INTEGRATED GRADIENTS (IG)
    if use_ig:
        t0 = time.perf_counter()
        ig_words, ig_scores = ig_importance(
            model,
            source_text,
        )
        latencies["IG"] = time.perf_counter() - t0
        explanations["IG"] = (ig_words, ig_scores)

    # LIME
    if use_lime:
        t0 = time.perf_counter()
        lime_words, lime_scores = lime_importance(
            model,
            source_text,
        )
        latencies["LIME"] = time.perf_counter() - t0
        explanations["LIME"] = (lime_words, lime_scores)

    methods = list(explanations.keys())

    # ========== 3. Faithfulness (per method) ========== #
    for m in methods:
        words, scores = explanations[m]
        if len(words) == 0 or len(scores) == 0:
            faithfulness[m] = float("nan")
            continue

        rho = faithfulness_via_deletion(
            text=source_text,
            words=words,
            importances=scores,
            model=model,
            reference_summary=reference_summary,
            max_tokens=faith_max_tokens,
        )
        faithfulness[m] = rho

    # ========== 4. Cross-method consistency ========== #
    rows = []
    if len(methods) >= 2:
        for m1, m2 in combinations(methods, 2):
            w1, s1 = explanations[m1]
            w2, s2 = explanations[m2]

            n = min(len(s1), len(s2))
            if n == 0:
                rho = float("nan")
            else:
                rho = compute_spearman(np.array(s1[:n]), np.array(s2[:n]))

            rows.append(
                {
                    "Pair": f"{m1} vs {m2}",
                    "Spearman ρ": rho,
                }
            )
    consistency_df = pd.DataFrame(rows)

    # ========== 5. LLM-as-judge evaluation (optional) ========== #
    llm_judgments: Dict[str, Dict] = {}
    if use_llm_judge and explanations:
        try:
            from llm_eval import llm_judge_explanations

            llm_judgments = llm_judge_explanations(
                source_text=source_text,
                reference_summary=reference_summary,
                generated_summary=generated_summary,
                explanations=explanations,
                top_k=15,
            )
        except Exception as e:
            llm_judgments = {
                "__error__": {
                    "faithfulness": None,
                    "interpretability": None,
                    "comment": f"LLM judge failed: {e}",
                }
            }


    return PipelineResult(
        source_text=source_text,
        reference_summary=reference_summary,
        generated_summary=generated_summary,
        rougeL=rougeL,
        explanations=explanations,
        latencies=latencies,
        faithfulness=faithfulness,
        consistency=consistency_df,
        llm_judgments=llm_judgments,
    )


# ---------- Debugging entry point ---------- #
'''
if __name__ == "__main__":
    # Example debug run
    config = {
        "dataset_name": "SAMSum",
        "sample_index": 0,
        "model_name": "BART-SAMSum",  
        "max_new_tokens": 60,
        "use_attention": True,
        "use_shap": False,  # turn on/off to test speed
        "use_ig": True,  
        "use_lime": False,
        "faith_max_tokens": 20,
    }

    result = run_pipeline(**config)

    print("=== DEBUG PIPELINE RUN ===")
    print("Dataset: ", config["dataset_name"])
    print("Model:   ", config["model_name"])
    print("Sample:  ", config["sample_index"])
    print()
    print("ROUGE-L:", result.rougeL)
    print()
    print("Methods:", list(result.explanations.keys()))
    print("Latencies:", result.latencies)
    print("Faithfulness:", result.faithfulness)
    print()
    print("Consistency table:")
    print(result.consistency)
'''