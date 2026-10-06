# llm_eval.py
"""
LLM-as-judge evaluation for explanations.

Given:
  - source text
  - reference summary
  - generated summary
  - explanations: method -> (words, scores)

This module asks an LLM to rate each explanation method on:
  - faithfulness (0–5): does it seem aligned with model behavior?
  - interpretability (0–5): how understandable is it to a human?
  - comment: short qualitative insight.

"""

from typing import Dict, List, Tuple, Any
import json
import numpy as np
import requests

OLLAMA_MODEL_NAME = "atla/selene-mini:latest"
OLLAMA_API_URL = "http://localhost:11434/api/chat"


def _format_explanation(words: List[str], scores: np.ndarray, top_k: int = 15) -> str:
    """Return a readable string of top-k words with scores."""
    if len(words) == 0 or len(scores) == 0:
        return "No tokens."
    scores = np.asarray(scores)
    idx = np.argsort(scores)[::-1][:top_k]
    items = [f"{words[i]} ({scores[i]:.2f})" for i in idx]
    return ", ".join(items)


def build_llm_judge_prompt(
    source_text: str,
    reference_summary: str,
    generated_summary: str,
    explanations: Dict[str, Tuple[List[str], np.ndarray]],
    top_k: int = 15,
) -> str:
    """Build a single prompt string for the LLM judge."""
    lines = []
    lines.append("You are an expert evaluator of explanation methods for text summarization models.")
    lines.append("Your job is to read the source document, the reference summary, the model summary,")
    lines.append("and the top important words for each explanation method, then rate each method.")
    lines.append("")
    lines.append("For each explanation method, you must provide:")
    lines.append("- faithfulness: integer 0–5 (0 = not faithful, 5 = highly faithful)")
    lines.append("- interpretability: integer 0–5 (0 = incomprehensible, 5 = very clear)")
    lines.append("- comment: a short 1–3 sentence explanation of your judgment.")
    lines.append("")
    lines.append("Respond ONLY with a valid JSON object where each key is one of the explanation method names below.")
    lines.append("For each method, the value must be an object with fields: faithfulness, interpretability, comment.")
    lines.append("Example format (keys here are placeholders using YOUR method names):")

    # Build an example object dynamically
    example_obj = {}
    for method_name in explanations.keys():
        example_obj[method_name] = {
            "faithfulness": 0,
            "interpretability": 0,
            "comment": "..."
        }

    example_json = json.dumps(example_obj, indent=2)
    lines.append(example_json)
    lines.append("Include ONLY the methods that are provided; do not invent new method names.")
    lines.append("")
    lines.append("-----")
    lines.append("SOURCE DOCUMENT:")
    lines.append(source_text)
    lines.append("-----")
    lines.append("REFERENCE SUMMARY:")
    lines.append(reference_summary)
    lines.append("-----")
    lines.append("MODEL SUMMARY:")
    lines.append(generated_summary)
    lines.append("-----")
    lines.append("EXPLANATIONS (top important words):")

    for method, (words, scores) in explanations.items():
        exp_str = _format_explanation(words, scores, top_k=top_k)
        lines.append(f"- {method}: {exp_str}")

    lines.append("-----")
    lines.append("Now output the JSON object described above.")

    return "\n".join(lines)


# ----------  LLM backend here ---------- #

def call_llm(prompt: str) -> str:
    """
    Call a local Ollama chat model as the LLM judge.

    Requires:
      - Ollama running locally (ollama serve)
      - A pulled model matching OLLAMA_MODEL_NAME (e.g., `ollama pull llama3`)
    """
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,  # easier to handle in this dashboard
    }

    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        # Bubble up a readable error for the UI / pipeline
        raise RuntimeError(f"Ollama request failed: {e}")

    data = resp.json()

    # Newer Ollama chat API returns e.g.:
    # {
    #   "model": "...",
    #   "message": {"role": "assistant", "content": "...."},
    #   ...
    # }
    if isinstance(data, dict) and "message" in data:
        content = data["message"].get("content", "")
    else:
        # Fallback: try a generic "content" key
        content = data.get("content", "")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Ollama response had no usable content: {data}")

    return content


def llm_judge_explanations(
    source_text: str,
    reference_summary: str,
    generated_summary: str,
    explanations: Dict[str, Tuple[List[str], np.ndarray]],
    top_k: int = 15,
) -> Dict[str, Dict[str, Any]]:
    """
    High-level function: build prompt, call LLM, parse JSON.

    Returns:
        Dict[method_name] -> {
            "faithfulness": int,
            "interpretability": int,
            "comment": str
        }
    """
    if not explanations:
        return {}

    prompt = build_llm_judge_prompt(
        source_text,
        reference_summary,
        generated_summary,
        explanations,
        top_k=top_k,
    )

    raw = call_llm(prompt)

    # Try to parse JSON; if it fails, wrap the raw text as an error comment.
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("LLM response is not a JSON object.")
        return data
    except Exception as e:
        # Fall back to a single pseudo-entry indicating failure
        return {
            "__error__": {
                "faithfulness": None,
                "interpretability": None,
                "comment": f"Failed to parse LLM response as JSON: {e}. Raw response: {raw[:500]}",
            }
        }
