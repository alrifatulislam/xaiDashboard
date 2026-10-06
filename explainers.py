# explainers.py
from typing import Dict, List, Tuple
import numpy as np
from difflib import SequenceMatcher

import torch
import shap
from lime.lime_text import LimeTextExplainer

from models import SummarizationModel


# ---------- ATTENTION EXPLAINER ---------- #

def attention_token_importance(model_info: Dict, agg: str = "mean_last") -> Tuple[List[str], np.ndarray]:
    """
    Compute token-level importance from encoder self-attention.

    Args:
        model_info: dict returned by SummarizationModel.summarize
        agg: how to aggregate across layers & heads:
             - "mean_last": mean over heads from last layer only
             - "mean_all": mean over all layers & heads

    Returns:
        tokens: list of input tokens (subword / BPE)
        scores: np.ndarray importance per token (normalized to [0, 1])
    """
    tokens = model_info["input_tokens"]
    encoder_attns = model_info["encoder_attentions"]  # tuple(L, B, H, S, S)

    # Convert to numpy: shape (L, H, S, S) for batch size 1
    attn = np.stack([a[0].cpu().numpy() for a in encoder_attns])  # (L, H, S, S)

    if agg == "mean_last":
        last = attn[-1]  # (H, S, S)
        token_scores = last.mean(axis=0).sum(axis=-1)  # avg over heads, sum over sources
    else:  # "mean_all"
        token_scores = attn.mean(axis=1).mean(axis=-1).sum(axis=-1)

    # Normalize to [0, 1]
    token_scores = token_scores - token_scores.min()
    if token_scores.max() > 0:
        token_scores = token_scores / token_scores.max()

    return tokens, token_scores


def merge_bpe_to_words(tokens: List[str], scores: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """
    Merge BPE/subword tokens into word-level scores.

    Heuristic:
      - 'Ġ' (BART) or '▁' (T5) indicates a new word.
      - Aggregate scores of sub-tokens within each word by mean.

    Returns:
        words: list of word strings
        word_scores: np.ndarray of word-level scores, normalized to [0, 1].
    """
    words: List[str] = []
    word_scores: List[float] = []

    current_word = ""
    current_scores: List[float] = []

    def flush():
        nonlocal current_word, current_scores
        if current_word:
            words.append(current_word)
            word_scores.append(float(np.mean(current_scores)))
        current_word = ""
        current_scores = []

    for tok, score in zip(tokens, scores):
        # detect new word marker
        is_new = tok.startswith("Ġ") or tok.startswith("▁")
        clean = tok.replace("Ġ", "").replace("▁", "")

        if is_new:
            # commit previous
            flush()
            current_word = clean
            current_scores = [score]
        else:
            # continuation of current word
            if current_word == "":
                current_word = clean
                current_scores = [score]
            else:
                current_word += clean
                current_scores.append(score)

    flush()

    if len(word_scores) == 0:
        return [], np.array([])

    ws = np.array(word_scores)
    ws = ws - ws.min()
    if ws.max() > 0:
        ws = ws / ws.max()
    return words, ws

def _summary_similarity(base_summary: str, new_summary: str) -> float:
    """
    Cheap similarity between two summaries.
    Used as the target function for SHAP/LIME.
    """
    return SequenceMatcher(None, base_summary, new_summary).ratio()



# ---------- COMMON SCORING FUNCTION FOR SHAP / LIME ---------- #

def shap_importance(
    model: SummarizationModel,
    text: str,
    max_words: int = 60,
    nsamples: int = 200,
) -> Tuple[List[str], np.ndarray]:
    """
    Compute word-level attributions using SHAP KernelExplainer.

    Features: words (whitespace-separated).
    Model function: similarity(base_summary, summary(masked_text))

    Returns:
        words: list of words (possibly truncated)
        scores: normalized importance per word (abs(SHAP value)).
    """
    words = text.split()
    if len(words) == 0:
        return [], np.array([])

    if len(words) > max_words:
        words = words[:max_words]

    base_text = " ".join(words)
    base_summary, _ = model.summarize(base_text)

    def predict(mask_matrix: np.ndarray) -> np.ndarray:
        """
        mask_matrix: (batch, num_words) in [0,1]; >0.5 means keep.
        Returns: (batch, 1) similarity scores.
        """
        outputs = []
        for mask in mask_matrix:
            mask = np.array(mask) > 0.5
            # ensure at least one word remains
            if not mask.any():
                mask[0] = True
            masked_text = " ".join(np.array(words)[mask])
            summary_i, _ = model.summarize(masked_text)
            s = _summary_similarity(base_summary, summary_i)
            outputs.append([s])
        return np.array(outputs, dtype=float)

    background = np.ones((1, len(words)), dtype=float)  # all words present
    explainer = shap.KernelExplainer(predict, background)
    shap_values = explainer.shap_values(
        np.ones((1, len(words)), dtype=float),
        nsamples=nsamples,
    )

    # KernelExplainer returns shape (1, num_words) for single output
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values
    shap_vals = np.array(shap_vals).flatten()

    scores = np.abs(shap_vals)
    if scores.max() > 0:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return words, scores


# ---------- INTEGRATED GRADIENTS (IG) EXPLAINER ---------- #

def ig_importance(
    model: SummarizationModel,
    text: str,
    steps: int = 20,
    max_tokens: int = 128,
) -> Tuple[List[str], np.ndarray]:
    """
    Integrated Gradients over input embeddings for a seq2seq summarization model.

    Scalar function f(x):
      - negative loss of the model predicting its own generated summary
        under teacher forcing: f = -loss(inputs_embeds, labels=generated_summary_ids)

    Returns:
        words: list of word strings
        scores: word-level importance scores in [0,1]
    """
    hf_model = model.model
    tokenizer = model.tokenizer
    device = next(hf_model.parameters()).device

    # Tokenize source
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    ).to(device)

    input_ids = enc["input_ids"]      # (1, S)
    attention_mask = enc["attention_mask"]

    # Generate summary to use as target for loss
    with torch.no_grad():
        gen_out = hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=False,
        )
        gen_ids = gen_out.sequences  # (1, T) tensor

    # Use generated summary as labels (teacher forcing)
    labels = gen_ids.clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100  # ignore pad in loss

    # Get input embeddings
    emb_layer = hf_model.get_input_embeddings()
    emb = emb_layer(input_ids)  # (1, S, D)
    baseline = torch.zeros_like(emb)

    delta = emb - baseline
    integrated_grads = torch.zeros_like(emb)

    # We need gradients, so make sure grad is enabled
    for k in range(1, steps + 1):
        alpha = float(k) / steps
        # point along the path
        emb_alpha = (baseline + alpha * delta).detach()
        emb_alpha.requires_grad_(True)

        hf_model.zero_grad(set_to_none=True)

        outputs = hf_model(
            inputs_embeds=emb_alpha,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        scalar = -loss  # higher scalar = better prediction

        # Compute grad of scalar w.r.t. emb_alpha explicitly
        grad = torch.autograd.grad(
            scalar,
            emb_alpha,
            retain_graph=False,
            create_graph=False,
        )[0]

        integrated_grads += grad.detach()

    avg_grads = integrated_grads / steps
    ig = delta * avg_grads  # (1, S, D)

    # Token importance = L1 norm across embedding dim
    token_scores = ig.abs().sum(dim=-1).squeeze(0)  # (S,)
    token_scores = token_scores.detach().cpu().numpy()

    # Normalize to [0,1]
    token_scores = token_scores - token_scores.min()
    if token_scores.max() > 0:
        token_scores = token_scores / token_scores.max()

    # Convert tokens to words
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    words, word_scores = merge_bpe_to_words(tokens, token_scores)
    return words, word_scores


# ---------- LIME EXPLAINER ---------- #

def lime_importance(
    model: SummarizationModel,
    text: str,
    num_samples: int = 80,
    max_words: int = 60,
) -> Tuple[List[str], np.ndarray]:
    """
    Compute word-level attributions using LimeTextExplainer.

    We set up a binary classification: "low_similarity" vs "high_similarity"
    relative to a base summary generated from the full text.

    Returns:
        words: list of words (truncated)
        scores: normalized importance per word.
    """
    words = text.split()
    if len(words) == 0:
        return [], np.array([])

    if len(words) > max_words:
        words = words[:max_words]
    truncated_text = " ".join(words)

    base_summary, _ = model.summarize(truncated_text)

    def predict_proba(texts: List[str]) -> np.ndarray:
        """
        LIME expects a probability distribution over classes.
        We use [1 - sim, sim] where sim ∈ [0,1].
        """
        probs = []
        for t in texts:
            summary_i, _ = model.summarize(t)
            s = _summary_similarity(base_summary, summary_i)
            s = float(max(0.0, min(1.0, s)))
            probs.append([1.0 - s, s])
        return np.array(probs, dtype=float)

    explainer = LimeTextExplainer(class_names=["low_sim", "high_sim"])
    exp = explainer.explain_instance(
        truncated_text,
        predict_proba,
        num_features=len(words),
        num_samples=num_samples,
    )

    # Map word -> weight (LIME output is (word, weight))
    weights_dict = dict(exp.as_list())

    scores = np.array([weights_dict.get(w, 0.0) for w in words], dtype=float)
    # Take absolute value and normalize
    scores = np.abs(scores)
    if scores.max() > 0:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return words, scores
