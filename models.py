# models.py
from typing import Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIGS = {
    # Generic CNN/DailyMail summarizer
    "BART-CNN": "facebook/bart-large-cnn",

    # XSum-specific models
    "BART-XSum": "facebook/bart-large-xsum",
    "DistilBART-XSum": "sshleifer/distilbart-xsum-12-6",

    # SAMSum-specific (dialogue) model
    "BART-SAMSum": "philschmid/bart-large-cnn-samsum",

    # Small baseline
    "T5-Small": "t5-small",
}

class SummarizationModel:
    """
    Thin wrapper for Hugging Face seq2seq summarization models.
    """

    def __init__(self, model_name: str):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")

        hf_id = MODEL_CONFIGS[model_name]
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            hf_id,
            output_attentions=True,
            output_hidden_states=False,
        ).to(DEVICE)

    def summarize(self, text: str, max_new_tokens: int = 60) -> Tuple[str, Dict]:
        """
        Generate a summary + raw model outputs (logits, attentions, etc.).
        Returns:
            summary_text, model_outputs_dict
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                return_dict_in_generate=True,
                output_attentions=True,
            )

        # Decode summary
        summary_ids = outputs.sequences
        summary_text = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Encoder attentions for explanations
        with torch.no_grad():
            encoder = self.model.get_encoder()
            enc_outputs = encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                output_attentions=True,
            )

        info = {
            "input_ids": inputs["input_ids"][0].cpu().tolist(),
            "input_tokens": self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0]
            ),
            "encoder_attentions": enc_outputs.attentions,  #  tuple(L, B, H, S, S)
        }
        return summary_text, info


# --- Dataset helpers --- #

from typing import Dict
from datasets import load_dataset

def load_sample(dataset_name: str, split: str = "validation", index: int = 0) -> Dict[str, str]:
    index = int(index)

    if dataset_name == "XSum":
        ds = load_dataset("EdinburghNLP/xsum", split=split)
        row = ds[index]
        return {
            "source": row["document"],
            "reference_summary": row["summary"],
        }

    elif dataset_name == "SAMSum":
        ds = load_dataset("knkarthick/samsum", split=split)
        row = ds[index]
        return {
            "source": row["dialogue"],
            "reference_summary": row["summary"],
        }

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

