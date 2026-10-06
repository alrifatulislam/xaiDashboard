# Summarization Explainability Dashboard

An interactive Streamlit app for comparing text summarization models and exploring how different methods highlight important words in a source document.

The app supports the **XSum** and **SAMSum** datasets and provides:

- Summaries from several pretrained Hugging Face models
- Word-importance visualizations from Attention, Integrated Gradients (IG), SHAP, and LIME
- ROUGE-L comparison with the dataset's reference summary
- Experimental token-deletion faithfulness and cross-method consistency metrics
- An optional local Ollama LLM judge

## Before you start

You will need:

- Python 3.10 or newer
- Internet access the first time you run the app, to download datasets and pretrained models
- Enough memory and disk space for the model you select; a GPU is optional, but can make model inference faster

SHAP and LIME can take significantly longer than Attention because they run the model repeatedly. For a first run, select just **Attention** and a small model such as **T5-Small**.

## Install

Open a terminal in the project folder (the folder containing `app.py` and `requirements.txt`).

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run this once in the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Start the dashboard

With the virtual environment activated, run:

```bash
streamlit run app.py
```

Open the local address printed in the terminal (usually <http://localhost:8501>) in your browser. To stop the app, press **Ctrl+C** in the terminal.

## Use the dashboard

1. Select **XSum** for news articles or **SAMSum** for dialogues.
2. Choose a sample index. The app loads examples from the dataset's validation split; index `0` is a good place to start.
3. Choose a summarization model. **T5-Small** is the smaller option; the BART models require more resources.
4. Set the maximum number of tokens for the generated summary.
5. Choose one or more explanation methods. For a quicker first run, leave only **Attention** selected. SHAP and LIME can be slow.
6. Optionally enable the experimental LLM judge (see below).
7. Click **Run explanation** and wait for the results.

The results include the source text, reference and generated summaries, a ROUGE-L score, highlighted words for each selected explainer, and metric tables. The highlighted words indicate relative importance within an explanation; they are not guarantees that a word caused a particular model output.

### Available models

| Dashboard name | Hugging Face model |
| --- | --- |
| BART-CNN | `facebook/bart-large-cnn` |
| BART-XSum | `facebook/bart-large-xsum` |
| DistilBART-XSum | `sshleifer/distilbart-xsum-12-6` |
| BART-SAMSum | `philschmid/bart-large-cnn-samsum` |
| T5-Small | `t5-small` |

The dataset and selected model are downloaded automatically when first needed and cached by their respective libraries.

## Understanding the metrics

- **ROUGE-L** compares the generated summary with the dataset reference summary. It is an overlap-based score, not a complete measure of summary quality.
- **Faithfulness via token deletion** measures how the ROUGE-L score changes when words are removed. Treat it as an experimental comparison, not proof of causal faithfulness.
- **Cross-method consistency** reports Spearman rank correlation between explanation scores from pairs of methods. Higher correlation means more similar rankings, not necessarily more accurate explanations.
- **Latency** is the time recorded while calculating each explanation; model and dataset loading are separate.

## Optional: use the local Ollama LLM judge

The LLM judge is off by default. To enable it:

1. Install Ollama from <https://ollama.com/> and start it.
2. In a terminal, download the configured model:

   ```bash
   ollama pull atla/selene-mini
   ```

3. Confirm that `OLLAMA_MODEL_NAME` in `llm_eval.py` matches the model you downloaded. The default is `atla/selene-mini:latest`.
4. In the dashboard, select **LLM judge for explanations (experimental)** before running an explanation.

The app sends the source document, summaries, and explanation highlights to the Ollama service at `http://localhost:11434`. The judge's scores and comments are model-generated and should be treated as subjective.

## Troubleshooting

- **`streamlit` or another package is not found:** Activate the project virtual environment and run `pip install -r requirements.txt` again.
- **Dataset or model download fails:** Check your internet connection and try again. The first download may take time, especially for larger models.
- **The app is slow or runs out of memory:** Use **T5-Small**, select only Attention, and reduce the number of explanation methods. SHAP, LIME, and faithfulness evaluation involve additional model runs.
- **The selected sample index fails:** Try index `0` or another index available in the dataset's validation split.
- **LLM judge reports a connection error:** Start Ollama and verify that it is reachable at `http://localhost:11434`; also check that the configured model has been pulled.

## Run the pipeline without the dashboard

`pipeline.py` contains the `run_pipeline` function for running the analysis from Python. Its example command-line configuration is currently commented out, so running `python pipeline.py` by itself does not launch an analysis.

## Project files

- `app.py` — Streamlit user interface
- `pipeline.py` — summarization and explanation workflow
- `models.py` — pretrained models and dataset loading
- `explainers.py` — Attention, IG, SHAP, and LIME implementations
- `metrics.py` — ROUGE-L, token-deletion, and consistency metrics
- `llm_eval.py` — optional Ollama-based judge
- `requirements.txt` — Python dependencies
