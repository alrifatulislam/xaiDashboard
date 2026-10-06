"""XAI Dashboard for Summarization

Interactive dashboard to generate summaries and visualize explanations (Attention, IG, LIME, SHAP) for different summarization models and datasets.


##  enable LLM judge (Ollama)


To use the LLM-judge feature, you need Ollama running locally in terminal.

1. Install and start Ollama.
2. Pull a model (example):

    ollama pull atla/selene-mini

3. Make sure `llm_eval.py` has:

    OLLAMA_MODEL_NAME = "atla/selene-mini:latest"


## Environment setup


From the project folder (where `app.py`, `pipeline.py`, `requirements.txt` live):

    # Create venv (only once)
    python -m venv .venv

    # Activate venv (Windows, PowerShell)
    .\\.venv\\Scripts\\activate

    # Install dependencies
    pip install -r requirements.txt

If `python` doesn’t work, use `python3` instead.


## Run the Streamlit app


    .\\.venv\\Scripts\\activate
    streamlit run app.py

Then open the URL shown in the terminal (usually http://localhost:8501).


## How to use the app


In the left sidebar:

1. Choose dataset (e.g., XSum, SAMSum).
2. Choose sample index (0, 1, 2, …).
3. Choose summarization model from the dropdown.
4. Select which explainers to run:
   - Attention
   - Integrated Gradients (IG)
   - LIME
   - SHAP (if enabled)
5. (Optional) Enable LLM judge to get scores and comments from a local LLM.

The main page shows:

- Source document
- Reference summary & generated summary
- ROUGE-L score in a highlighted box
- Colored-word explanations for each method
- Faithfulness / latency / consistency tables
- (Optional) LLM judge table with scores + comments



## Debug mode (pipeline only)


To run the pipeline without the UI (for debugging / printing results):

    .\\.venv\\Scripts\\activate
    python pipeline.py

You can edit the `config` dict at the bottom of `pipeline.py` to change dataset, model, sample index, and which explainers run.
"""