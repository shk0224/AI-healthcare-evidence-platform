# Experiment Package for AI Healthcare Evidence Platform

## Where to put this folder

Put the `experiments` folder at the root level of your existing project:

```text
AI-healthcare-evidence-platform/
├── functions/
├── experiments/
├── app.py
├── main.py
├── requirements.txt
└── pyproject.toml
```

## Install requirements

From the project root folder:

```bash
pip install -r requirements.txt
pip install pandas scikit-learn openai python-dotenv
```

## Step 1: Create 200-case synthetic dataset

```bash
python experiments/create_dataset.py
```

This creates:

```text
experiments/evaluation_dataset.csv
```

## Step 2: Run symptom extraction experiment

If you have OpenAI API key:

```bash
set OPENAI_API_KEY=your_key_here
python experiments/run_extraction_experiment.py
```

For Mac/Linux:

```bash
export OPENAI_API_KEY=your_key_here
python experiments/run_extraction_experiment.py
```

This creates:

```text
experiments/results/extraction_predictions.csv
experiments/results/extraction_metrics.csv
experiments/results/ablation_results.csv
experiments/results/error_analysis.csv
```

Important: If no API key is found, the script uses a local heuristic fallback and clearly marks the run as non-LLM fallback. For manuscript-quality results, use real API-based LLM extraction.

## Step 3: Run PubMed and ClinicalTrials retrieval experiment

```bash
python experiments/run_retrieval_experiment.py
```

This creates:

```text
experiments/results/pubmed_retrieval_results.csv
experiments/results/clinicaltrials_retrieval_results.csv
experiments/results/pubmed_relevance_review.csv
experiments/results/clinicaltrials_relevance_review.csv
```

The review files include manual scoring columns. For publication, manually review the returned titles/abstracts and fill the relevance columns.

## Step 4: Create final tables and manuscript wording

```bash
python experiments/create_final_tables.py
```

This creates:

```text
experiments/results/final_tables_summary.txt
experiments/results/manuscript_results_wording.txt
```

## Recommended order

```bash
python experiments/create_dataset.py
python experiments/run_extraction_experiment.py
python experiments/run_retrieval_experiment.py
python experiments/create_final_tables.py
```

## Notes for the paper

Use the generated results to replace the current generated/example tables in the manuscript.

Do not claim the results are from real patient data. The dataset is synthetic but realistic and should be described exactly that way.
