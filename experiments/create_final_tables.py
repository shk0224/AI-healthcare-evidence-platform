from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("experiments/results")
METRICS_PATH = RESULTS_DIR / "extraction_metrics.csv"
ABLATION_PATH = RESULTS_DIR / "ablation_results.csv"
PUBMED_RESULTS_PATH = RESULTS_DIR / "pubmed_retrieval_results.csv"
TRIAL_RESULTS_PATH = RESULTS_DIR / "clinicaltrials_retrieval_results.csv"
FINAL_SUMMARY_PATH = RESULTS_DIR / "final_tables_summary.txt"
MANUSCRIPT_WORDING_PATH = RESULTS_DIR / "manuscript_results_wording.txt"

def safe_read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()

def compute_topk_metrics(dataframe: pd.DataFrame, relevance_column: str):
    if dataframe.empty:
        return {}

    case_ids = sorted(dataframe["case_id"].unique())
    total_cases = len(case_ids)

    top1_success = 0
    top3_success = 0
    top5_success = 0

    relevance_values = []

    for case_id in case_ids:
        case_rows = dataframe[dataframe["case_id"] == case_id].copy()
        case_rows = case_rows.sort_values("rank")

        top1_rows = case_rows[case_rows["rank"] <= 1]
        top3_rows = case_rows[case_rows["rank"] <= 3]
        top5_rows = case_rows[case_rows["rank"] <= 5]

        if (top1_rows[relevance_column] > 0).any():
            top1_success += 1

        if (top3_rows[relevance_column] > 0).any():
            top3_success += 1

        if (top5_rows[relevance_column] > 0).any():
            top5_success += 1

        relevance_values.extend(case_rows[relevance_column].dropna().tolist())

    mean_latency = dataframe.groupby("case_id")["latency_seconds"].first().mean()

    return {
        "total_cases": total_cases,
        "top1_relevance": round(top1_success / total_cases, 4) if total_cases else 0,
        "top3_relevance": round(top3_success / total_cases, 4) if total_cases else 0,
        "top5_relevance": round(top5_success / total_cases, 4) if total_cases else 0,
        "mean_relevance_0_to_2": round(sum(relevance_values) / len(relevance_values), 4) if relevance_values else 0,
        "mean_latency_seconds": round(mean_latency, 4) if total_cases else 0,
    }

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_dataframe = safe_read_csv(METRICS_PATH)
    ablation_dataframe = safe_read_csv(ABLATION_PATH)
    pubmed_dataframe = safe_read_csv(PUBMED_RESULTS_PATH)
    trial_dataframe = safe_read_csv(TRIAL_RESULTS_PATH)

    pubmed_metrics = compute_topk_metrics(pubmed_dataframe, "automatic_relevance_0_to_2")
    trial_metrics = compute_topk_metrics(trial_dataframe, "automatic_relevance_0_to_2")

    summary_lines = []

    summary_lines.append("TABLE 1: Symptom Extraction Performance")
    summary_lines.append(metrics_dataframe.to_string(index=False) if not metrics_dataframe.empty else "No extraction metrics found.")
    summary_lines.append("")

    summary_lines.append("TABLE 2: Ablation Analysis")
    summary_lines.append(ablation_dataframe.to_string(index=False) if not ablation_dataframe.empty else "No ablation results found.")
    summary_lines.append("")

    summary_lines.append("TABLE 3: PubMed Retrieval Performance - Automatic Pre-Scoring")
    summary_lines.append(str(pubmed_metrics))
    summary_lines.append("")

    summary_lines.append("TABLE 4: ClinicalTrials.gov Retrieval Performance - Automatic Pre-Scoring")
    summary_lines.append(str(trial_metrics))
    summary_lines.append("")

    FINAL_SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

    wording = f"""
Experimental Results Draft Wording

The framework was evaluated using a synthetic but realistic dataset of 200 free-text symptom-description scenarios. The dataset covered common acute symptoms, chronic-disease-related symptoms, colloquial symptom descriptions, multi-symptom descriptions, red-flag-like presentations, and negative or irrelevant inputs. Each case was assigned gold-standard symptom labels, and equivalent lay expressions were normalized to common symptom terms before scoring.

Symptom extraction was evaluated using three methods: rule-based extraction, LLM-only extraction, and the proposed hybrid rule-based plus LLM fallback method. The evaluation measured precision, recall, F1-score, exact-match accuracy, and average latency per case. The extraction results are saved in extraction_metrics.csv and should be inserted into the manuscript after manual verification of the run configuration.

The retrieval modules were evaluated as information retrieval components. For PubMed retrieval, extracted symptom terms were used as search queries and the top retrieved records were assessed using top-k relevance and latency. For ClinicalTrials.gov retrieval, returned studies were evaluated as preliminary trial-discovery outputs rather than definitive eligibility matches. The current output files include automatic relevance pre-scoring. For publication-quality results, the review files should be manually reviewed and the manual relevance columns should be used for final reporting.

Important manuscript note:
If the LLM extraction was run without an OPENAI_API_KEY, the LLM-only and hybrid fallback results are not true LLM results. They should not be reported as LLM-based experimental findings. Re-run with OPENAI_API_KEY enabled before using final numbers in the paper.
"""

    MANUSCRIPT_WORDING_PATH.write_text(wording.strip(), encoding="utf-8")

    print(f"Saved final summary: {FINAL_SUMMARY_PATH}")
    print(f"Saved manuscript wording: {MANUSCRIPT_WORDING_PATH}")

if __name__ == "__main__":
    main()
