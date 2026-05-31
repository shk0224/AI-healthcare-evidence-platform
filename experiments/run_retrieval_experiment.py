import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from functions.pubmed_articles import fetch_pubmed_articles_with_metadata
from functions.clinicaltrials_articles import fetch_clinicaltrials_articles_with_metadata

DATASET_PATH = Path("experiments/evaluation_dataset.csv")
RESULTS_DIR = Path("experiments/results")
PUBMED_RESULTS_PATH = RESULTS_DIR / "pubmed_retrieval_results.csv"
TRIAL_RESULTS_PATH = RESULTS_DIR / "clinicaltrials_retrieval_results.csv"
PUBMED_REVIEW_PATH = RESULTS_DIR / "pubmed_relevance_review.csv"
TRIAL_REVIEW_PATH = RESULTS_DIR / "clinicaltrials_relevance_review.csv"

def parse_symptom_list(value: str) -> List[str]:
    if value is None:
        return []

    value = str(value).strip()

    if not value or value.lower() == "none":
        return []

    parts = [item.strip().lower() for item in value.split(";")]
    final_parts = []

    for item in parts:
        if item and item not in final_parts:
            final_parts.append(item)

    return final_parts

def build_query(symptoms: List[str]) -> str:
    if not symptoms:
        return ""

    query = " ".join(symptoms)
    return query

def auto_relevance_score(text: str, symptoms: List[str]) -> int:
    lower_text = text.lower()
    score = 0

    for symptom in symptoms:
        symptom_terms = symptom.split()

        for term in symptom_terms:
            if len(term) > 2 and term in lower_text:
                score += 1
                break

    if score >= 2:
        return 2

    if score == 1:
        return 1

    return 0

def run_pubmed(dataset: pd.DataFrame):
    result_rows = []
    review_rows = []

    for _, row in dataset.iterrows():
        case_id = row["case_id"]
        category = row["category"]
        symptom_description = row["symptom_description"]
        symptoms = parse_symptom_list(row["gold_standard_symptoms"])
        query = build_query(symptoms)

        if not query:
            continue

        print(f"PubMed retrieval for case {case_id}: {query}")

        start_time = time.perf_counter()
        articles = fetch_pubmed_articles_with_metadata(
            query=query,
            max_results=5,
            use_mock_if_empty=False,
        )
        end_time = time.perf_counter()

        latency_seconds = end_time - start_time

        for rank_index, article in enumerate(articles, start=1):
            title = article.get("title", "")
            abstract = article.get("abstract", "")
            article_url = article.get("article_url", "")

            combined_text = f"{title} {abstract}"
            automatic_relevance = auto_relevance_score(combined_text, symptoms)

            result_row = {
                "case_id": case_id,
                "category": category,
                "symptom_description": symptom_description,
                "query": query,
                "rank": rank_index,
                "title": title,
                "abstract": abstract,
                "article_url": article_url,
                "automatic_relevance_0_to_2": automatic_relevance,
                "latency_seconds": round(latency_seconds, 4),
            }

            review_row = {
                **result_row,
                "manual_relevance_0_to_2": "",
                "manual_summary_faithfulness_0_to_2": "",
                "reviewer_notes": "",
            }

            result_rows.append(result_row)
            review_rows.append(review_row)

    pd.DataFrame(result_rows).to_csv(PUBMED_RESULTS_PATH, index=False)
    pd.DataFrame(review_rows).to_csv(PUBMED_REVIEW_PATH, index=False)

def run_clinical_trials(dataset: pd.DataFrame):
    result_rows = []
    review_rows = []

    for _, row in dataset.iterrows():
        case_id = row["case_id"]
        category = row["category"]
        symptom_description = row["symptom_description"]
        symptoms = parse_symptom_list(row["gold_standard_symptoms"])
        query = build_query(symptoms)

        if not query:
            continue

        print(f"ClinicalTrials.gov retrieval for case {case_id}: {query}")

        start_time = time.perf_counter()
        trials = fetch_clinicaltrials_articles_with_metadata(
            query=query,
            max_results=5,
            use_mock_if_empty=False,
        )
        end_time = time.perf_counter()

        latency_seconds = end_time - start_time

        for rank_index, trial in enumerate(trials, start=1):
            title = trial.get("title", "")
            abstract = trial.get("abstract", "")
            conditions = trial.get("conditions", [])
            status = trial.get("overall_status", "")
            article_url = trial.get("article_url", "")

            conditions_text = "; ".join(conditions) if isinstance(conditions, list) else str(conditions)
            combined_text = f"{title} {abstract} {conditions_text}"
            automatic_relevance = auto_relevance_score(combined_text, symptoms)

            result_row = {
                "case_id": case_id,
                "category": category,
                "symptom_description": symptom_description,
                "query": query,
                "rank": rank_index,
                "title": title,
                "abstract": abstract,
                "conditions": conditions_text,
                "overall_status": status,
                "article_url": article_url,
                "automatic_relevance_0_to_2": automatic_relevance,
                "latency_seconds": round(latency_seconds, 4),
            }

            review_row = {
                **result_row,
                "manual_trial_relevance_0_to_2": "",
                "manual_recruitment_status_correct_0_or_1": "",
                "reviewer_notes": "",
            }

            result_rows.append(result_row)
            review_rows.append(review_row)

    pd.DataFrame(result_rows).to_csv(TRIAL_RESULTS_PATH, index=False)
    pd.DataFrame(review_rows).to_csv(TRIAL_REVIEW_PATH, index=False)

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run this first: python experiments/create_dataset.py"
        )

    dataset = pd.read_csv(DATASET_PATH, dtype={"case_id": str})

    run_pubmed(dataset)
    run_clinical_trials(dataset)

    print(f"Saved PubMed results: {PUBMED_RESULTS_PATH}")
    print(f"Saved PubMed review file: {PUBMED_REVIEW_PATH}")
    print(f"Saved ClinicalTrials results: {TRIAL_RESULTS_PATH}")
    print(f"Saved ClinicalTrials review file: {TRIAL_REVIEW_PATH}")

if __name__ == "__main__":
    main()
