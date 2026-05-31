import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from functions.symptom_extractor import extract_symptoms

DATASET_PATH = Path("experiments/evaluation_dataset.csv")
RESULTS_DIR = Path("experiments/results")
PREDICTIONS_PATH = RESULTS_DIR / "extraction_predictions.csv"
METRICS_PATH = RESULTS_DIR / "extraction_metrics.csv"
ABLATION_PATH = RESULTS_DIR / "ablation_results.csv"
ERROR_ANALYSIS_PATH = RESULTS_DIR / "error_analysis.csv"
RUN_CONFIG_PATH = RESULTS_DIR / "run_config.json"

NORMALIZATION_MAP = {
    "shortness of breath": "dyspnea",
    "trouble breathing": "dyspnea",
    "difficulty breathing": "dyspnea",
    "breathlessness": "dyspnea",
    "chest tightness": "chest pain",
    "tired": "fatigue",
    "tiredness": "fatigue",
    "low energy": "fatigue",
    "swollen legs": "edema",
    "swollen feet": "edema",
    "throwing up": "vomiting",
    "belly pain": "abdominal pain",
    "stomach pain": "abdominal pain",
    "back pain": "lower back pain",
    "body pain": "joint pain",
    "body ache": "joint pain",
}

KNOWN_SYMPTOMS = [
    "fever",
    "headache",
    "sore throat",
    "nausea",
    "vomiting",
    "abdominal pain",
    "joint pain",
    "runny nose",
    "cough",
    "watery eyes",
    "sneezing",
    "chills",
    "dizziness",
    "lower back pain",
    "edema",
    "fatigue",
    "excessive thirst",
    "frequent urination",
    "weakness",
    "insomnia",
    "depression",
    "dyspnea",
    "chest pain",
    "sweating",
    "confusion",
    "blurred vision",
    "fainting",
    "pelvic pain",
]


def normalize_symptom(symptom: str) -> str:
    cleaned_symptom = symptom.strip().lower()
    normalized_symptom = NORMALIZATION_MAP.get(cleaned_symptom, cleaned_symptom)
    return normalized_symptom


def parse_symptom_list(value: str) -> List[str]:
    if value is None:
        return []

    cleaned_value = str(value).strip()

    if not cleaned_value or cleaned_value.lower() == "none":
        return []

    symptom_parts = [item.strip() for item in cleaned_value.split(";")]

    normalized_symptoms = []

    for symptom in symptom_parts:
        normalized_symptom = normalize_symptom(symptom)

        if normalized_symptom and normalized_symptom not in normalized_symptoms:
            normalized_symptoms.append(normalized_symptom)

    return normalized_symptoms


def normalize_symptom_list(symptoms: List[str]) -> List[str]:
    normalized_symptoms = []

    for symptom in symptoms:
        normalized_symptom = normalize_symptom(str(symptom))

        if normalized_symptom and normalized_symptom not in normalized_symptoms:
            normalized_symptoms.append(normalized_symptom)

    return normalized_symptoms


def heuristic_llm_extraction(text: str) -> List[str]:
    input_text = text.lower()
    detected_symptoms = []

    phrase_map = {
        "cannot catch my breath": "dyspnea",
        "shortness of breath": "dyspnea",
        "can't breathe": "dyspnea",
        "cannot breathe": "dyspnea",
        "trouble breathing": "dyspnea",
        "chest feels tight": "chest pain",
        "chest tightness": "chest pain",
        "chest hurts": "chest pain",
        "head is pounding": "headache",
        "feel hot": "fever",
        "high temperature": "fever",
        "throwing up": "vomiting",
        "feel like vomiting": "vomiting",
        "stomach pain": "abdominal pain",
        "belly is hurting": "abdominal pain",
        "belly pain": "abdominal pain",
        "body aches": "joint pain",
        "body pain": "joint pain",
        "no energy": "fatigue",
        "feel drained": "fatigue",
        "legs are swollen": "edema",
        "feet are swollen": "edema",
        "low mood": "depression",
        "feel down": "depression",
        "cannot sleep": "insomnia",
        "can't sleep": "insomnia",
        "poor sleep": "insomnia",
        "lightheaded": "dizziness",
    }

    for phrase, symptom in phrase_map.items():
        if phrase in input_text and symptom not in detected_symptoms:
            detected_symptoms.append(symptom)

    for symptom in KNOWN_SYMPTOMS:
        if symptom in input_text and symptom not in detected_symptoms:
            detected_symptoms.append(symptom)

    return detected_symptoms


def llm_only_extraction(text: str) -> List[str]:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return heuristic_llm_extraction(text)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = f"""
Extract only symptom terms from the user text.

Rules:
1. Return valid JSON only.
2. JSON format must be: {{"symptoms": ["symptom 1", "symptom 2"]}}
3. Use lowercase symptom names.
4. Do not include diagnoses.
5. Do not include medications.
6. Do not include body parts unless they are part of a symptom phrase.
7. If there are no symptoms, return {{"symptoms": []}}.

User text:
{text}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You extract symptoms from health-related free text for research evaluation only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    output_text = response.choices[0].message.content.strip()

    try:
        parsed_output = json.loads(output_text)
    except json.JSONDecodeError:
        start_index = output_text.find("{")
        end_index = output_text.rfind("}") + 1
        parsed_output = json.loads(output_text[start_index:end_index])

    symptoms = parsed_output.get("symptoms", [])

    if not isinstance(symptoms, list):
        symptoms = []

    cleaned_symptoms = []

    for symptom in symptoms:
        if isinstance(symptom, str):
            cleaned_symptoms.append(symptom)

    normalized_symptoms = normalize_symptom_list(cleaned_symptoms)
    return normalized_symptoms


def rule_based_extraction(text: str) -> List[str]:
    symptoms = extract_symptoms(text)
    normalized_symptoms = normalize_symptom_list(symptoms)
    return normalized_symptoms


def hybrid_extraction(text: str) -> List[str]:
    rule_symptoms = rule_based_extraction(text)

    if rule_symptoms:
        return rule_symptoms

    llm_symptoms = llm_only_extraction(text)
    normalized_symptoms = normalize_symptom_list(llm_symptoms)

    return normalized_symptoms


def calculate_row_metrics(gold: Set[str], predicted: Set[str]) -> Dict[str, Any]:
    true_positive = len(gold.intersection(predicted))
    false_positive = len(predicted.difference(gold))
    false_negative = len(gold.difference(predicted))
    exact_match = gold == predicted

    row_metrics = {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "exact_match": exact_match,
    }

    return row_metrics


def calculate_overall_metrics(rows: List[Dict[str, Any]], method_name: str) -> Dict[str, Any]:
    total_true_positive = sum(row["true_positive"] for row in rows)
    total_false_positive = sum(row["false_positive"] for row in rows)
    total_false_negative = sum(row["false_negative"] for row in rows)
    total_exact_match = sum(1 for row in rows if row["exact_match"])
    total_cases = len(rows)

    precision_denominator = total_true_positive + total_false_positive
    recall_denominator = total_true_positive + total_false_negative

    if precision_denominator > 0:
        precision = total_true_positive / precision_denominator
    else:
        precision = 0.0

    if recall_denominator > 0:
        recall = total_true_positive / recall_denominator
    else:
        recall = 0.0

    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    if total_cases > 0:
        exact_match_accuracy = total_exact_match / total_cases
        average_latency = sum(row["latency_seconds"] for row in rows) / total_cases
    else:
        exact_match_accuracy = 0.0
        average_latency = 0.0

    overall_metrics = {
        "method": method_name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "exact_match_accuracy": round(exact_match_accuracy, 4),
        "average_latency_seconds": round(average_latency, 4),
        "total_true_positive": total_true_positive,
        "total_false_positive": total_false_positive,
        "total_false_negative": total_false_negative,
        "total_cases": total_cases,
    }

    return overall_metrics


def run_method(method_name: str, function_to_run, dataset: pd.DataFrame):
    prediction_rows = []
    metric_rows = []

    for _, row in dataset.iterrows():
        case_id = row["case_id"]
        symptom_description = row["symptom_description"]
        category = row["category"]
        gold_symptoms = parse_symptom_list(row["gold_standard_symptoms"])

        start_time = time.perf_counter()
        predicted_symptoms = function_to_run(symptom_description)
        end_time = time.perf_counter()

        latency_seconds = end_time - start_time

        gold_set = set(gold_symptoms)
        predicted_set = set(predicted_symptoms)

        row_metrics = calculate_row_metrics(gold_set, predicted_set)
        row_metrics["latency_seconds"] = latency_seconds

        prediction_row = {
            "case_id": case_id,
            "category": category,
            "symptom_description": symptom_description,
            "method": method_name,
            "gold_standard_symptoms": "; ".join(gold_symptoms) if gold_symptoms else "none",
            "predicted_symptoms": "; ".join(predicted_symptoms) if predicted_symptoms else "none",
            "true_positive": row_metrics["true_positive"],
            "false_positive": row_metrics["false_positive"],
            "false_negative": row_metrics["false_negative"],
            "exact_match": row_metrics["exact_match"],
            "latency_seconds": round(latency_seconds, 4),
        }

        prediction_rows.append(prediction_row)
        metric_rows.append(row_metrics)

    overall_metrics = calculate_overall_metrics(metric_rows, method_name)

    return prediction_rows, overall_metrics


def create_error_analysis(prediction_dataframe: pd.DataFrame):
    hybrid_rows = prediction_dataframe[prediction_dataframe["method"] == "hybrid_rule_plus_llm"]
    error_rows = hybrid_rows[hybrid_rows["exact_match"] == False]

    selected_rows = []

    for _, row in error_rows.head(25).iterrows():
        error_row = {
            "case_id": row["case_id"],
            "category": row["category"],
            "symptom_description": row["symptom_description"],
            "gold_standard_symptoms": row["gold_standard_symptoms"],
            "predicted_symptoms": row["predicted_symptoms"],
            "observed_error": "Predicted symptom set did not exactly match the gold-standard set.",
            "likely_cause": "Possible lexicon gap, symptom normalization issue, LLM over-extraction, or ambiguous wording.",
        }

        selected_rows.append(error_row)

    error_dataframe = pd.DataFrame(selected_rows)
    error_dataframe.to_csv(ERROR_ANALYSIS_PATH, index=False)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run this first: python experiments/create_dataset.py"
        )

    dataset = pd.read_csv(DATASET_PATH, dtype={"case_id": str})

    method_config = [
        ("rule_based_only", rule_based_extraction),
        ("llm_only", llm_only_extraction),
        ("hybrid_rule_plus_llm", hybrid_extraction),
    ]

    all_prediction_rows = []
    all_metric_rows = []

    for method_name, function_to_run in method_config:
        print(f"Running method: {method_name}")

        prediction_rows, overall_metrics = run_method(
            method_name,
            function_to_run,
            dataset,
        )

        all_prediction_rows.extend(prediction_rows)
        all_metric_rows.append(overall_metrics)

    prediction_dataframe = pd.DataFrame(all_prediction_rows)
    metrics_dataframe = pd.DataFrame(all_metric_rows)

    prediction_dataframe.to_csv(PREDICTIONS_PATH, index=False)
    metrics_dataframe.to_csv(METRICS_PATH, index=False)

    ablation_dataframe = metrics_dataframe.copy()
    ablation_dataframe["system_variant"] = ablation_dataframe["method"]

    ablation_dataframe = ablation_dataframe[
        [
            "system_variant",
            "precision",
            "recall",
            "f1_score",
            "exact_match_accuracy",
            "average_latency_seconds",
        ]
    ]

    ablation_dataframe.to_csv(ABLATION_PATH, index=False)

    create_error_analysis(prediction_dataframe)

    run_config = {
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "note": (
            "If openai_api_key_present is false, llm_only and hybrid fallback used local heuristic extraction. "
            "Use an OpenAI API key for manuscript-quality LLM-based results."
        ),
    }

    with RUN_CONFIG_PATH.open("w", encoding="utf-8") as config_file:
        json.dump(run_config, config_file, indent=2)

    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved ablation results: {ABLATION_PATH}")
    print(f"Saved error analysis: {ERROR_ANALYSIS_PATH}")
    print(f"Saved run config: {RUN_CONFIG_PATH}")


if __name__ == "__main__":
    main()