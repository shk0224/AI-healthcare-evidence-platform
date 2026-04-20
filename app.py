from fastapi import FastAPI
from pydantic import BaseModel

from functions.symptom_extractor import extract_symptoms
from functions.diagnosis_symptoms import get_diagnosis
from functions.pubmed_articles import fetch_pubmed_articles_with_metadata
from functions.clinicaltrials_articles import fetch_clinicaltrials_articles_with_metadata
from functions.summerize_pubmed import summarize_text
from functions.summarize_clinicaltrials import summarize_clinical_trials


app = FastAPI()


class SymptomInput(BaseModel):
    description: str


def build_search_query(symptoms: list[str], original_text: str) -> str:
    if symptoms:
        query = " OR ".join(symptoms)
        return query

    cleaned_text = original_text.strip()
    return cleaned_text


@app.post("/diag")
def diagnosis(data: SymptomInput):
    symptom = extract_symptoms(data.description)

    diagnosis_result = get_diagnosis(symptom)

    search_query = build_search_query(symptom, data.description)

    pubmed_articles = fetch_pubmed_articles_with_metadata(search_query, max_results=3)

    clinical_trials = fetch_clinicaltrials_articles_with_metadata(search_query, max_results=3)

    pubmed_text_parts = []

    for article in pubmed_articles:
        title = article.get("title", "No title")
        abstract = article.get("abstract", "No abstract available")
        publication_date = article.get("publication_date", "No date")
        article_url = article.get("article_url", "No URL available")

        article_text = (
            f"Title: {title}\n"
            f"Date: {publication_date}\n"
            f"Abstract: {abstract}\n"
            f"URL: {article_url}"
        )
        pubmed_text_parts.append(article_text)

    pubmed_text_for_summary = "\n\n".join(pubmed_text_parts)

    if pubmed_text_for_summary.strip():
        pubmed_summary = summarize_text(pubmed_text_for_summary[:5000])
    else:
        pubmed_summary = "No PubMed summary available."

    clinical_text_parts = []

    for trial in clinical_trials:
        title = trial.get("title", "No title")
        abstract = trial.get("abstract", "No summary available")
        conditions = trial.get("conditions", ["No condition listed"])
        interventions = trial.get("interventions", ["Not specified"])
        overall_status = trial.get("overall_status", "Unknown")
        study_type = trial.get("study_type", "Unknown")
        publication_date = trial.get("publication_date", "No date")
        article_url = trial.get("article_url", "No URL available")

        trial_text = (
            f"Title: {title}\n"
            f"Condition: {', '.join(conditions)}\n"
            f"Intervention: {', '.join(interventions)}\n"
            f"Status: {overall_status}\n"
            f"Study Type: {study_type}\n"
            f"Summary: {abstract}\n"
            f"Date: {publication_date}\n"
            f"URL: {article_url}"
        )

        clinical_text_parts.append(trial_text)

    clinical_text_for_summary = "\n\n".join(clinical_text_parts)

    if clinical_text_for_summary.strip():
        clinical_trials_summary = summarize_clinical_trials(clinical_text_for_summary[:5000])
    else:
        clinical_trials_summary = "No ClinicalTrials summary available."

    return {
        "symptom": symptom,
        "search_query": search_query,
        "diagnosis": diagnosis_result,
        "pubmed_articles": pubmed_articles,
        "pubmed_summary": pubmed_summary,
        "clinical_trials": clinical_trials,
        "clinical_trials_summary": clinical_trials_summary
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)