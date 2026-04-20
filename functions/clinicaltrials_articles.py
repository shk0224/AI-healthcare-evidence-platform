import requests


def _normalize_query_terms(query: str) -> list[str]:
    raw_terms = query.lower().replace(",", " ").split()

    stop_words = {
        "and", "or", "the", "a", "an", "of", "for", "in", "on",
        "with", "to", "my", "is", "are"
    }

    cleaned_terms = []
    for term in raw_terms:
        if term.strip() and term not in stop_words and len(term) > 2:
            cleaned_terms.append(term.strip())

    unique_terms = []
    for term in cleaned_terms:
        if term not in unique_terms:
            unique_terms.append(term)

    return unique_terms


def _score_trial_text(text: str, query_terms: list[str]) -> int:
    score = 0
    lower_text = text.lower()

    for term in query_terms:
        if term in lower_text:
            score += 1

    return score


def fetch_clinicaltrials_articles_with_metadata(query: str, max_results=3, use_mock_if_empty=True):
    headers = {"User-Agent": "Mozilla/5.0"}
    query_terms = _normalize_query_terms(query)

    search_url = "https://clinicaltrials.gov/api/v2/studies"
    search_params = {
        "query.term": query,
        "pageSize": max_results * 5,
        "format": "json"
    }

    try:
        search_response = requests.get(
            search_url,
            params=search_params,
            headers=headers,
            timeout=15
        )
        search_response.raise_for_status()

        search_data = search_response.json()
        studies = search_data.get("studies", [])

        if not studies:
            raise ValueError("No studies found for this query.")

        scored_trials = []

        for study in studies:
            protocol_section = study.get("protocolSection", {})
            identification_module = protocol_section.get("identificationModule", {})
            description_module = protocol_section.get("descriptionModule", {})
            status_module = protocol_section.get("statusModule", {})
            conditions_module = protocol_section.get("conditionsModule", {})
            sponsor_module = protocol_section.get("sponsorCollaboratorsModule", {})
            arms_module = protocol_section.get("armsInterventionsModule", {})
            design_module = protocol_section.get("designModule", {})

            title = identification_module.get("briefTitle", "No title")
            abstract = description_module.get("briefSummary", "No summary available")

            sponsor_name = "No sponsor listed"
            lead_sponsor = sponsor_module.get("leadSponsor", {})
            if isinstance(lead_sponsor, dict):
                sponsor_name = lead_sponsor.get("name", "No sponsor listed")

            authors = [sponsor_name]

            publication_date = "No date"
            start_date_struct = status_module.get("startDateStruct", {})
            if isinstance(start_date_struct, dict):
                publication_date = start_date_struct.get("date", "No date")

            overall_status = status_module.get("overallStatus", "Unknown")
            study_type = design_module.get("studyType", "Unknown")

            conditions = conditions_module.get("conditions", [])
            if not conditions:
                conditions = ["No condition listed"]

            interventions = []
            intervention_list = arms_module.get("interventions", [])
            if isinstance(intervention_list, list):
                for intervention in intervention_list:
                    intervention_name = intervention.get("name")
                    if intervention_name:
                        interventions.append(intervention_name)

            if not interventions:
                interventions = ["Not specified"]

            nct_id = identification_module.get("nctId", "")
            if nct_id:
                article_url = f"https://clinicaltrials.gov/study/{nct_id}"
            else:
                article_url = "No URL available"

            combined_text = " ".join(
                [
                    title,
                    abstract,
                    " ".join(conditions),
                    " ".join(interventions),
                    overall_status,
                    study_type
                ]
            )

            relevance_score = _score_trial_text(combined_text, query_terms)

            article_info = {
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "publication_date": publication_date,
                "conditions": conditions,
                "interventions": interventions,
                "overall_status": overall_status,
                "study_type": study_type,
                "article_url": article_url
            }

            scored_trials.append((relevance_score, article_info))

        scored_trials.sort(key=lambda item: item[0], reverse=True)

        final_trials = []
        for score_value, trial_data in scored_trials:
            if score_value > 0:
                final_trials.append(trial_data)

        if len(final_trials) < max_results:
            for score_value, trial_data in scored_trials:
                if trial_data not in final_trials:
                    final_trials.append(trial_data)
                if len(final_trials) >= max_results:
                    break

        final_trials = final_trials[:max_results]

        if not final_trials and use_mock_if_empty:
            return [{
                "title": "Simulated Clinical Trial on Fever",
                "abstract": "This is a simulated summary of a clinical trial on fever treatment.",
                "authors": ["Mock Sponsor"],
                "publication_date": "2024-03-01",
                "conditions": ["Fever"],
                "interventions": ["Supportive care"],
                "overall_status": "Unknown",
                "study_type": "Unknown",
                "article_url": "https://clinicaltrials.gov/study/NCT00000000"
            }]

        return final_trials

    except Exception as e:
        print(f"Error during ClinicalTrials.gov fetch: {e}")

        if use_mock_if_empty:
            return [{
                "title": "Simulated Clinical Trial on Fever",
                "abstract": "This is a simulated summary of a clinical trial on fever treatment.",
                "authors": ["Mock Sponsor"],
                "publication_date": "2024-03-01",
                "conditions": ["Fever"],
                "interventions": ["Supportive care"],
                "overall_status": "Unknown",
                "study_type": "Unknown",
                "article_url": "https://clinicaltrials.gov/study/NCT00000000"
            }]

        return [{"message": f"Error: {e}"}]