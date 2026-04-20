import requests
from bs4 import BeautifulSoup


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


def _score_pubmed_article(article_text: str, query_terms: list[str]) -> int:
    score = 0
    lower_text = article_text.lower()

    for term in query_terms:
        if term in lower_text:
            score += 1

    return score


def fetch_pubmed_articles_with_metadata(query: str, max_results=3, use_mock_if_empty=True):
    headers = {"User-Agent": "Mozilla/5.0"}

    query_terms = _normalize_query_terms(query)

    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results * 5,
        "retmode": "json",
        "sort": "relevance"
    }

    try:
        search_response = requests.get(
            search_url,
            params=search_params,
            headers=headers,
            timeout=15
        )
        search_response.raise_for_status()

        search_json = search_response.json()
        id_list = search_json.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            raise ValueError("No PubMed IDs found for this query.")

        ids = ",".join(id_list)

        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ids,
            "retmode": "xml"
        }

        fetch_response = requests.get(
            fetch_url,
            params=fetch_params,
            headers=headers,
            timeout=15
        )
        fetch_response.raise_for_status()

        soup = BeautifulSoup(fetch_response.text, "lxml")
        articles_xml = soup.find_all("pubmedarticle")

        scored_articles = []

        for article, pmid in zip(articles_xml, id_list):
            title_tag = article.find("articletitle")
            abstract_tag = article.find("abstract")
            date_tag = article.find("pubdate")
            author_tags = article.find_all("author")

            title = title_tag.get_text(strip=True) if title_tag else "No title"
            abstract = abstract_tag.get_text(separator=" ", strip=True) if abstract_tag else "No abstract available"

            authors = []
            for author in author_tags:
                last_name = author.find("lastname")
                fore_name = author.find("forename")

                if last_name and fore_name:
                    full_name = f"{fore_name.get_text()} {last_name.get_text()}"
                    authors.append(full_name)
                elif last_name:
                    authors.append(last_name.get_text())

            if not authors:
                authors = ["No authors listed"]

            publication_date = "No date"
            if date_tag:
                year_tag = date_tag.find("year")
                month_tag = date_tag.find("month")

                if year_tag and month_tag:
                    publication_date = f"{month_tag.get_text()} {year_tag.get_text()}"
                elif year_tag:
                    publication_date = year_tag.get_text()

            article_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            combined_text = f"{title} {abstract}"
            relevance_score = _score_pubmed_article(combined_text, query_terms)

            article_info = {
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "publication_date": publication_date,
                "article_url": article_url
            }

            scored_articles.append((relevance_score, article_info))

        scored_articles.sort(key=lambda item: item[0], reverse=True)

        final_articles = []
        for score_value, article_data in scored_articles:
            if score_value > 0:
                final_articles.append(article_data)

        if len(final_articles) < max_results:
            for score_value, article_data in scored_articles:
                if article_data not in final_articles:
                    final_articles.append(article_data)
                if len(final_articles) >= max_results:
                    break

        final_articles = final_articles[:max_results]

        if not final_articles and use_mock_if_empty:
            return [{
                "title": "Simulated Study on Fever",
                "abstract": "This is a simulated abstract on the treatment of fever in adults.",
                "authors": ["John Doe", "Jane Smith"],
                "publication_date": "March 2024",
                "article_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"
            }]

        return final_articles

    except Exception as e:
        print(f"Error during PubMed fetch: {e}")

        if use_mock_if_empty:
            return [{
                "title": "Simulated Study on Fever",
                "abstract": "This is a simulated abstract on the treatment of fever in adults.",
                "authors": ["John Doe", "Jane Smith"],
                "publication_date": "March 2024",
                "article_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"
            }]

        return [{"message": f"Error: {e}"}]