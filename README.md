# 🧠 AI Healthcare Evidence Platform

An AI-powered healthcare decision-support system that analyzes user symptoms and retrieves real medical evidence from **PubMed** and **ClinicalTrials.gov**, combined with LLM-based explanations.

> ⚠️ This project is for research and educational purposes only. It is NOT a medical diagnosis tool.

---

## 🚀 Overview

This project is designed to bridge the gap between symptom-based AI predictions and real clinical evidence.

The system:

* Extracts symptoms from user input
* Generates a cautious AI-based diagnosis
* Retrieves relevant research articles from PubMed
* Retrieves clinical studies from ClinicalTrials.gov
* Summarizes findings using a Large Language Model (LLM)

---

## 🧱 System Architecture

```
User Input → Symptom Extraction → Diagnosis (LLM)
↓
Search Query Builder
↓
PubMed API + ClinicalTrials API
↓
Relevant Articles / Studies
↓
LLM Summarization
↓
Final Structured Output (JSON)
```

---

## ✨ Features

* 🔍 Symptom extraction from natural language
* 🧠 AI-assisted diagnosis (safe & non-final)
* 📚 PubMed research article retrieval
* 🧪 ClinicalTrials study retrieval
* 📝 LLM-based structured summarization
* 📊 Clean JSON output for integration or analysis

---

## 📁 Project Structure

```
AI-healthcare-evidence-platform/
│
├── app.py
├── main.py
├── requirements.txt
├── .env
│
├── functions/
│   ├── symptom_extractor.py
│   ├── diagnosis_symptoms.py
│   ├── pubmed_articles.py
│   ├── clinicaltrials_articles.py
│   ├── summerize_pubmed.py
│   ├── summarize_clinicaltrials.py
│
├── test.ipynb
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-healthcare-evidence-platform.git
cd ai-healthcare-evidence-platform
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Setup environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Running the App

```bash
python app.py
```

App will run on:

```
http://localhost:8080
```

---

## 🧪 API Usage

### Endpoint

```
POST /diag
```

### Sample Input

```json
{
  "description": "I have lower back pain and extreme fatigue for the last 2 weeks"
}
```

---

### Sample Output

```json
{
  "symptom": ["lower back pain", "fatigue"],
  "search_query": "lower back pain OR fatigue",
  "diagnosis": "...",
  "pubmed_articles": [...],
  "pubmed_summary": "...",
  "clinical_trials": [...],
  "clinical_trials_summary": "..."
}
```

---

## 🧠 How It Works

1. **Symptom Extraction**
   Detects structured symptoms from user text

2. **Diagnosis Generation**
   Uses LLM to generate cautious, non-final medical suggestions

3. **Evidence Retrieval**

   * PubMed API → research articles
   * ClinicalTrials API → clinical studies

4. **Summarization**
   LLM summarizes evidence into simple, structured output

---

## 🎯 Use Cases

* Research prototype for AI in healthcare
* Clinical decision-support exploration
* Evidence-based AI systems
* Educational medical tools

---

## ⚠️ Limitations

* Not a real diagnostic system
* No validation dataset yet
* Retrieval may include partially relevant studies
* LLM responses depend on prompt quality

---

## 🚀 Future Improvements

* Add evaluation framework (LLM vs Evidence-based comparison)
* Improve symptom extraction using ML/NLP models
* Add filtering for highly relevant medical evidence
* Integrate vector database → true RAG system
* Build frontend UI (Streamlit / React)

---

## 📌 Tech Stack

* Python
* FastAPI
* OpenAI GPT (LLM)
* PubMed API (NCBI)
* ClinicalTrials.gov API
* BeautifulSoup
* Requests

---

## 👨‍💻 Author

**Shuvo Dutta**
Data Analyst | GenAI Engineer | AI Healthcare Research Enthusiast

---

## 📜 License

This project is for educational and research purposes.

