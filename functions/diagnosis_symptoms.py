import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_diagnosis(symptoms: list[str]) -> str:
    if not symptoms:
        return "No clear symptoms were detected from the input."

    symptom_text = ", ".join(symptoms)

    prompt = f"""
Patient symptoms: {symptom_text}

Give a short and careful response in simple language.

Rules:
1. Do NOT claim a final diagnosis.
2. Give 3 possible conditions only if they reasonably match the symptoms.
3. Keep it concise.
4. Mention that a real doctor is needed for confirmation.
5. Do NOT give extreme diseases unless strongly supported.
6. Add a short general care suggestion.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a careful medical assistant for educational use only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    final_text = response.choices[0].message.content
    return final_text.strip()