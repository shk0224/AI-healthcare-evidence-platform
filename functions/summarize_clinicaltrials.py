import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_clinical_trials(text: str) -> str:
    prompt = f"""
Summarize the following clinical trial information in simple language.

Rules:
1. Cover all trials briefly.
2. Only use the information provided in the text.
3. If status or intervention is missing, say "Not clearly stated".
4. Do not assume a trial is ongoing unless explicitly stated.
5. Keep the output concise and structured.

For each trial, mention:
- study title
- condition
- purpose
- intervention
- status

Text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a careful clinical trial summarizer."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    final_text = response.choices[0].message.content
    return final_text.strip()