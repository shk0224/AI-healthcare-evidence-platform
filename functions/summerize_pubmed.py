import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_text(text: str) -> str:
    prompt = f"""
Summarize the following PubMed research articles in simple medical language.

Rules:
1. Cover all articles briefly.
2. Only use the provided text.
3. Do not invent missing facts.
4. Keep the summary concise and structured.
5. Mention article title and its main finding.

Text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a careful medical research summarizer."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    final_text = response.choices[0].message.content
    return final_text.strip()