import re


def extract_symptoms(text: str) -> list[str]:
    input_text = text.lower()

    symptom_patterns = {
        "lower back pain": [
            r"\blower back pain\b",
            r"\bpain in (my )?lower back\b",
            r"\bback pain\b"
        ],
        "joint pain": [
            r"\bjoint pain\b",
            r"\bpain in (my )?joints\b",
            r"\bbody ache\b",
            r"\bbody pain\b"
        ],
        "chest pain": [
            r"\bchest pain\b",
            r"\bpain in (my )?chest\b"
        ],
        "abdominal pain": [
            r"\babdominal pain\b",
            r"\bstomach pain\b",
            r"\bbelly pain\b"
        ],
        "pelvic pain": [
            r"\bpelvic pain\b"
        ],
        "headache": [
            r"\bheadache\b",
            r"\bhead pain\b"
        ],
        "fever": [
            r"\bfever\b",
            r"\bhigh temperature\b"
        ],
        "nausea": [
            r"\bnausea\b",
            r"\bnauseous\b",
            r"\bvomiting\b",
            r"\bfeel like vomiting\b"
        ],
        "fatigue": [
            r"\bfatigue\b",
            r"\btired\b",
            r"\bextreme tiredness\b",
            r"\blow energy\b",
            r"\bweakness\b"
        ],
        "insomnia": [
            r"\binsomnia\b",
            r"\btrouble sleeping\b",
            r"\bcannot sleep\b",
            r"\bcan't sleep\b",
            r"\bpoor sleep\b"
        ],
        "depression": [
            r"\bdepression\b",
            r"\bdepressed\b",
            r"\blow mood\b"
        ]
    }

    detected_symptoms = []

    for symptom_name, pattern_list in symptom_patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, input_text):
                detected_symptoms.append(symptom_name)
                break

    unique_symptoms = []
    for symptom_name in detected_symptoms:
        if symptom_name not in unique_symptoms:
            unique_symptoms.append(symptom_name)

    if unique_symptoms:
        return unique_symptoms

    fallback_matches = re.findall(
        r"\b(headache|fever|nausea|fatigue|pain|weakness|insomnia|depression)\b",
        input_text
    )

    fallback_unique = []
    for item in fallback_matches:
        if item not in fallback_unique:
            fallback_unique.append(item)

    return fallback_unique