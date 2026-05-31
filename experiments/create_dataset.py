import csv
from pathlib import Path

OUTPUT_PATH = Path("experiments/evaluation_dataset.csv")

def normalize_symptom(symptom: str) -> str:
    symptom = symptom.strip().lower()

    mapping = {
        "shortness of breath": "dyspnea",
        "trouble breathing": "dyspnea",
        "difficulty breathing": "dyspnea",
        "breathlessness": "dyspnea",
        "tired": "fatigue",
        "tiredness": "fatigue",
        "low energy": "fatigue",
        "weak": "weakness",
        "throwing up": "vomiting",
        "swollen legs": "edema",
        "swollen feet": "edema",
        "chest tightness": "chest pain",
        "stomach pain": "abdominal pain",
        "belly pain": "abdominal pain",
        "back pain": "lower back pain",
    }

    normalized = mapping.get(symptom, symptom)
    return normalized

def build_cases():
    cases = []

    acute_cases = [
        ("I have had a fever and headache since last night.", ["fever", "headache"]),
        ("My throat hurts and I have a fever.", ["sore throat", "fever"]),
        ("I feel nauseous and I have been vomiting today.", ["nausea", "vomiting"]),
        ("I have stomach pain with nausea.", ["abdominal pain", "nausea"]),
        ("I woke up with headache and body pain.", ["headache", "joint pain"]),
        ("I have a runny nose, cough, and mild fever.", ["runny nose", "cough", "fever"]),
        ("My eyes are watery and I keep sneezing.", ["watery eyes", "sneezing"]),
        ("I have chills and a high temperature.", ["chills", "fever"]),
        ("I feel dizzy when I stand up.", ["dizziness"]),
        ("I have pain in my lower back after lifting a box.", ["lower back pain"]),
    ]

    chronic_cases = [
        ("My legs are swollen and I feel tired all the time.", ["edema", "fatigue"]),
        ("I feel thirsty often and I urinate frequently.", ["excessive thirst", "frequent urination"]),
        ("I have been very tired and weak for several weeks.", ["fatigue", "weakness"]),
        ("My joints hurt every morning.", ["joint pain"]),
        ("I have trouble sleeping almost every night.", ["insomnia"]),
        ("I feel low mood and have poor sleep.", ["depression", "insomnia"]),
        ("I have ongoing lower back pain.", ["lower back pain"]),
        ("My feet are swollen and I feel short of breath while walking.", ["edema", "dyspnea"]),
        ("I get frequent headaches during work.", ["headache"]),
        ("I have abdominal pain after meals for many weeks.", ["abdominal pain"]),
    ]

    colloquial_cases = [
        ("I just feel off, dizzy, and weak.", ["dizziness", "weakness"]),
        ("My head is pounding and I feel hot.", ["headache", "fever"]),
        ("I feel like throwing up.", ["vomiting"]),
        ("I cannot catch my breath.", ["dyspnea"]),
        ("My chest feels tight.", ["chest pain"]),
        ("I have no energy and feel drained.", ["fatigue"]),
        ("My belly is hurting badly.", ["abdominal pain"]),
        ("I feel down and cannot sleep well.", ["depression", "insomnia"]),
        ("My body aches all over.", ["joint pain"]),
        ("I feel shaky and lightheaded.", ["dizziness", "weakness"]),
    ]

    multi_cases = [
        ("I have chest pain, shortness of breath, and dizziness.", ["chest pain", "dyspnea", "dizziness"]),
        ("I have fever, cough, headache, and fatigue.", ["fever", "cough", "headache", "fatigue"]),
        ("I feel nausea, stomach pain, and weakness.", ["nausea", "abdominal pain", "weakness"]),
        ("I have lower back pain and pain in my joints.", ["lower back pain", "joint pain"]),
        ("I have trouble sleeping, low mood, and fatigue.", ["insomnia", "depression", "fatigue"]),
        ("I have fever, chills, and vomiting.", ["fever", "chills", "vomiting"]),
        ("My legs are swollen, and I feel chest tightness.", ["edema", "chest pain"]),
        ("I have headache, dizziness, and nausea.", ["headache", "dizziness", "nausea"]),
        ("I feel tired, weak, and short of breath.", ["fatigue", "weakness", "dyspnea"]),
        ("I have abdominal pain, nausea, and poor sleep.", ["abdominal pain", "nausea", "insomnia"]),
    ]

    red_flag_cases = [
        ("I have severe chest pain and shortness of breath.", ["chest pain", "dyspnea"]),
        ("I have sudden weakness on one side of my body.", ["weakness"]),
        ("I have chest tightness with sweating and dizziness.", ["chest pain", "sweating", "dizziness"]),
        ("I have a very high fever and confusion.", ["fever", "confusion"]),
        ("I have severe abdominal pain with vomiting.", ["abdominal pain", "vomiting"]),
        ("I cannot breathe properly and my chest hurts.", ["dyspnea", "chest pain"]),
        ("I have a terrible headache and blurry vision.", ["headache", "blurred vision"]),
        ("I have fainting and chest pain.", ["fainting", "chest pain"]),
        ("I have severe pelvic pain and dizziness.", ["pelvic pain", "dizziness"]),
        ("I feel extreme weakness and cannot stand.", ["weakness"]),
    ]

    negative_cases = [
        ("I want to know about healthy food.", []),
        ("Can you explain what vitamin C does?", []),
        ("I need general fitness advice.", []),
        ("What is a balanced diet?", []),
        ("Tell me about medical insurance.", []),
        ("How can I sleep better in general?", ["insomnia"]),
        ("I want information about exercise plans.", []),
        ("Can you summarize a health article?", []),
        ("What is PubMed?", []),
        ("I am researching clinical trials.", []),
    ]

    category_blocks = [
        ("Common acute symptoms", acute_cases, 40),
        ("Chronic-disease-related symptoms", chronic_cases, 35),
        ("Colloquial symptom descriptions", colloquial_cases, 35),
        ("Multi-symptom descriptions", multi_cases, 45),
        ("Red-flag-like presentations", red_flag_cases, 25),
        ("Negative or irrelevant inputs", negative_cases, 20),
    ]

    case_id = 1

    for category, templates, target_count in category_blocks:
        index = 0

        while index < target_count:
            template_text, symptoms = templates[index % len(templates)]

            if index < len(templates):
                symptom_description = template_text
            else:
                variation_number = index // len(templates)

                if variation_number == 1:
                    symptom_description = "Since yesterday, " + template_text[0].lower() + template_text[1:]
                elif variation_number == 2:
                    symptom_description = template_text.replace("I have", "I am experiencing")
                elif variation_number == 3:
                    symptom_description = template_text.replace("I feel", "Recently I feel")
                else:
                    symptom_description = template_text + " It has been bothering me."

            normalized_symptoms = []
            for symptom in symptoms:
                normalized_symptom = normalize_symptom(symptom)

                if normalized_symptom not in normalized_symptoms:
                    normalized_symptoms.append(normalized_symptom)

            case = {
                "case_id": f"{case_id:03d}",
                "symptom_description": symptom_description,
                "category": category,
                "gold_standard_symptoms": "; ".join(normalized_symptoms) if normalized_symptoms else "none",
            }

            cases.append(case)
            case_id += 1
            index += 1

    return cases

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cases = build_cases()

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "case_id",
                "symptom_description",
                "category",
                "gold_standard_symptoms",
            ],
        )
        writer.writeheader()
        writer.writerows(cases)

    print(f"Created dataset: {OUTPUT_PATH}")
    print(f"Total cases: {len(cases)}")

if __name__ == "__main__":
    main()
