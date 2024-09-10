import json
import pandas as pd

with open('questions.json', 'r') as f:
    MAPPINGS = json.load(f)

def apply_mappings(df):
    for category, questions in MAPPINGS.items():
        for question, data in questions.items():
            if question in df.columns:
                df[f'{question}_description'] = df[question].astype(str).map(data['mappings'])
                df[f'{question}_label'] = data['label']
    return df

def get_question_label(category, question):
    return MAPPINGS[category][question]['label']

def get_question_mappings(category, question):
    return MAPPINGS[category][question]['mappings']