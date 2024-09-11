import json

with open('questions.json', 'r') as f:
    MAPPINGS = json.load(f)

def map_value(value, mappings):
    # Check for exact matches first
    if str(value) in mappings:
        return mappings[str(value)]

    # Check for range matches
    for key, mapped_value in mappings.items():
        if '-' in key:
            min_val, max_val = map(int, key.split('-'))
            try:
                if min_val <= int(value) <= max_val:
                    return value if mapped_value == "Range of Values" else mapped_value
            except ValueError:
                pass  # If value can't be converted to int, skip this check

    return 'Invalid Response'

def apply_mappings(df):
    for category, questions in MAPPINGS.items():
        for question, data in questions.items():
            if question in df.columns:
                df[f'{question}_description'] = df[question].apply(lambda x: map_value(x, data['mappings']))
                df[f'{question}_label'] = data['label']
    return df

def get_question_label(category, question):
    return MAPPINGS[category][question]['label']

def get_question_mappings(category, question):
    return MAPPINGS[category][question]['mappings']