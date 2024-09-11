import json

with open('questions.json', 'r') as f:
    MAPPINGS = json.load(f)

def map_value(value, mappings):
    # Check if value is None or empty
    if value is None or value == '':
        return 'No Response'

    # Check for exact matches first
    if str(value) in mappings:
        return mappings[str(value)].get('description', 'Description Not Available')

    # Check for range matches
    for key, mapped_value in mappings.items():
        if '-' in key:
            try:
                min_val, max_val = map(int, key.split('-'))
                if min_val <= int(value) <= max_val:
                    return value if mapped_value.get('description') == "Range of Values" else mapped_value.get('description', 'Description Not Available')
            except ValueError:
                pass  # If value can't be converted to int, skip this check

    return 'Invalid Response'

def apply_mappings(df):
    for category, questions in MAPPINGS.items():
        for question, data in questions.items():
            if question in df.columns:
                if 'mappings' in data:
                    df[f'{question}_description'] = df[question].apply(lambda x: map_value(x, data['mappings']))
                else:
                    df[f'{question}_description'] = 'Mappings Not Available'
                df[f'{question}_label'] = data.get('label', 'Label Not Available')
    return df

def get_question_label(category, question):
    try:
        return MAPPINGS[category][question]['label']
    except KeyError:
        return 'Label Not Available'

def get_question_mappings(category, question):
    try:
        return MAPPINGS[category][question]['mappings']
    except KeyError:
        return {}

def get_question_populate(category, question, value):
    try:
        return MAPPINGS[category][question]['mappings'][str(value)].get('populate', {})
    except KeyError:
        return {}