import json

QUESTIONS_FILE = 'questions.json'
NO_RESPONSE = 'No Response'
DESCRIPTION_NOT_AVAILABLE = 'Description Not Available'
INVALID_RESPONSE = 'Invalid Response'
RANGE_OF_VALUES = "Range of Values"
MAPPINGS_NOT_AVAILABLE = 'Mappings Not Available'
LABEL_NOT_AVAILABLE = 'Label Not Available'
DESCRIPTION_KEY = 'description'
LABEL_KEY = 'label'
MAPPINGS_KEY = 'mappings'
POPULATE_KEY = 'populate'

with open(QUESTIONS_FILE, 'r') as f:
    MAPPINGS = json.load(f)

def map_value(value, mappings):
    # Check if value is None or empty
    if value is None or value == '':
        return NO_RESPONSE

    # Check for exact matches first
    if str(value) in mappings:
        return mappings[str(value)].get(DESCRIPTION_KEY, DESCRIPTION_NOT_AVAILABLE)

    # Check for range matches
    for key, mapped_value in mappings.items():
        if '-' in key:
            try:
                min_val, max_val = map(int, key.split('-'))
                if min_val <= int(value) <= max_val:
                    return value if mapped_value.get(DESCRIPTION_KEY) == RANGE_OF_VALUES else mapped_value.get(DESCRIPTION_KEY, DESCRIPTION_NOT_AVAILABLE)
            except ValueError:
                pass  # If value can't be converted to int, skip this check

    return INVALID_RESPONSE

def apply_mappings(df):
    for category, questions in MAPPINGS.items():
        for question, data in questions.items():
            if question in df.columns:
                if MAPPINGS_KEY in data:
                    df[f'{question}_description'] = df[question].apply(lambda x: map_value(x, data[MAPPINGS_KEY]))
                else:
                    df[f'{question}_description'] = MAPPINGS_NOT_AVAILABLE
                df[f'{question}_label'] = data.get(LABEL_KEY, LABEL_NOT_AVAILABLE)
    return df

def get_question_label(category, question):
    try:
        return MAPPINGS[category][question][LABEL_KEY]
    except KeyError:
        return LABEL_NOT_AVAILABLE

def get_question_mappings(category, question):
    try:
        return MAPPINGS[category][question][MAPPINGS_KEY]
    except KeyError:
        return {}

def get_question_populate(category, question, value):
    try:
        return MAPPINGS[category][question][MAPPINGS_KEY][str(value)].get(POPULATE_KEY, {})
    except KeyError:
        return {}