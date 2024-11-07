# ... --- ... (Morse Code)
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Example ML model
from sklearn.feature_selection import SelectKBest, f_classif
import pickle


def load_data() -> pd.DataFrame:
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed/autofilled"))
    files = [
        "autofilled_data_2011-2012.csv",
        "autofilled_data_2013-2014.csv",
        "autofilled_data_2015-2016.csv",
        "autofilled_data_2017-2020.csv"
    ]

    data_frames = []
    for file in files:
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            print(f"Loading {file_path}")
            data_frames.append(pd.read_csv(file_path))
        else:
            print(f"Error: File {file_path} not found.")
            return None

    data = pd.concat([data_frames[0], data_frames[1], data_frames[2], data_frames[3]], axis=0)
    data.dropna(subset=["BPXOSYAVG"], inplace=True)
    data.dropna(subset=["BPXODIAVG"], inplace=True)
    return data


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42)
    return X_train, X_test, y_train, y_test


def feature_selection(X_train, y_train):
    # Select top k features
    selector = SelectKBest(f_classif, k=10)  # MODIFY K TO UR DESIRED AMOUNT OF FEATURES
    X_train_reduced = selector.fit_transform(X_train, y_train)
    return X_train_reduced, selector


def train_model(X_train, y_train, X_train_reduced, model_name):
    # Full Feature Model
    model_full = RandomForestClassifier(random_state=42)
    model_full.fit(X_train, y_train)

    with open(f'{model_name}_full.model', 'wb') as f:
        pickle.dump(model_full, f)

    # Reduced Feature Model
    model_reduced = RandomForestClassifier(random_state=42)
    model_reduced.fit(X_train_reduced, y_train)

    with open(f'{model_name}_reduced.model', 'wb') as f:
        pickle.dump(model_reduced, f)


def load_and_evaluate(model_name, X_test, y_test, selector=None):
    with open(f'{model_name}_full.model', 'rb') as f:
        model_full = pickle.load(f)
    score_full = model_full.score(X_test, y_test)
    print(f"Full model accuracy: {score_full}")

    if selector:
        # Apply selector on X_test for reduced feature model evaluation
        X_test_reduced = selector.transform(X_test)
        with open(f'{model_name}_reduced.model', 'rb') as f:
            model_reduced = pickle.load(f)
        score_reduced = model_reduced.score(X_test_reduced, y_test)
        print(f"Reduced model accuracy: {score_reduced}")


if __name__ == "__main__":
    data = load_data()

    if "BPXOSYAVG" in data.columns.to_list():
        X = data.drop("BPXOSYAVG", axis=1)
    else:
        print("Column 'BPXOSYAVG' not found in data.")

    X = data.drop("BPXOSYAVG", axis=1)  # Replace "temp_name_column" with actual name of blood pressure column later
    y = data["BPXOSYAVG"]

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Feature selection for reduced model
    X_train_reduced, selector = feature_selection(X_train, y_train)

    train_model(X_train, y_train, X_train_reduced, model_name="hypertension_predictor")

    load_and_evaluate("hypertension_predictor", X_test, y_test, selector)
