import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score


from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


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
            data_frames.append(pd.read_csv(file_path, index_col="SEQN"))
        else:
            print(f"Error: File {file_path} not found.")
            return None

    # Combine all the loaded dataframes into one
    data = pd.concat([data_frames[0], data_frames[1], data_frames[2], data_frames[3]], axis=0)

    # Change the HYPERTENSION column to int
    data["HYPERTENSION"] = data["HYPERTENSION"].astype(int)
    data = data.drop(["BPXOSYAVG", "BPXODIAVG"], axis=1)
    return data


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def feature_selection(X_train, y_train):
    # Select top k features
    selector = SelectKBest(f_classif, k=20)  # MODIFY K TO UR DESIRED AMOUNT OF FEATURES
    X_train_reduced = selector.fit_transform(X_train, y_train)
    return X_train_reduced, selector


def train_models(X_train, y_train, X_train_reduced, model_name):
    """Train all models (4+ classifiers)"""

    # Initialize all models
    rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=2, min_samples_leaf=4, random_state=42)
    rfr = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=2, min_samples_leaf=4, random_state=42)
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, min_samples_leaf=1, min_samples_split=10, random_state=42)
    dt = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
    dt_bag = BaggingClassifier(estimator=dt, n_estimators=100, max_samples=0.8, max_features=0.8, oob_score=True, random_state=42)
    lr = LogisticRegression(max_iter=1000, solver='newton-cg', random_state=42)

    rf.fit(X_train, y_train)
    with open(f'{model_name}_rf.model', 'wb') as f:
        pickle.dump(rf, f)

    rfr.fit(X_train_reduced, y_train)
    with open(f'{model_name}_rf_reduced.model', 'wb') as f:
        pickle.dump(rfr, f)

    gbm.fit(X_train, y_train)
    with open(f'{model_name}_gb.model', 'wb') as f:
        pickle.dump(gbm, f)

    dt_bag.fit(X_train, y_train)
    with open(f'{model_name}_dt_bag.model', 'wb') as f:
        pickle.dump(dt_bag, f)

    lr.fit(X_train, y_train)
    with open(f'{model_name}_lr.model', 'wb') as f:
        pickle.dump(lr, f)



    # base_models = [
    #     ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    #     ('svm', SVC(kernel='rbf', probability=True, random_state=42)),  # Use probability=True for stacking
    #     ('neural_network', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
    # ]
    #
    # # Define the meta-model (final estimator)
    # meta_model = LogisticRegression(random_state=42)
    #
    # # Create the Stacking Classifier
    # stacking_clf = StackingClassifier(
    #     estimators=base_models,  # Base models
    #     final_estimator=meta_model,  # Meta-model
    #     cv=5  # Cross-validation for blending
    # )
    #
    # # Fit the Stacking Classifier
    # stacking_clf.fit(X_train, y_train)
    #
    # # Make predictions on the test set
    # y_pred = stacking_clf.predict(X_test)
    #
    # # Evaluate the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Stacking Classifier Accuracy: {accuracy:.2f}")


def load_and_evaluate(model_name, X_test, y_test, selector=None):
    with open(f'{model_name}_rf.model', 'rb') as f:
        rf_full = pickle.load(f)
    score_rf_full = rf_full.score(X_test, y_test)
    print(f"Random Forest Classifier Accuracy: {score_rf_full:.4f}")

    if selector:
        # Apply selector on X_test for reduced feature model evaluation
        X_test_reduced = selector.transform(X_test)
        with open(f'{model_name}_rf_reduced.model', 'rb') as f:
            rf_reduced = pickle.load(f)
        score_rf_reduced = rf_reduced.score(X_test_reduced, y_test)
        print(f"Reduced Random Forest Classifier Accuracy: {score_rf_reduced: .4f}")

    with open(f'{model_name}_gb.model', 'rb') as f:
        gb = pickle.load(f)
    score_gb = gb.score(X_test, y_test)
    print(f"Gradient Boosting Classifier Accuracy: {score_gb:.4f}")

    with open(f'{model_name}_dt_bag.model', 'rb') as f:
        dt_bag = pickle.load(f)
    score_dt_bag = dt_bag.score(X_test, y_test)
    print(f"Bagging Classifier (Decision Tree) Accuracy: {score_dt_bag:.4f}")

    with open(f'{model_name}_lr.model', 'rb') as f:
        lr = pickle.load(f)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)  # Turn regression into 2 classes
    lr_accuracy = accuracy_score(y_test, lr_pred)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

if __name__ == "__main__":
    data = load_data()

    if "HYPERTENSION" in data.columns.to_list():
        X = data.drop("HYPERTENSION", axis=1)
    else:
        print("Column 'HYPERTENSION' not found in data.")

    X = data.drop(["HYPERTENSION"], axis=1)
    y = data["HYPERTENSION"]

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Feature selection for reduced model
    X_train_reduced, selector = feature_selection(X_train, y_train)

    train_models(X_train, y_train, X_train_reduced, model_name="hypertension_predictor")

    load_and_evaluate("hypertension_predictor", X_test, y_test, selector)
