import os
import pandas as pd
import pickle

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def load_data() -> pd.DataFrame:
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed/autofilled"))
    files = [
        "autofilled_data_2007-2008.csv",
        "autofilled_data_2009-2010.csv",
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
    data = pd.concat([data_frames[0], data_frames[1], data_frames[2], data_frames[3], data_frames[4], data_frames[5]], axis=0)

    # Change the HYPERTENSION column to int
    data["HYPERTENSION"] = data["HYPERTENSION"].astype(int)
    return data


def split_data(X, y):
    """Create 80-20 train-test split"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def feature_selection(X_train, y_train):
    """Select top k features"""
    selector = SelectKBest(f_classif, k=20)  # MODIFY K TO UR DESIRED AMOUNT OF FEATURES
    X_train_reduced = selector.fit_transform(X_train, y_train)
    return X_train_reduced, selector


def train_models(X_train, y_train, X_train_reduced, model_name):
    """Train all models (4+ classifiers)"""

    # Initialize all models
    rf = RandomForestClassifier(n_estimators=700, max_depth=25, min_samples_split=2, min_samples_leaf=8, random_state=42)
    rfr = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=10, min_samples_leaf=6, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, min_samples_leaf=1, min_samples_split=10, random_state=42)
    gbr = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, min_samples_leaf=1, min_samples_split=10, random_state=42)
    dt = DecisionTreeClassifier(max_depth=10, min_samples_split=15, min_samples_leaf=4, random_state=42)
    dt_bag = BaggingClassifier(estimator=dt, n_estimators=300, max_samples=0.5, max_features=1.0, bootstrap=True, oob_score=True, random_state=42)
    dt_bag_r = BaggingClassifier(estimator=dt, n_estimators=100, max_samples=0.8, max_features=0.8, oob_score=True, random_state=42)
    lr = LogisticRegression(max_iter=1000, solver='newton-cg', random_state=42)
    lrr = LogisticRegression(max_iter=1000, solver='newton-cg', random_state=42)

    # Train all models
    _train_model(X_train, y_train, model_name + "_rf", rf)
    _train_model(X_train_reduced, y_train, model_name + "_rf_reduced", rfr)
    _train_model(X_train, y_train, model_name + "_gb", gb)
    _train_model(X_train_reduced, y_train, model_name + "_gb_reduced", gbr)
    _train_model(X_train, y_train, model_name + "_dt_bag", dt_bag)
    _train_model(X_train_reduced, y_train, model_name + "_dt_bag_reduced", dt_bag_r)
    _train_model(X_train, y_train, model_name + "_lr", lr)
    _train_model(X_train_reduced, y_train, model_name + "_lr_reduced", lrr)


def _train_model(X_train, y_train, model_name, model):
    """Train and save a single model"""
    model.fit(X_train, y_train)
    with open(f'{model_name}.model', 'wb') as f:
        pickle.dump(model, f)


def load_and_evaluate(model_name, X_test, y_test, selector=None):
    """Load and evaluate all models"""
    score_rf_full = _load_and_evaluate(model_name + '_rf', X_test, y_test)
    print(f"Random Forest Classifier Accuracy: {score_rf_full:.4f}")

    score_gb = _load_and_evaluate(model_name + '_gb', X_test, y_test)
    print(f"Gradient Boosting Classifier Accuracy: {score_gb:.4f}")

    score_dt_bag = _load_and_evaluate(model_name + '_dt_bag', X_test, y_test)
    print(f"Bagging Classifier (Decision Tree) Accuracy: {score_dt_bag:.4f}")

    with open(f'{model_name}_lr.model', 'rb') as f:
        lr = pickle.load(f)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)  # Turn regression into 2 classes
    lr_accuracy = accuracy_score(y_test, lr_pred)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

    if selector:
        # Apply selector on X_test for reduced feature model evaluation
        X_test_reduced = selector.transform(X_test)

        score_rf_reduced = _load_and_evaluate(model_name + '_rf_reduced', X_test_reduced, y_test)
        print(f"Reduced Random Forest Classifier Accuracy: {score_rf_reduced: .4f}")

        score_gb_reduced = _load_and_evaluate(model_name + '_gb_reduced', X_test_reduced, y_test)
        print(f"Reduced Gradient Boosting Classifier Accuracy: {score_gb_reduced:.4f}")

        score_dt_bag_reduced = _load_and_evaluate(model_name + '_dt_bag_reduced', X_test_reduced, y_test)
        print(f"Reduced Bagging Classifier (Decision Tree) Accuracy: {score_dt_bag_reduced:.4f}")

        with open(f'{model_name}_lr_reduced.model', 'rb') as f:
            lr_reduced = pickle.load(f)
        lr_reduced_prob = lr_reduced.predict_proba(X_test_reduced)[:, 1]
        lr_reduced_pred = (lr_reduced_prob >= 0.5).astype(int)  # Turn regression into 2 classes
        lr_reduced_accuracy = accuracy_score(y_test, lr_reduced_pred)
        print(f"Reduced Logistic Regression Accuracy: {lr_reduced_accuracy:.4f}")


def _load_and_evaluate(model_name, X_test, y_test):
    """Load and evaluate single model"""
    with open(f'{model_name}.model', 'rb') as f:
        model = pickle.load(f)
    score = model.score(X_test, y_test)
    return score


def find_best_params(X_train, y_train, X_test, y_test ,X_train_reduced, selector):
    # Create a base estimator (e.g., Decision Tree)
    base_estimator = DecisionTreeClassifier(random_state=42)

    # Create a Bagging Classifier
    bagging_clf = BaggingClassifier(estimator=base_estimator, random_state=42)

    # Define the parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of base estimators
        'max_samples': [0.5, 0.7, 1.0],  # Fraction of samples to draw for each base estimator
        'max_features': [0.5, 0.7, 1.0],  # Fraction of features to draw for each base estimator
        'bootstrap': [True, False],  # Whether to use bootstrap sampling
        'estimator__max_depth': [None, 5, 10, 15],  # Maximum depth for the base Decision Tree
        'estimator__min_samples_split': [5, 10, 15],  # Minimum samples to split in the base Decision Tree
        'estimator__min_samples_leaf': [2, 4, 6, 8]
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=bagging_clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2,
                               n_jobs=-1)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Print the best parameters and the corresponding accuracy
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)

    # Evaluate on the test set
    best_bagging_clf = grid_search.best_estimator_
    y_pred = best_bagging_clf.predict(X_test)
    print("Test Set Accuracy:", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    data = load_data()

    X = data.drop("HYPERTENSION", axis=1)
    y = data["HYPERTENSION"]

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Feature selection for reduced model
    X_train_reduced, selector = feature_selection(X_train, y_train)

    train_models(X_train, y_train, X_train_reduced, model_name="hypertension_predictor")

    load_and_evaluate("hypertension_predictor", X_test, y_test, selector)

    # find_best_params(X_train, y_train, X_test, y_test, X_train_reduced, selector)
