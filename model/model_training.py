import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle


def load_data() -> pd.DataFrame:
    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/processed/autofilled")
    )
    files = [
        "AutoFilled_Data_2007-2008.csv",
        "AutoFilled_Data_2009-2010.csv",
        "AutoFilled_Data_2011-2012.csv",
        "AutoFilled_Data_2013-2014.csv",
        "AutoFilled_Data_2015-2016.csv",
        "AutoFilled_Data_2017-2020.csv",
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
    data = pd.concat(
        [
            data_frames[0],
            data_frames[1],
            data_frames[2],
            data_frames[3],
            data_frames[4],
            data_frames[5],
        ],
        axis=0,
    )

    # Change the HYPERTENSION column to int
    data["HYPERTENSION"] = data["HYPERTENSION"].astype(int)

    return shuffle(data)


def split_data(X, y):
    """Create 80-20 train-test split"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def feature_selection(X_train, y_train):
    """Select top k features"""
    selector = SelectKBest(f_classif, k=20)  # MODIFY K TO UR DESIRED AMOUNT OF FEATURES
    X_train_reduced = selector.fit_transform(X_train, y_train)
    return X_train_reduced, selector


def train_models(X_train, y_train, X_train_reduced, model_name):
    """Train all models (4+ classifiers)"""

    # Initialize all models
    rf = RandomForestClassifier(
        n_estimators=700,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=8,
        random_state=42,
    )
    rfr = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=6,
        random_state=42,
    )
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=1,
        min_samples_split=10,
        random_state=42,
    )
    gbr = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=1,
        min_samples_split=10,
        random_state=42,
    )
    dt = DecisionTreeClassifier(
        max_depth=10, min_samples_split=15, min_samples_leaf=4, random_state=42
    )
    dt_bag = BaggingClassifier(
        estimator=dt,
        n_estimators=300,
        max_samples=0.5,
        max_features=1.0,
        bootstrap=True,
        oob_score=True,
        random_state=42,
    )
    dt_bag_r = BaggingClassifier(
        estimator=dt,
        n_estimators=100,
        max_samples=0.8,
        max_features=0.8,
        oob_score=True,
        random_state=42,
    )
    lr = LogisticRegression(max_iter=1000, solver="newton-cg", random_state=42)
    lrr = LogisticRegression(max_iter=1000, solver="newton-cg", random_state=42)

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
    with open(f"{model_name}.model", "wb") as f:
        pickle.dump(model, f)


def load_and_evaluate(model_name, X_test, y_test, selector=None):
    """Load and evaluate all models"""
    _load_and_evaluate(model_name + "_rf", X_test, y_test)
    _load_and_evaluate(model_name + "_gb", X_test, y_test)
    _load_and_evaluate(model_name + "_dt_bag", X_test, y_test)
    _load_and_evaluate(model_name + "_lr", X_test, y_test)

    if selector:
        # Apply selector on X_test for reduced feature model evaluation
        X_test_reduced = selector.transform(X_test)

        _load_and_evaluate(model_name + "_rf_reduced", X_test_reduced, y_test)
        _load_and_evaluate(model_name + "_gb_reduced", X_test_reduced, y_test)
        _load_and_evaluate(model_name + "_dt_bag_reduced", X_test_reduced, y_test)
        _load_and_evaluate(model_name + "_lr_reduced", X_test_reduced, y_test)


def _load_and_evaluate(model_name, X_test, y_test):
    """Load and evaluate single model with detailed visualization"""
    with open(f"{model_name}.model", "rb") as f:
        model = pickle.load(f)

    if "lr" in model_name:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"{model_name} Accuracy: {accuracy}")
    print(f"{model_name} Precision: {precision}")
    print(f"{model_name} Recall: {recall}")
    print(f"{model_name} F1 Score: {f1}")
    print(f"{model_name} ROC AUC: {roc_auc}")
    print(f"{model_name} Confusion Matrix: \n", conf_matrix, "\n")

    eval_dir = "evaluation_plots"
    model_dir = os.path.join(eval_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(
        os.path.join(model_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    plt.figure(figsize=(10, 8))
    plt.plot(
        recall_curve,
        precision_curve,
        color="darkgreen",
        lw=2,
        label=f"PR curve (AUC = {pr_auc:.2f})",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(
        os.path.join(model_dir, "precision_recall_curve.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def find_best_params(X_train, y_train, X_test, y_test, X_train_reduced, selector):
    params = {
        "n_estimators": [500, 700, 1000],
        "max_depth": [None, 20, 25],
        "min_samples_split": [2, 4, 8, 10, 12],
        "min_samples_leaf": [4, 6, 8, 10],
        "bootstrap": [True, False],
    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator=rf, param_grid=params, cv=5, scoring="f1", verbose=3, n_jobs=-1
    )
    grid_search.fit(X_train_reduced, y_train)

    # Print the best parameters and the corresponding accuracy
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)
    X_test_reduced = selector.transform(X_test)
    # Evaluate on the test set
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test_reduced)
    print("Test Set Accuracy:", accuracy_score(y_test, y_pred))

    # # Create a base estimator (e.g., Decision Tree)
    # base_estimator = DecisionTreeClassifier(random_state=42)
    #
    # # Create a Bagging Classifier
    # bagging_clf = BaggingClassifier(estimator=base_estimator, random_state=42)
    #
    # # Define the parameter grid for tuning
    # param_grid = {
    #     'n_estimators': [50, 100, 200],  # Number of base estimators
    #     'max_samples': [0.5, 0.7, 1.0],  # Fraction of samples to draw for each base estimator
    #     'max_features': [0.5, 0.7, 1.0],  # Fraction of features to draw for each base estimator
    #     'bootstrap': [True, False],  # Whether to use bootstrap sampling
    #     'estimator__max_depth': [None, 5, 10, 15],  # Maximum depth for the base Decision Tree
    #     'estimator__min_samples_split': [5, 10, 15],  # Minimum samples to split in the base Decision Tree
    #     'estimator__min_samples_leaf': [2, 4, 6, 8]
    # }
    #
    # # Set up GridSearchCV
    # grid_search = GridSearchCV(estimator=bagging_clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1,
    #                            n_jobs=-1)
    #
    # # Fit the grid search to the training data
    # grid_search.fit(X_train_reduced, y_train)
    #
    # # Print the best parameters and the corresponding accuracy
    # print("Best Parameters:", grid_search.best_params_)
    # print("Best Cross-Validation Accuracy:", grid_search.best_score_)
    #
    # X_test_reduced = selector.transform(X_test)
    #
    # # Evaluate on the test set
    # best_bagging_clf = grid_search.best_estimator_
    # y_pred = best_bagging_clf.predict(X_test_reduced)
    # print("Test Set Accuracy:", accuracy_score(y_test, y_pred))


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
