import glob
import os
import pickle
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import KMeansSMOTE
from imblearn.pipeline import Pipeline
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_data() -> pd.DataFrame:
    """Load and combine all autofilled data files."""
    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/processed/autofilled")
    )
    pattern = os.path.join(base_path, "AutoFilled_Data_*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"Error: No files found matching pattern: {pattern}")
        return None

    data_frames = []
    for file in sorted(files):
        if os.path.exists(file):
            data_frames.append(pd.read_csv(file, index_col="SEQN"))
        else:
            print(f"Error: File {file} not found.")
            return None

    # Combine all the loaded dataframes into one
    data = pd.concat(data_frames, axis=0)

    # Change the HYPERTENSION column to int
    data["HYPERTENSION"] = data["HYPERTENSION"].astype(bool)
    return data


def split_data(X, y):
    """Create 80-20 train-test split"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
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
        n_estimators=750,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=8,
        bootstrap=True,
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
    _train_model(
        X_train,
        y_train,
        model_name + "_rf",
        rf,
        {"k_neighbors": 3, "sampling_strategy": 1.0, "cluster_balance_threshold": 0.4},
    )
    _train_model(
        X_train_reduced,
        y_train,
        model_name + "_rf_reduced",
        rfr,
        {"k_neighbors": 3, "sampling_strategy": 1.0, "cluster_balance_threshold": 0.4},
    )
    _train_model(
        X_train,
        y_train,
        model_name + "_gb",
        gb,
        {"k_neighbors": 3, "sampling_strategy": 1.0, "cluster_balance_threshold": 0.3},
    )
    _train_model(
        X_train_reduced,
        y_train,
        model_name + "_gb_reduced",
        gbr,
        {"k_neighbors": 3, "sampling_strategy": 1.0, "cluster_balance_threshold": 0.3},
    )
    _train_model(
        X_train,
        y_train,
        model_name + "_dt_bag",
        dt_bag,
        {"k_neighbors": 3, "sampling_strategy": 1.0, "cluster_balance_threshold": 0.3},
    )
    _train_model(
        X_train_reduced,
        y_train,
        model_name + "_dt_bag_reduced",
        dt_bag_r,
        {"k_neighbors": 3, "sampling_strategy": 1.0, "cluster_balance_threshold": 0.3},
    )
    _train_model(
        X_train,
        y_train,
        model_name + "_lr",
        lr,
        {"k_neighbors": 3, "sampling_strategy": 1.0, "cluster_balance_threshold": 0.3},
    )
    _train_model(
        X_train_reduced,
        y_train,
        model_name + "_lr_reduced",
        lrr,
        {"k_neighbors": 3, "sampling_strategy": 1.0, "cluster_balance_threshold": 0.3},
    )


def _train_model(X_train, y_train, model_name, model, smote_params):
    """Train and save a single model"""
    # Define pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "smote",
                KMeansSMOTE(
                    random_state=42,
                    k_neighbors=smote_params["k_neighbors"],
                    sampling_strategy=smote_params["sampling_strategy"],
                    cluster_balance_threshold=smote_params["cluster_balance_threshold"],
                ),
            ),
            ("classifier", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    with open(f"{model_name}.model", "wb") as f:
        pickle.dump(pipeline, f)


def load_and_evaluate(model_name, X_test, y_test, selector=None):
    """Load and evaluate all models"""
    # Create lists to store results for combined plots
    models_data = []

    # Evaluate full-feature models
    models_data.extend(
        [
            (
                model_name + "_rf",
                *_load_and_evaluate(model_name + "_rf", X_test, y_test),
            ),
            (
                model_name + "_gb",
                *_load_and_evaluate(model_name + "_gb", X_test, y_test),
            ),
            (
                model_name + "_dt_bag",
                *_load_and_evaluate(model_name + "_dt_bag", X_test, y_test),
            ),
            (
                model_name + "_lr",
                *_load_and_evaluate(model_name + "_lr", X_test, y_test),
            ),
        ]
    )

    if selector:
        # Apply selector on X_test for reduced feature model evaluation
        X_test_reduced = selector.transform(X_test)

        # Evaluate reduced-feature models
        models_data.extend(
            [
                (
                    model_name + "_rf_reduced",
                    *_load_and_evaluate(
                        model_name + "_rf_reduced", X_test_reduced, y_test
                    ),
                ),
                (
                    model_name + "_gb_reduced",
                    *_load_and_evaluate(
                        model_name + "_gb_reduced", X_test_reduced, y_test
                    ),
                ),
                (
                    model_name + "_dt_bag_reduced",
                    *_load_and_evaluate(
                        model_name + "_dt_bag_reduced", X_test_reduced, y_test
                    ),
                ),
                (
                    model_name + "_lr_reduced",
                    *_load_and_evaluate(
                        model_name + "_lr_reduced", X_test_reduced, y_test
                    ),
                ),
            ]
        )

    # Plot combined ROC curves
    _plot_combined_roc_curves(models_data, y_test)

    # Plot combined confusion matrices
    _plot_combined_confusion_matrices(models_data)


def _plot_combined_roc_curves(models_data, y_test):
    """Plot ROC curves for all models in a single figure"""
    plt.figure(figsize=(12, 8))
    colors = cycle(
        [
            "aqua",
            "darkorange",
            "cornflowerblue",
            "red",
            "green",
            "purple",
            "brown",
            "pink",
        ]
    )

    for model_name, _, _, _, _, roc_auc, _, y_prob in models_data:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(
            fpr,
            tpr,
            color=next(colors),
            lw=2,
            label=f"{model_name} (AUC = {roc_auc:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - All Models")
    plt.legend(loc="lower right", bbox_to_anchor=(1.45, 0))
    plt.tight_layout()
    plt.savefig(
        "evaluation_plots/combined_roc_curves.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def _plot_combined_confusion_matrices(models_data):
    """Plot confusion matrices for all models in a single figure"""
    n_models = len(models_data)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for idx, (model_name, _, _, _, _, _, conf_matrix, _) in enumerate(models_data):
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=axes[idx],
        )
        axes[idx].set_title(f"Confusion Matrix - {model_name}")
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xlabel("Predicted Label")

    # Remove empty subplots if any
    for idx in range(len(models_data), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(
        "evaluation_plots/combined_confusion_matrices.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def _load_and_evaluate(model_name, X_test, y_test):
    """Load and evaluate single model"""
    with open(f"{model_name}.model", "rb") as f:
        model = pickle.load(f)

    if "lr" in model_name:
        # Turn probability into 2 classes
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        accuracy = model.score(X_test, y_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Precision: {precision:.4f}")
    print(f"{model_name} Recall: {recall:.4f}")
    print(f"{model_name} F1 Score: {f1:.4f}")
    print(f"{model_name} ROC AUC: {roc_auc:.4f}")
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

    return [accuracy, precision, recall, f1, roc_auc, conf_matrix, y_prob]


if __name__ == "__main__":
    data = load_data()

    X = data.drop("HYPERTENSION", axis=1)
    y = data["HYPERTENSION"]

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Feature selection for reduced model
    X_train_reduced, selector = feature_selection(X_train, y_train)

    train_models(X_train, y_train, X_train_reduced, model_name="hypertension_predictor")

    load_and_evaluate("hypertension_predictor", X_test, y_test, selector)
