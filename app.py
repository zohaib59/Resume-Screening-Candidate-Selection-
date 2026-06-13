# STEP 1: IMPORTS
# =====================================================
import os
import gc
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# STEP 2: SETTINGS
# =====================================================
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

DATA_PATH = "resume.csv"   # change dataset path
TARGET_COL = "selected"                    # change target column

TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# STEP 3: LOAD DATA
# =====================================================
print("\nLoading Dataset...")

data = pd.read_csv(DATA_PATH)

# Remove duplicates
data = data.drop_duplicates().reset_index(drop=True)

# Check target
if TARGET_COL not in data.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found")

print("Dataset Shape:", data.shape)

# STEP 4: HANDLE MISSING VALUES
# =====================================================
# Replace common missing symbols
data.replace(["?", "NA", "N/A", "null", "None"], np.nan, inplace=True)

# STEP 5: SPLIT FEATURES & TARGET
# =====================================================
X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]

# STEP 6: ENCODE TARGET
# =====================================================
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))

joblib.dump(
    target_encoder,
    os.path.join(MODEL_DIR, "target_encoder.joblib")
)


# STEP 7: ENCODE ONLY CATEGORICAL FEATURES
# =====================================================
label_encoders = {}

categorical_cols = X.select_dtypes(include=["object"]).columns

for col in categorical_cols:

    le = LabelEncoder()

    X[col] = le.fit_transform(X[col].astype(str))

    label_encoders[col] = le

joblib.dump(
    label_encoders,
    os.path.join(MODEL_DIR, "feature_encoders.joblib")
)

# STEP 8: TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# STEP 9: PREPROCESSING PIPELINES
# =====================================================

# For Logistic Regression
linear_preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# For Tree Models
tree_preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# STEP 10: MODELS
# =====================================================
models = {

    "LogisticRegression": {

        "model": LogisticRegression(
            max_iter=500,
            solver="lbfgs"
        ),

        "preprocess": linear_preprocess
    },

    "RandomForest": {

        "model": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

        "preprocess": tree_preprocess
    },

    "ExtraTrees": {

        "model": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

        "preprocess": tree_preprocess
    },

    "XGBoost": {

        "model": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

        "preprocess": tree_preprocess
    }
}

# STEP 11: TRAINING & EVALUATION
# =====================================================
results = []

best_model_name = None
best_f1 = -1

print("\nStarting Training...\n")

for name, config in models.items():

    print("=" * 80)
    print("MODEL:", name)

    try:

        pipeline = Pipeline([
            ("preprocess", config["preprocess"]),
            ("model", config["model"])
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_f1 = f1_score(
            y_train,
            y_train_pred,
            average="macro"
        )

        test_f1 = f1_score(
            y_test,
            y_test_pred,
            average="macro"
        )

        print(f"Train Accuracy : {train_acc:.4f}")
        print(f"Test Accuracy  : {test_acc:.4f}")

        print(f"Train F1 Score : {train_f1:.4f}")
        print(f"Test F1 Score  : {test_f1:.4f}")

        print("\nClassification Report:\n")
        print(classification_report(y_test, y_test_pred))

        # Save model
        model_path = os.path.join(
            MODEL_DIR,
            f"{name}.joblib"
        )

        joblib.dump(pipeline, model_path)

        # Save results
        results.append({
            "Model": name,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Train F1": train_f1,
            "Test F1": test_f1
        })

        # Best model tracking
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model_name = name

    except Exception as e:

        print("Error:", e)

    finally:
        gc.collect()

# STEP 12: RESULTS TABLE
# =====================================================
results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(results_df.sort_values(
    by="Test F1",
    ascending=False
))

print("\nBest Model:", best_model_name)

# STEP 13: SHAP ANALYSIS
# =====================================================
USE_SHAP = True

if USE_SHAP:

    print("\n" + "=" * 80)
    print("RUNNING SHAP ANALYSIS")
    print("=" * 80)

    try:

        import shap

        # Load best model
        pipeline = joblib.load(
            os.path.join(
                MODEL_DIR,
                f"{best_model_name}.joblib"
            )
        )

        # Sample data
        X_sample = X_test.iloc[:300]

        # Preprocess
        X_transformed = pipeline.named_steps[
            "preprocess"
        ].transform(X_sample)

        model = pipeline.named_steps["model"]

        feature_names = X.columns.tolist()

        # ------------------------------------------------
        # SHAP EXPLAINER
        # ------------------------------------------------
        if best_model_name in [
            "RandomForest",
            "ExtraTrees",
            "XGBoost"
        ]:

            explainer = shap.TreeExplainer(model)

        else:

            explainer = shap.Explainer(
                model,
                X_transformed
            )

        shap_values = explainer.shap_values(
            X_transformed
        )

        # ------------------------------------------------
        # UNIVERSAL MULTICLASS HANDLING
        # ------------------------------------------------
        if isinstance(shap_values, list):

            # OLD SHAP FORMAT
            shap_array = np.mean(
                np.abs(np.array(shap_values)),
                axis=0
            )

        elif len(np.array(shap_values).shape) == 3:

            # NEW MULTICLASS FORMAT
            shap_array = np.mean(
                np.abs(shap_values),
                axis=2
            )

        else:

            # BINARY CLASSIFICATION
            shap_array = shap_values

        # ------------------------------------------------
        # SHAP SUMMARY PLOT
        # ------------------------------------------------
        print("\nGenerating SHAP Summary Plot...")

        shap.summary_plot(
            shap_array,
            X_transformed,
            feature_names=feature_names
        )

        plt.show()

        # =================================================
        # FEATURE IMPORTANCE
        # =================================================
        print("\n" + "=" * 80)
        print("TOP FEATURE IMPORTANCE")
        print("=" * 80)

        importance = np.abs(
            shap_array
        ).mean(axis=0)

        TOP_N = min(10, len(feature_names))

        top_indices = np.argsort(
            importance
        )[-TOP_N:][::-1]

        for rank, i in enumerate(
            top_indices,
            start=1
        ):

            feature = feature_names[i]

            print(
                f"{rank}. {feature} "
                f"→ Importance: {importance[i]:.5f}"
            )

    except Exception as e:

        print("\nSHAP ERROR:", e)

# STEP 14: COMPLETE
# =====================================================
print("\n" + "=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)






















