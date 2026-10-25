
from flask import Flask, render_template, request
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "credit_data.csv"

model = None
feature_columns = []
score_accuracy = None
error_message = None


def train_model():
    global model, feature_columns, score_accuracy, error_message

    if not DATA_FILE.exists():
        error_message = (
            "Dataset not found. Please place credit_data.csv "
            "inside the Credit_Score_Prediction_New folder."
        )
        return

    try:
        data = pd.read_csv(DATA_FILE)
        data.columns = data.columns.str.strip()

        target = "Credit_Score"
        if target not in data.columns:
            error_message = "The CSV must contain a Credit_Score column."
            return

        data = data.dropna(subset=[target])
        X = data.drop(columns=[target]).copy()
        y = data[target].astype(str).str.strip()

        # Remove identifiers and columns that cannot be used as inputs.
        excluded = [
            col for col in X.columns
            if col.strip().lower() in {
                "id", "customer_id", "name", "ssn"
            }
        ]
        X = X.drop(columns=excluded, errors="coerce")

        if X.empty or len(data) < 5 or y.nunique() < 2:
            error_message = (
                "The dataset needs at least five usable rows "
                "and at least two Credit_Score categories."
            )
            return

        feature_columns = list(X.columns)

        numeric_columns = X.select_dtypes(
            include=["number"]
        ).columns.tolist()
        categorical_columns = [
            col for col in X.columns if col not in numeric_columns
        ]

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers = []
        if numeric_columns:
            transformers.append(
                ("numeric", numeric_pipeline, numeric_columns)
            )
        if categorical_columns:
            transformers.append(
                ("categorical", categorical_pipeline, categorical_columns)
            )

        preprocessing = ColumnTransformer(transformers=transformers)

        model_pipeline = Pipeline([
            ("preprocessing", preprocessing),
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                class_weight="balanced"
            ))
        ])

        # Use a test split only when each class has enough examples.
        class_counts = y.value_counts()
        can_split = (
            len(data) >= 10
            and class_counts.min() >= 2
        )

        if can_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            model_pipeline.fit(X_train, y_train)
            score_accuracy = round(
                accuracy_score(y_test, model_pipeline.predict(X_test))
                * 100, 1
            )
        else:
            model_pipeline.fit(X, y)
            score_accuracy = None

        model = model_pipeline
        error_message = None

    except Exception as exc:
        model = None
        error_message = f"Could not train the model: {exc}"


train_model()


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    submitted = {}

    if request.method == "POST":
        submitted = request.form.to_dict()

        if model is None:
            result = {
                "error": error_message or "The prediction model is not ready."
            }
        else:
            try:
                row = {}
                for column in feature_columns:
                    value = submitted.get(column, "").strip()
                    if value == "":
                        row[column] = None
                    else:
                        row[column] = value

                input_data = pd.DataFrame([row])

                # Convert fields to numeric when the training data used numbers.
                for column in feature_columns:
                    if column in input_data.columns:
                        input_data[column] = pd.to_numeric(
                            input_data[column], errors="coerce"
                        )

                prediction = model.predict(input_data)[0]

                result = {
                    "prediction": str(prediction),
                    "accuracy": score_accuracy
                }
            except Exception as exc:
                result = {
                    "error": (
                        "Prediction failed. Check that all input fields "
                        f"match the dataset. Details: {exc}"
                    )
                }

    return render_template(
        "index.html",
        columns=feature_columns,
        result=result,
        submitted=submitted,
        error_message=error_message
    )


if __name__ == "__main__":
    app.run(debug=True)
