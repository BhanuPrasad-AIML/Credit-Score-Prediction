
import os
import pandas as pd

from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "credit_data.csv")

# Possible target column names
TARGET_COLUMNS = [
    "Credit_Score",
    "Credit Score",
    "credit_score",
    "CreditScore",
    "Credit_Category",
    "Credit_Category_Label",
]


def train_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "credit_data.csv is missing. Put it beside app.py."
        )

    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.astype(str).str.strip()
    data = data.dropna(axis=1, how="all")

    if data.empty or len(data.columns) < 2:
        raise ValueError(
            "Your CSV must contain data and at least two columns."
        )

    # Find the prediction target
    target = next(
        (col for col in TARGET_COLUMNS if col in data.columns),
        data.columns[-1]
    )

    data = data.dropna(subset=[target]).copy()
    data[target] = data[target].astype(str).str.strip()
    data = data[data[target] != ""]

    # Remove common ID fields from model inputs
    excluded = {
        "id", "customer_id", "customerid", "name",
        "ssn", "social_security_number", "index"
    }

    features = [
        col for col in data.columns
        if col != target
        and col.lower().replace(" ", "_") not in excluded
        and not col.lower().startswith("unnamed:")
        and not col.lower().endswith("_id")
    ]

    if not features:
        raise ValueError("No usable input columns were found.")

    X = data[features].copy()
    y = data[target]

    # Convert numeric-looking text to numeric values
    for col in X.columns:
        if X[col].dtype == "object":
            converted = pd.to_numeric(X[col], errors="coerce")
            if converted.notna().mean() >= 0.8:
                X[col] = converted

    numeric = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical = [
        col for col in X.columns if col not in numeric
    ]

    transformers = []

    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric
        ))

    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent"
                )),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore"
                ))
            ]),
            categorical
        ))

    preprocessing = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", DecisionTreeClassifier(
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X, y)

    # Generate fields for the prediction form
    fields = []

    for col in features:
        if col in numeric:
            values = X[col].dropna()
            default = (
                float(values.median())
                if not values.empty else 0
            )

            fields.append({
                "name": col,
                "label": col.replace("_", " "),
                "type": "number",
                "default": default,
                "options": []
            })
        else:
            values = X[col].dropna().astype(str).str.strip()
            counts = values.value_counts()
            default = (
                str(counts.index[0])
                if not counts.empty else ""
            )

            options = (
                counts.index.tolist()
                if 1 < len(counts) <= 30 else []
            )

            fields.append({
                "name": col,
                "label": col.replace("_", " "),
                "type": "select" if options else "text",
                "default": default,
                "options": options
            })

    return model, target, fields, len(data)


try:
    model, target_column, fields, total_rows = train_model()
    startup_error = None
except Exception as exc:
    model = None
    target_column = None
    fields = []
    total_rows = 0
    startup_error = str(exc)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    submitted = {}
    error = startup_error

    if request.method == "POST" and model is not None:
        try:
            values = {}

            for field in fields:
                name = field["name"]
                raw = request.form.get(name, "").strip()
                submitted[name] = raw

                if raw == "":
                    values[name] = None
                elif field["type"] == "number":
                    values[name] = float(raw)
                else:
                    values[name] = raw

            input_data = pd.DataFrame(
                [values],
                columns=[field["name"] for field in fields]
            )

            prediction = str(model.predict(input_data)[0])

        except (ValueError, TypeError):
            error = (
                "Please enter valid values in all fields "
                "and check your numeric inputs."
            )
        except Exception:
            error = (
                "Prediction failed. Check the dataset "
                "and the submitted values."
            )

    return render_template(
        "index.html",
        fields=fields,
        prediction=prediction,
        error=error,
        target_column=target_column,
        row_count=total_rows,
        class_count=(
            len(model.named_steps["classifier"].classes_)
            if model is not None else 0
        ),
        submitted=submitted
    )


if __name__ == "__main__":
    # Debug mode stays off for safer local testing.
    app.run(host="127.0.0.1", port=5000, debug=False)
