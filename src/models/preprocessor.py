from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.features.feature_store import NUMERIC_FEATURES, CATEGORICAL_FEATURES

# Explicit order matters — OrdinalEncoder will assign 0,1,2,3 in this sequence
TENURE_BUCKET_ORDER = [["new", "developing", "established", "loyal"]]

def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    ordinal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=TENURE_BUCKET_ORDER,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    nominal_features = [f for f in CATEGORICAL_FEATURES if f != "tenure_bucket"]
    nominal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("ord", ordinal_transformer, ["tenure_bucket"]),
        ("nom", nominal_transformer, nominal_features),
    ])