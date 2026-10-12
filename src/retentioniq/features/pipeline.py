"""Shared preprocessing. The *same fitted object* is used at train and inference
time because it is persisted inside the model pipeline."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retentioniq.data.schema import CATEGORICAL_COLUMNS, FIELD_BY_NAME, NUMERIC_COLUMNS

# SeniorCitizen is a 0/1 category in the contract; encode it as a number.
NUMERIC_MODEL_COLUMNS = [*NUMERIC_COLUMNS, "SeniorCitizen"]
CATEGORICAL_MODEL_COLUMNS = [c for c in CATEGORICAL_COLUMNS if c != "SeniorCitizen"]


def build_preprocessor() -> ColumnTransformer:
    categories = [list(FIELD_BY_NAME[c].allowed) for c in CATEGORICAL_MODEL_COLUMNS]
    return ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC_MODEL_COLUMNS),
            ("cat", OneHotEncoder(categories=categories, handle_unknown="ignore",
                                  sparse_output=False), CATEGORICAL_MODEL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_to_source_index() -> list[str]:
    """Source column for each transformed column, in output order."""
    mapping = list(NUMERIC_MODEL_COLUMNS)
    for col in CATEGORICAL_MODEL_COLUMNS:
        mapping.extend([col] * len(FIELD_BY_NAME[col].allowed))
    return mapping
