"""Data contract for customer feature records.

The contract is the single source of truth for which columns the model needs,
their types, allowed categories and valid ranges. Every entry point (training,
single prediction, batch upload, API) validates through ``validate_frame`` so
malformed input is rejected with an actionable message instead of being
silently mis-encoded by the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0.0"
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

YES_NO = ("Yes", "No")
NO_INTERNET = "No internet service"
NO_PHONE = "No phone service"


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    kind: str  # "category" | "number"
    allowed: tuple[Any, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    nullable: bool = False
    description: str = ""


INTERNET_ADDON = (*YES_NO, NO_INTERNET)

FIELDS: tuple[Field, ...] = (
    Field("gender", "Gender", "category", ("Female", "Male"),
          description="Recorded for fairness checks; review before use in targeting."),
    Field("SeniorCitizen", "Senior citizen", "category", (0, 1),
          description="1 if the customer is a senior citizen."),
    Field("Partner", "Has partner", "category", YES_NO),
    Field("Dependents", "Has dependents", "category", YES_NO),
    Field("tenure", "Tenure (months)", "number", min_value=0, max_value=120,
          description="Months the customer has been with the company."),
    Field("PhoneService", "Phone service", "category", YES_NO),
    Field("MultipleLines", "Multiple lines", "category", (*YES_NO, NO_PHONE)),
    Field("InternetService", "Internet service", "category", ("DSL", "Fiber optic", "No")),
    Field("OnlineSecurity", "Online security", "category", INTERNET_ADDON),
    Field("OnlineBackup", "Online backup", "category", INTERNET_ADDON),
    Field("DeviceProtection", "Device protection", "category", INTERNET_ADDON),
    Field("TechSupport", "Tech support", "category", INTERNET_ADDON),
    Field("StreamingTV", "Streaming TV", "category", INTERNET_ADDON),
    Field("StreamingMovies", "Streaming movies", "category", INTERNET_ADDON),
    Field("Contract", "Contract", "category", ("Month-to-month", "One year", "Two year")),
    Field("PaperlessBilling", "Paperless billing", "category", YES_NO),
    Field("PaymentMethod", "Payment method", "category",
          ("Electronic check", "Mailed check", "Bank transfer (automatic)",
           "Credit card (automatic)")),
    Field("MonthlyCharges", "Monthly charges", "number", min_value=0, max_value=1000),
    Field("TotalCharges", "Total charges", "number", min_value=0, max_value=200_000,
          nullable=True,
          description="Blank for brand-new customers; imputed by the model pipeline."),
)

FIELD_BY_NAME = {f.name: f for f in FIELDS}
FEATURE_COLUMNS = [f.name for f in FIELDS]
CATEGORICAL_COLUMNS = [f.name for f in FIELDS if f.kind == "category"]
NUMERIC_COLUMNS = [f.name for f in FIELDS if f.kind == "number"]
INTERNET_ADDON_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
]

# Shorthand values seen in hand-made CSVs, mapped to the canonical category.
VALUE_ALIASES = {
    "PaymentMethod": {
        "bank transfer": "Bank transfer (automatic)",
        "credit card": "Credit card (automatic)",
    },
    "SeniorCitizen": {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0},
}


class SchemaError(ValueError):
    """The input cannot be validated at all (e.g. required columns missing)."""


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_CANONICAL_KEYS = {_norm_key(c): c for c in [*FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN]}
_CANONICAL_KEYS.update({"customerid": ID_COLUMN, "id": ID_COLUMN, "tenuremonths": "tenure"})


@dataclass
class ColumnMapping:
    mapped: dict[str, str] = field(default_factory=dict)  # source -> canonical
    missing: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    duplicated_targets: list[str] = field(default_factory=list)


def map_columns(columns: list[str]) -> ColumnMapping:
    """Map user-supplied column names onto the contract (case/punctuation-insensitive)."""
    result = ColumnMapping()
    seen: set[str] = set()
    for col in columns:
        canonical = _CANONICAL_KEYS.get(_norm_key(col))
        if canonical is None:
            result.ignored.append(col)
        elif canonical in seen:
            result.duplicated_targets.append(col)
        else:
            seen.add(canonical)
            result.mapped[col] = canonical
    result.missing = [c for c in FEATURE_COLUMNS if c not in seen]
    return result


@dataclass
class ValidationResult:
    valid: pd.DataFrame            # accepted rows, canonical columns + types
    rejected: pd.DataFrame         # original values + row_number + errors
    row_warnings: dict[int, list[str]]  # index in `valid` -> warnings
    mapping: ColumnMapping
    has_ids: bool

    @property
    def summary(self) -> dict[str, Any]:
        total = len(self.valid) + len(self.rejected)
        return {
            "rows_total": total,
            "rows_accepted": len(self.valid),
            "rows_rejected": len(self.rejected),
            "rows_with_warnings": sum(1 for w in self.row_warnings.values() if w),
            "ignored_columns": self.mapping.ignored,
            "renamed_columns": {k: v for k, v in self.mapping.mapped.items() if k != v},
            "schema_version": SCHEMA_VERSION,
        }

    def quality_status(self, idx: Any) -> str:
        return "warning" if self.row_warnings.get(idx) else "passed"


def _coerce_category(f: Field, raw: Any) -> tuple[Any, str | None]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None, f"{f.name} is missing"
    text = str(raw).strip()
    if text == "":
        return None, f"{f.name} is missing"
    if f.name == "SeniorCitizen":
        alias = VALUE_ALIASES["SeniorCitizen"].get(text.lower())
        if alias is None:
            try:
                alias = int(float(text))
            except ValueError:
                alias = None
        if alias in (0, 1):
            return alias, None
        return None, f"SeniorCitizen must be 0 or 1 (got '{text}')"
    for option in f.allowed or ():
        if text.lower() == str(option).lower():
            return option, None
    alias = VALUE_ALIASES.get(f.name, {}).get(text.lower())
    if alias is not None:
        return alias, None
    allowed = ", ".join(f"'{a}'" for a in f.allowed or ())
    return None, f"{f.name} has unknown value '{text}' (allowed: {allowed})"


def _coerce_number(f: Field, raw: Any) -> tuple[Any, str | None, str | None]:
    """Returns (value, error, warning)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() == "":
        if f.nullable:
            return np.nan, None, f"{f.name} is blank and will be imputed"
        return None, f"{f.name} is missing", None
    try:
        value = float(str(raw).strip().replace(",", ""))
    except ValueError:
        return None, f"{f.name} must be numeric (got '{raw}')", None
    if not np.isfinite(value):
        return None, f"{f.name} must be a finite number", None
    if f.min_value is not None and value < f.min_value:
        return None, f"{f.name}={value:g} is below the minimum {f.min_value:g}", None
    if f.max_value is not None and value > f.max_value:
        return None, f"{f.name}={value:g} is above the maximum {f.max_value:g}", None
    return value, None, None


def _consistency_checks(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if row.get("InternetService") == "No":
        bad = [c for c in INTERNET_ADDON_COLUMNS if row.get(c) not in (None, NO_INTERNET)]
        if bad:
            errors.append(
                f"InternetService is 'No' but {', '.join(bad)} not '{NO_INTERNET}'")
    elif row.get("InternetService") in ("DSL", "Fiber optic"):
        bad = [c for c in INTERNET_ADDON_COLUMNS if row.get(c) == NO_INTERNET]
        if bad:
            errors.append(f"Customer has internet but {', '.join(bad)} says '{NO_INTERNET}'")
    if row.get("PhoneService") == "No" and row.get("MultipleLines") not in (None, NO_PHONE):
        errors.append(f"PhoneService is 'No' but MultipleLines is not '{NO_PHONE}'")
    if row.get("PhoneService") == "Yes" and row.get("MultipleLines") == NO_PHONE:
        errors.append(f"PhoneService is 'Yes' but MultipleLines says '{NO_PHONE}'")

    tenure, monthly, total = row.get("tenure"), row.get("MonthlyCharges"), row.get("TotalCharges")
    if tenure is not None and monthly is not None and total is not None and not np.isnan(total):
        if tenure > 0 and monthly > 0:
            ratio = total / (tenure * monthly)
            if ratio < 0.4 or ratio > 2.5:
                warnings.append(
                    "TotalCharges is inconsistent with tenure x MonthlyCharges "
                    f"(ratio {ratio:.2f})")
        if tenure == 0 and total > monthly * 1.5:
            warnings.append("tenure is 0 but TotalCharges exceeds one month of charges")
    return errors, warnings


def validate_frame(df: pd.DataFrame, *, require_ids: bool = False) -> ValidationResult:
    """Validate and canonicalise a frame of customer records.

    Raises ``SchemaError`` only when the file as a whole is unusable; otherwise
    bad rows are moved to ``rejected`` with a readable error list.
    """
    if df is None or len(df.columns) == 0:
        raise SchemaError("The input has no columns.")
    mapping = map_columns(list(df.columns))
    if mapping.missing:
        raise SchemaError(
            "Missing required column(s): " + ", ".join(mapping.missing)
            + ". Column names are matched case-insensitively, ignoring spaces and underscores.")
    has_ids = ID_COLUMN in mapping.mapped.values()
    if require_ids and not has_ids:
        raise SchemaError(f"A '{ID_COLUMN}' column is required for this operation.")

    renamed = df.rename(columns=mapping.mapped)
    keep = [c for c in [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN] if c in renamed.columns]
    renamed = renamed[keep]

    valid_rows: list[dict[str, Any]] = []
    valid_index: list[Any] = []
    warnings_by_row: dict[int, list[str]] = {}
    rejected_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    records = renamed.to_dict(orient="records")
    for position, (idx, raw) in enumerate(zip(renamed.index, records, strict=True)):
        errors: list[str] = []
        warnings: list[str] = []
        clean: dict[str, Any] = {}

        if has_ids:
            cid = raw.get(ID_COLUMN)
            cid = "" if cid is None or (isinstance(cid, float) and np.isnan(cid)) else str(cid).strip()
            if not cid:
                errors.append(f"{ID_COLUMN} is blank")
            elif cid in seen_ids:
                errors.append(f"duplicate {ID_COLUMN} '{cid}' (first occurrence kept)")
            else:
                seen_ids.add(cid)
            clean[ID_COLUMN] = cid

        for f in FIELDS:
            raw_value = raw.get(f.name)
            if f.kind == "category":
                value, err = _coerce_category(f, raw_value)
                if err:
                    errors.append(err)
            else:
                value, err, warn = _coerce_number(f, raw_value)
                if err:
                    errors.append(err)
                if warn:
                    warnings.append(warn)
            clean[f.name] = value

        if not errors:
            c_err, c_warn = _consistency_checks(clean)
            errors.extend(c_err)
            warnings.extend(c_warn)

        if TARGET_COLUMN in renamed.columns:
            label = str(raw.get(TARGET_COLUMN, "")).strip().lower()
            clean[TARGET_COLUMN] = {"yes": 1, "1": 1, "no": 0, "0": 0}.get(label, np.nan)

        if errors:
            rejected = {"row_number": position + 2}  # +1 header, +1 one-based
            rejected.update({k: raw.get(k) for k in keep})
            rejected["errors"] = "; ".join(errors)
            rejected_rows.append(rejected)
        else:
            warnings_by_row[len(valid_rows)] = warnings
            valid_rows.append(clean)
            valid_index.append(idx)

    columns = ([ID_COLUMN] if has_ids else []) + FEATURE_COLUMNS
    if TARGET_COLUMN in renamed.columns:
        columns.append(TARGET_COLUMN)
    valid = pd.DataFrame(valid_rows, columns=columns)
    if len(valid):
        for c in NUMERIC_COLUMNS:
            valid[c] = valid[c].astype(float)
        valid["SeniorCitizen"] = valid["SeniorCitizen"].astype(int)
    valid.index = pd.RangeIndex(len(valid))
    rejected_df = pd.DataFrame(rejected_rows)
    return ValidationResult(valid, rejected_df, warnings_by_row, mapping, has_ids)


def data_dictionary() -> pd.DataFrame:
    rows = []
    for f in FIELDS:
        if f.kind == "category":
            domain = ", ".join(str(a) for a in f.allowed or ())
        else:
            domain = f"{f.min_value:g} - {f.max_value:g}" + (" (blank allowed)" if f.nullable else "")
        rows.append({"column": f.name, "label": f.label, "type": f.kind,
                     "allowed values / range": domain, "notes": f.description})
    return pd.DataFrame(rows)
