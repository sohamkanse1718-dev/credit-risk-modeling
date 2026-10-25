"""Stateless feature transformations shared by data preparation and serving."""

from __future__ import annotations

import numpy as np
import pandas as pd


EMP_LENGTH_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}
EVENT_HISTORY_COLUMNS = (
    "mths_since_last_record",
    "mths_since_recent_bc_dlq",
    "mths_since_last_major_derog",
    "mths_since_recent_revol_delinq",
    "mths_since_last_delinq",
)


def term_to_months(term: pd.Series) -> pd.Series:
    """Convert Lending Club terms such as ``" 36 months"`` to month counts."""
    if pd.api.types.is_numeric_dtype(term):
        result = pd.to_numeric(term, errors="coerce")
    else:
        result = pd.to_numeric(term.astype("string").str.extract(r"(\d+)", expand=False), errors="coerce")
    invalid = result.notna() & ~result.isin([36, 60])
    if invalid.any():
        raise ValueError("term must be either 36 or 60 months.")
    return result


def emp_length_to_years(emp_length: pd.Series) -> pd.Series:
    """Convert Lending Club employment strings into the notebook's ordinal years."""
    if pd.api.types.is_numeric_dtype(emp_length):
        return pd.to_numeric(emp_length, errors="coerce")
    unmapped = set(emp_length.dropna().astype(str).unique()) - set(EMP_LENGTH_MAP)
    if unmapped:
        raise ValueError(f"Unsupported emp_length values: {sorted(unmapped)}")
    return emp_length.map(EMP_LENGTH_MAP)


def zip_code_to_first_digit(zip_code: pd.Series) -> pd.Series:
    """Aggregate sparse ZIP codes into their first digit, retaining unknown values."""
    return (
        zip_code.astype("string")
        .str.extract(r"^(\d)", expand=False)
        .fillna("Unknown")
        .astype("string")
    )


def clean_dti(dti: pd.Series) -> pd.Series:
    """Treat invalid negative debt-to-income values as missing."""
    numeric_dti = pd.to_numeric(dti, errors="coerce")
    return numeric_dti.mask(numeric_dti < 0, np.nan)


def credit_hist_years(issue_d: pd.Series, earliest_cr_line: pd.Series) -> pd.Series:
    """Return credit-history length in years using the notebook's 365.25-day rule."""
    return (pd.to_datetime(issue_d) - pd.to_datetime(earliest_cr_line)).dt.days / 365.25


def loan_to_income(loan_amnt: pd.Series, annual_inc: pd.Series) -> pd.Series:
    """Return the loan amount divided by annual income."""
    return pd.to_numeric(loan_amnt, errors="coerce") / pd.to_numeric(annual_inc, errors="coerce")


def engineer_application_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's deterministic cleaning and feature rules.

    The function is intentionally idempotent for already-cleaned data, which lets
    callers use it safely before fitting or serving a model.
    """
    result = frame.copy()
    if "term" in result:
        result["term"] = term_to_months(result["term"])
    if "zip_code" in result:
        result["zip_code"] = zip_code_to_first_digit(result["zip_code"])
    if "dti" in result:
        result["dti"] = clean_dti(result["dti"])
    if {"issue_d", "earliest_cr_line"}.issubset(result.columns):
        result["credit_hist_years"] = credit_hist_years(
            result["issue_d"], result["earliest_cr_line"]
        )
    if {"loan_amnt", "annual_inc"}.issubset(result.columns):
        result["loan_to_income"] = loan_to_income(result["loan_amnt"], result["annual_inc"])
    # Derive missingness flags only for raw requests. Processed training rows
    # already contain these flags and must retain their original values.
    for column in (*EVENT_HISTORY_COLUMNS, "emp_length"):
        indicator = f"{column}_missing"
        if column in result and indicator not in result:
            result[indicator] = result[column].isna().astype("int8")
    if "emp_length" in result:
        result["emp_length"] = emp_length_to_years(result["emp_length"])
    # Event-history nulls use the notebook's -1 sentinel after their flags are
    # recorded. This keeps raw serving input aligned with processed training data.
    for column in EVENT_HISTORY_COLUMNS:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(-1)
    # This is the missing newer-credit-bureau-field block identified in the
    # notebooks. ``tot_coll_amt`` is representative of that block.
    if "tot_coll_amt" in result and "history_block_flag" not in result:
        result["history_block_flag"] = result["tot_coll_amt"].isna().astype("int8")
    return result
