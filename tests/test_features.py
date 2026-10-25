import numpy as np
import pandas as pd

from src.features import (
    clean_dti,
    credit_hist_years,
    loan_to_income,
    term_to_months,
    zip_code_to_first_digit,
)


def test_negative_dti_becomes_nan() -> None:
    result = clean_dti(pd.Series([-1, 12.5, np.nan]))
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 12.5
    assert np.isnan(result.iloc[2])


def test_term_and_zip_transformations_match_notebook_rules() -> None:
    assert term_to_months(pd.Series([" 36 months", "60 months"])).tolist() == [36, 60]
    assert zip_code_to_first_digit(pd.Series(["941xx", " 021xx", None])).tolist() == ["9", "Unknown", "Unknown"]


def test_credit_history_and_loan_to_income() -> None:
    history = credit_hist_years(
        pd.Series(pd.to_datetime(["2020-01-01"])),
        pd.Series(pd.to_datetime(["2010-01-01"])),
    )
    assert history.iloc[0] == 3652 / 365.25
    ratio = loan_to_income(pd.Series([10_000.0]), pd.Series([50_000.0]))
    assert ratio.iloc[0] == 0.2
