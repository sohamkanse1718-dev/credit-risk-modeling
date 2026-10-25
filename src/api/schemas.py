"""Pydantic contracts for the prediction API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


# Must stay aligned with ``model_metadata.json`` produced by the training job.
MODEL_INPUT_COLUMNS = (
    "loan_amnt", "term", "annual_inc", "dti", "delinq_2yrs", "fico_range_low",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "collections_12_mths_ex_med", "acc_now_delinq", "tot_coll_amt", "total_rev_hi_lim",
    "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd",
    "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_tl_120dpd_2m", "num_tl_30dpd",
    "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort",
    "total_bc_limit", "total_il_high_credit_limit", "mths_since_last_record_missing",
    "mths_since_recent_bc_dlq_missing", "mths_since_last_major_derog_missing",
    "mths_since_recent_revol_delinq_missing", "mths_since_last_delinq_missing",
    "emp_length_missing", "credit_hist_years", "loan_to_income", "history_block_flag",
    "emp_length", "mths_since_last_record", "mths_since_recent_bc_dlq",
    "mths_since_last_major_derog", "mths_since_recent_revol_delinq",
    "mths_since_last_delinq", "home_ownership", "verification_status", "purpose", "zip_code",
)
Numeric: TypeAlias = float
OptionalNumeric: TypeAlias = float | None
Category: TypeAlias = str


class ApplicantInput(BaseModel):
    """Raw applicant fields; engineered fields are computed by the API."""

    model_config = ConfigDict(extra="forbid")

    issue_d: datetime = Field(description="Loan issue/application date used for credit-history length.")
    earliest_cr_line: datetime = Field(description="Earliest credit-line date used for credit-history length.")
    loan_amnt: Numeric
    term: str | int = Field(description="Raw Lending Club term, for example ' 36 months' or 36.")
    annual_inc: Numeric
    dti: Numeric
    delinq_2yrs: Numeric
    fico_range_low: Numeric
    inq_last_6mths: Numeric
    open_acc: Numeric
    pub_rec: Numeric
    revol_bal: Numeric
    revol_util: Numeric
    total_acc: Numeric
    collections_12_mths_ex_med: Numeric
    acc_now_delinq: Numeric
    tot_coll_amt: Numeric
    total_rev_hi_lim: Numeric
    acc_open_past_24mths: Numeric
    avg_cur_bal: Numeric
    bc_open_to_buy: Numeric
    bc_util: Numeric
    chargeoff_within_12_mths: Numeric
    delinq_amnt: Numeric
    mo_sin_old_il_acct: Numeric
    mo_sin_old_rev_tl_op: Numeric
    mo_sin_rcnt_rev_tl_op: Numeric
    mo_sin_rcnt_tl: Numeric
    mort_acc: Numeric
    mths_since_recent_bc: Numeric
    mths_since_recent_inq: Numeric
    num_accts_ever_120_pd: Numeric
    num_actv_bc_tl: Numeric
    num_actv_rev_tl: Numeric
    num_bc_sats: Numeric
    num_bc_tl: Numeric
    num_il_tl: Numeric
    num_op_rev_tl: Numeric
    num_rev_accts: Numeric
    num_tl_120dpd_2m: Numeric
    num_tl_30dpd: Numeric
    num_tl_90g_dpd_24m: Numeric
    num_tl_op_past_12m: Numeric
    pct_tl_nvr_dlq: Numeric
    percent_bc_gt_75: Numeric
    pub_rec_bankruptcies: Numeric
    tax_liens: Numeric
    tot_hi_cred_lim: Numeric
    total_bal_ex_mort: Numeric
    total_bc_limit: Numeric
    total_il_high_credit_limit: Numeric
    emp_length: str | int = Field(description="Raw employment length, for example '10+ years'; cleaned 0-10 values are also accepted.")
    mths_since_last_record: OptionalNumeric
    mths_since_recent_bc_dlq: OptionalNumeric
    mths_since_last_major_derog: OptionalNumeric
    mths_since_recent_revol_delinq: OptionalNumeric
    mths_since_last_delinq: OptionalNumeric
    home_ownership: Category
    verification_status: Category
    purpose: Category
    zip_code: Category


class PredictionResponse(BaseModel):
    default_probability_percent: float
    flag_as_default: bool
    risk_category: Literal["low", "medium", "high"]
    risk_range_percent: str
    decision: Literal["not_flagged", "flagged_for_review"]
    threshold_percent: float
    model_version: str
    shap_top_features: list["ShapFeatureContribution"]


class ShapFeatureContribution(BaseModel):
    """A native XGBoost TreeSHAP contribution for one transformed feature."""

    feature: str
    shap_value: float
    impact_percent: float
    effect: Literal["increases_default_risk", "reduces_default_risk"]


class BatchPredictionRequest(BaseModel):
    applicants: list[ApplicantInput] = Field(..., min_length=1)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ServiceInfoResponse(BaseModel):
    service: str
    message: str
    docs_url: str
    health_url: str


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    model_name: str
    threshold: float
    cost_ratio: float
    validation_metrics: dict[str, Any]
    input_columns: list[str]
    risk_categories: dict[str, str]
