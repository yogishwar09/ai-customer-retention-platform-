"""Typed request/response contracts (v1)."""


from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
Addon = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "customerID": "DEMO-0001", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 84.5, "TotalCharges": 250.3}})

    customerID: str | None = Field(None, max_length=64, description="Pseudonymous customer id")
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Addon
    OnlineBackup: Addon
    DeviceProtection: Addon
    TechSupport: Addon
    StreamingTV: Addon
    StreamingMovies: Addon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)",
                           "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float | None = Field(None, ge=0, le=200_000)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[CustomerFeatures] = Field(min_length=1)
    persist: bool = False


class Explanation(BaseModel):
    feature: str
    label: str
    value: str | float | int | None
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]
    reason: str


class DataQuality(BaseModel):
    status: Literal["passed", "warning"]
    warnings: list[str]


class RecommendationOut(BaseModel):
    code: str
    name: str
    description: str
    channel: str
    requires_approval: bool
    expected_value: float
    assumptions: dict
    eligible_alternatives: list[dict]
    policy_version: str
    rationale: str


class PredictionResponse(BaseModel):
    prediction_id: str
    customer_id: str | None
    model_version: str
    scored_at: str
    churn_probability: float
    risk_tier: Literal["low", "medium", "high"]
    above_operating_threshold: bool
    borderline: bool
    revenue_at_risk_estimate: float
    explanation_method: str
    explanations: list[Explanation]
    recommendation: RecommendationOut
    data_quality: DataQuality


class RejectedRow(BaseModel):
    row_number: int
    errors: str


class BatchResponse(BaseModel):
    model_version: str
    rows_total: int
    rows_accepted: int
    rows_rejected: int
    latency_ms: float
    predictions: list[PredictionResponse]
    rejected: list[RejectedRow]


class ModelInfo(BaseModel):
    model_version: str
    algorithm: str
    trained_at: str
    schema_version: str
    operating_threshold: float
    risk_tiers: dict
    explanation_method: str
    training_data_fingerprint: str
    test_metrics: dict
    release_gate_passed: bool
    policy_version: str | None
    challenger_version: str | None = None


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: str
    create_action: bool = True
    note: str = Field("", max_length=500)


class ActionOut(BaseModel):
    action_id: str | None
    prediction_id: str
    customer_id: str | None
    experiment_id: str | None = None
    arm: str | None = None
    recommendation: RecommendationOut


class ActionStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["approved", "rejected", "in_progress", "cancelled"]
    note: str = Field("", max_length=500)


class JobOut(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress: float | None = None
    result: dict | None = None
    error: str | None = None


class WebhookCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    url: str = Field(max_length=500)
    event_types: list[str] = Field(min_length=1)


class ExperimentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=3, max_length=160)
    hypothesis: str = Field("", max_length=2000)
    action_code: str | None = None
    control_share: float = Field(0.1, ge=0.01, le=0.5)
    min_detectable_effect: float = Field(0.05, gt=0, lt=0.5)
    alpha: float = Field(0.05, gt=0, lt=0.2)
    power: float = Field(0.8, ge=0.5, lt=1)
    start: bool = False


class AssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=500)


class RetrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tune: bool = False
    use_feedback: bool = True


class ModelStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["staging", "production", "archived", "candidate"]
    reason: str = Field("", max_length=500)


class OutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome_type: Literal["retained", "churned", "no_response", "declined", "opted_out"]
    outcome_window_days: int | None = Field(None, ge=1, le=365)


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    challenger_version: str | None = None
    database: bool
    queue: dict | None = None
    detail: str | None = None
